import pickle
import os
import time
import argparse
import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from model import *
from utils import *
from metrics import *
import config
# from config import device
from visualization import Visualizer
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

torch.backends.cudnn.enabled=False

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='eth',
                    help='eth,hotel,univ,zara1,zara2')
parser.add_argument('--load_pre', action='store_true', help="load pretrain weight")
parser.add_argument('--world_size', type=int, default=2)

args = parser.parse_args()

model_name = config.model_name
max_node_num =  config.max_node_num
mask = config.mask
distributed = len(config.device_ids) > 1

vis_name = f"{config.vis_name}_{config.model_name}_{args.dataset}"
vis = Visualizer(env=vis_name)

print('*'*30)
print("Training initiating....")

#Data prep
obs_seq_len = config.obs_seq_len
pred_seq_len = config.pred_seq_len
dataset = args.dataset
data_set = './datasets/'+dataset+'/'
loader_batchsize = config.loader_size
if loader_batchsize == 0:
    loader_batchsize = config.batch_size
cache_name = config.cache_name

if distributed:
    dist.init_process_group("nccl", init_method="env://")
    print('distributed.')

dset_train = TrajectoryDataset(
        data_set+'train/',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1,
        norm_lap_matr=False,
        delim='\t',
        max_nodes=max_node_num,
        test='train',
        dataset=dataset,cache_name_=cache_name)

if distributed:
    world_size = args.world_size
    assert loader_batchsize % world_size == 0
    loader_batchsize = loader_batchsize // world_size
    train_sampler = DistributedSampler(dset_train)
    loader_train = DataLoader(dset_train, sampler=train_sampler, 
        batch_size=loader_batchsize, num_workers=1)
else:
    loader_train = DataLoader(
            dset_train,
            batch_size=loader_batchsize, #This is irrelative to the args batch size parameter
            shuffle=True,
            num_workers=8)

vald = True
if vald:
    dset_val = TrajectoryDataset(
            data_set+'val/',
            obs_len=obs_seq_len,
            pred_len=pred_seq_len,
            skip=1,
            norm_lap_matr=False, 
            delim='\t', 
            max_nodes=max_node_num, 
            test='val',
            dataset=dataset,cache_name_=cache_name)
    if distributed:
        val_sampler = DistributedSampler(dset_val)
        loader_val = DataLoader(dset_val, sampler=val_sampler, 
            batch_size=loader_batchsize, shuffle=False, num_workers=1)
    else:
        loader_val = DataLoader(
                dset_val,
                batch_size=loader_batchsize, #This is irrelative to the args batch size parameter
                shuffle =False,
                num_workers=1)

# model
generator, discriminator = get_model(model_name)
print(generator)
print(discriminator)

device_count = len(config.device_ids)
print(config.cuda)
rank = -1
if device_count > 1:
    rank = dist.get_rank()
    device_id = config.device_ids[rank % device_count]
    print(f"Running DDP on rank {rank}, device {device_id}.")
    device = torch.device('cuda', device_id)
    generator = generator.to(device_id)
    discriminator = discriminator.to(device_id)
    generator = DDP(generator, device_ids=[device_id],find_unused_parameters=False)
    discriminator = DDP(discriminator, device_ids=[device_id])
else:
    device = config.device
    generator.to(device)
    discriminator.to(device)

# optimizer
pretrain_optimizer = optim.RMSprop(generator.parameters(), lr=config.pretrain_learning_rate,weight_decay=0)
generator_optimizer = optim.RMSprop(generator.parameters(), lr=config.g_learning_rate,weight_decay=1e-4, alpha=0.999)
discriminator_optimizer = optim.RMSprop(discriminator.parameters(), lr=config.d_learning_rate, alpha=0.999)

# loss
mse_criterion = nn.MSELoss(reduction='mean')

# lr scheduler
lr_scheduler_on = config.use_lrschd != ""
if lr_scheduler_on:
    lrschd = config.use_lrschd
    pretrain_scheduler = optim.lr_scheduler.StepLR(pretrain_optimizer, step_size=5, gamma=0.7)
    if lrschd.startswith('exp'):
        gamma = 0.8 if len(lrschd) == 3 else float(lrschd[3:])/10.0
        g_scheduler = torch.optim.lr_scheduler.ExponentialLR(generator_optimizer, gamma=gamma, last_epoch=-1)
        d_scheduler = torch.optim.lr_scheduler.ExponentialLR(discriminator_optimizer, gamma=gamma, last_epoch=-1)
    elif lrschd.startswith('step'):
        gamma = 0.8 if len(lrschd) == 4 else float(lrschd[4:])/10.0
        g_scheduler = torch.optim.lr_scheduler.StepLR(generator_optimizer, step_size=5, gamma=gamma)
        d_scheduler = torch.optim.lr_scheduler.StepLR(discriminator_optimizer, step_size=6, gamma=gamma)
    elif lrschd == 'ronp':
        g_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(generator_optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7)
        d_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(discriminator_optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7)
    else:
        print(f'Not supported lr scheduler: {lrschd}')
        sys.exit(1)

checkpoint_dir = './checkpoint/'+'%s-%s/' % (model_name, dataset)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

print('Data and model loaded')
print('Checkpoint dir:', checkpoint_dir)


def get_pos_loss_mse(pred_, gt_):
    """
    :param V_tr, shape (batch, seq, node, 2)
    """
    loss1 = mse_criterion(pred_[...,:2], gt_[...,:2])
    loss = loss1

    return loss

def get_pos_loss_k(pred_, gt_, loss_mask_, mode='avg',mask_loss=True):
    """
    :param loss_mask: Tensor of shape (batch, node, seq)
    """
    # batch, seq_len, node_num, _ = pred_.shape
    if mask_loss:
        loss = (loss_mask_.permute(0,2,1).unsqueeze(3)*(pred_-gt_)**2)
    else:
        loss = (pred_-gt_)**2
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'avg':
        return torch.sum(loss) / torch.sum(loss_mask_.data)
    elif mode == 'raw':
        return loss.sum(dim=3).sum(dim=2)

        
loss_type = config.loss_type
if loss_type == "biv":
    loss_func = bivariate_loss
elif loss_type == "mse":
    loss_func = get_pos_loss_mse
else:
    print(f"loss_type '{loss_type}' not supported !!!")
    sys.exit(1)


def graph_loss(pred_, gt_, *args):
    loss = loss_func(pred_, gt_)
    return loss

def eval_pred_convert(x, *args):
    return x

def disc_pred_convert(x, *args):
    return x

def get_pred_bv(V_pred, gt_, ksteps=20):
    sx = torch.exp(V_pred[...,2])  # sx
    sy = torch.exp(V_pred[...,3])  # sy
    corr = torch.tanh(V_pred[...,4])  # corr
    
    cov = torch.zeros(V_pred.shape[0],V_pred.shape[1],V_pred.shape[2],2,2).to(V_pred.device)
    cov[...,0,0]= sx*sx
    cov[...,0,1]= corr*sx*sy
    cov[...,1,0]= corr*sx*sy
    cov[...,1,1]= sy*sy
    mean = V_pred[...,0:2]
    
    mvnormal = torchdist.MultivariateNormal(mean,cov)

    min_loss = 1e16
    min_pred = None
    for _ in range(ksteps):
        pred_ = mvnormal.sample()
        loss_ = torch.mean((pred_-gt_)**2)
        if loss_ < min_loss:
            min_loss = loss_
            min_pred = pred_
    return min_pred

if config.pred_type == "sample":
    eval_pred_convert = get_pred_bv

def evaluate(model, epoch, name_window='validation'):
    total_ade = 0
    total_fde = 0
    total_samples = 0
    total_mse = 0
    total_kl = 0
    total_peds = 0
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader_val):
            batch = [tensor.to(device) for tensor in data]
            V_obs, A_obs, V_tr, loss_mask, obs_traj, num_peds = batch
            obs_mask = loss_mask[...,:obs_seq_len]
            loss_mask = loss_mask[...,obs_seq_len:]

            V_pred = generator(V_obs, A_obs, obs_mask)
            V_pred = V_pred * loss_mask.permute(0,2,1).unsqueeze(3)
            predicted_shot = eval_pred_convert(V_pred, V_tr[...,:2], 10)

            V_x = seq_to_nodes(obs_traj).to(device)
            V_pred_rel_to_abs = nodes_rel_to_node_abs(predicted_shot[...,:2], V_x[:,-1,:,:])
            V_y_rel_to_abs= nodes_rel_to_node_abs(V_tr[...,:2], V_x[:,-1,:,:])
            predicted_shot = V_pred_rel_to_abs
            out_shot = V_y_rel_to_abs

            batch_size =V_obs.size(0)
            total_samples += batch_size
            total_mse += batch_size * MSE(predicted_shot, out_shot)
            total_loss += batch_size * graph_loss(V_pred, V_tr).item()

            predicted_shot1 = predicted_shot.permute(0,2,1,3)
            out_shot1 = out_shot.permute(0,2,1,3)
            predicted_shot1 = predicted_shot1.data.cpu().numpy()
            out_shot1 = out_shot1.data.cpu().numpy()
            num_peds = num_peds.data.cpu().numpy()
            ade_ = ade_sstgcnn(predicted_shot1, out_shot1, num_peds, False)
            fde_ = fde_sstgcnn(predicted_shot1, out_shot1, num_peds, False)
            total_ade += ade_
            total_fde += fde_
            total_peds += sum(num_peds)

    avg_ade = total_ade/total_peds
    avg_fde = total_fde/total_peds
    avg_mse = total_mse/total_samples
    avg_loss = total_loss/total_samples
    print("[Epoch %d] ADE: %.4f, FDE: %.4f, MSE: %.4f, Loss: %.4f" %
        (epoch, avg_ade, avg_fde, avg_mse, avg_loss))
    vis.plot_many_stack(epoch+1, 
        {'ade': avg_ade, 'fde': avg_fde, 'mse': avg_mse}, xlabel='epoch', name_window=name_window)  
    vis.plot_one(epoch+1, avg_loss, 'loss_'+name_window, xlabel='epoch')
    return avg_ade, avg_fde, avg_mse, avg_loss

pre_train_count = 0
train_count = 0

def pre_train_step(data, i, best_k=4, nll=False):
    global pre_train_count
    pre_train_count += 1
    batch = [tensor.to(device) for tensor in data]
    V_obs, A_obs, V_tr, loss_mask, num_peds = batch
    obs_mask = loss_mask[...,:obs_seq_len]
    loss_mask = loss_mask[...,obs_seq_len:]

    if best_k > 0:
        loss_rel = []
        for _ in range(best_k):
            predicted_shot = generator(V_obs, A_obs, obs_mask)
            pos_loss = get_pos_loss_k(predicted_shot[...,:2], V_tr[...,:2], loss_mask, mode='raw')
            loss_rel.append(pos_loss)
        loss_rel = torch.stack(loss_rel, dim=2)
        loss = torch.zeros(1).to(V_tr)
        for b_ in range(V_obs.shape[0]):
            num_ped_ = int(num_peds[b_])
            _loss_rel = loss_rel[b_, 0:num_ped_]
            _loss_rel = torch.sum(_loss_rel, dim=0)
            _loss_rel = torch.min(_loss_rel) / torch.sum(loss_mask[b_, 0:num_ped_])
            loss += _loss_rel
        pos_loss = loss
    else:
        predicted_shot = generator(V_obs, A_obs)
        predicted_shot = predicted_shot * loss_mask.permute(0,2,1).unsqueeze(3)
        pos_loss = graph_loss(predicted_shot, V_tr, pre_train_count, 'pre')
        loss = pos_loss

    print('[epoch %d] [step %d] [pos loss %.4f, loss %.4f]' % (epoch, i, pos_loss.item(), loss.item()))
    return loss

def train_step(data, i, best_k=4, to_device=True):
    global train_count
    train_count += 1
    if to_device:
        batch = [tensor.to(device) for tensor in data]
    else:
        batch = data
    V_obs, A_obs, V_tr, loss_mask, num_peds = batch
    obs_mask = loss_mask[...,:obs_seq_len]
    loss_mask = loss_mask[...,obs_seq_len:]

    if best_k > 0:
        loss_rel = []
        for _ in range(best_k):
            predicted_shot = generator(V_obs, A_obs, obs_mask)
            predicted_shot = predicted_shot * loss_mask.permute(0,2,1).unsqueeze(3)
            pos_loss = get_pos_loss_k(predicted_shot[...,:2], V_tr[...,:2], loss_mask, mode='raw',mask_loss=False)
            loss_rel.append(pos_loss)
        loss_rel = torch.stack(loss_rel, dim=2)
        loss = torch.zeros(1).to(V_tr)
        for b_ in range(V_obs.shape[0]):
            num_ped_ = int(num_peds[b_])
            _loss_rel = loss_rel[b_, 0:num_ped_]
            _loss_rel = torch.sum(_loss_rel, dim=0)
            _loss_rel = torch.min(_loss_rel) / torch.sum(loss_mask[b_, 0:num_ped_])
            loss += _loss_rel
        pos_loss = loss
    else:
        predicted_shot = generator(V_obs, A_obs, obs_mask)
        predicted_shot = predicted_shot * loss_mask.permute(0,2,1).unsqueeze(3)
        pos_loss = graph_loss(predicted_shot, V_tr, train_count, 'train')
    loss = pos_loss
    real_traj = V_tr[...,:2]
    fake_traj = predicted_shot[...,:2]
    real_score = discriminator(real_traj).mean()
    fake_score = discriminator(fake_traj.detach()).mean()

    print('[epoch %d] [step %d] [loss %.4f] [real_score: %.4f, fake_score: %.4f]' % (epoch, i, 
        loss.item(), real_score.item(), fake_score.item()))    
    return real_score, fake_score, fake_traj, loss_mask, loss

def save_model(model, path):
    if distributed and rank != -1:
        if rank == 0:
            torch.save(model.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)


loader_len = len(loader_train)
turn_point = loader_len -1
acc_batchsize = config.batch_size
nll=False
pepoch = config.p_epoch
gepoch = config.g_epoch
acc_train_on = (loader_batchsize == 1)
constant_metrics = {'pretrain':{'min_val_epoch':-1, 'min_val_loss': 1e16},
                    'train':{'min_val_epoch':-1, 'min_val_loss': 1e16}}
if args.load_pre:
    generator.load_state_dict(torch.load(checkpoint_dir+'pretrain_best.pth',map_location=device))
    pepoch = 0

######################### pretrain ########################
step = 0
since = time.time()
for epoch in range(pepoch):
    generator.train()
    if distributed:
        train_sampler.set_epoch(epoch)
    batch_count = 0
    for i, data in enumerate(loader_train):
        batchsize_ = len(data[0])
        batch_count += batchsize_
        loss = pre_train_step(data, i, best_k=config.bestk)

        pretrain_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(generator.parameters(), config.gradient_clip)
        pretrain_optimizer.step()
        step += 1

        vis.plot_one(step, loss.item(), 'pre_loss')
    if vald:
        _ade, _fde, _, _loss = evaluate(generator, epoch, 'pre_validation')
        if _ade < constant_metrics['pretrain']['min_val_loss']:
            constant_metrics['pretrain']['min_val_loss'] = _ade
            constant_metrics['pretrain']['min_val_epoch'] = epoch
            save_model(generator, checkpoint_dir+'pretrain_best.pth')
    if lr_scheduler_on:
        vis.plot_one(epoch+1, pretrain_optimizer.param_groups[0]['lr'], 'pretrain_lr', xlabel='epoch')
        pretrain_scheduler.step()
spend1 = time.time() - since
with open(checkpoint_dir+'constant_metrics.pkl', 'wb') as fp:
        pickle.dump(constant_metrics, fp)  
save_model(generator, checkpoint_dir+'pretrain_final.pth')

######################### train ########################
step = 0
since = time.time()
disc_params = []
for epoch in range(gepoch):
    generator.train()
    batch_count = 0
    if distributed:
        train_sampler.set_epoch(epoch)
    for i, data in enumerate(loader_train):
        batchsize_ = len(data[0])
        batch_count += batchsize_
        step += 1
        data = [tensor.to(device) for tensor in data]

        for _ in range(config.n_critic):
            real_score, fake_score, pred, loss_mask, loss = train_step(data, i, best_k=0, to_device=False)
            d_loss = -real_score + fake_score
            for param in discriminator.parameters():
                param.grad = None
            d_loss.backward()
            discriminator_optimizer.step()
            for p in discriminator.parameters():
                p.data.clamp_(-config.weight_clip, config.weight_clip)
        fake_score1 = discriminator(pred).mean()
        g_loss = -fake_score1 + loss
        for param in generator.parameters():
            param.grad = None
        g_loss.backward()
        generator_optimizer.step()

        vis.log('[epoch %d] [step %d] [real_score %.4f, fake_score %.4f] [d_loss %.4f, g_loss %.4f] [loss %.4f]' % (
            epoch, i, real_score.item(), fake_score.item(), d_loss.item(), g_loss.item(), loss.item()))
        vis.plot_one(step, loss.item(), 'loss')
        vis.plot_many_stack(step, {'d_loss': d_loss.item(), 'g_loss': g_loss.item(),
            'real_score':real_score.item(),'fake_score':fake_score.item()})
    if vald:
        _ade, _fde, _, _loss = evaluate(generator, epoch, 'train_validation')
        if _ade < constant_metrics['train']['min_val_loss']:
            constant_metrics['train']['min_val_loss'] = _ade
            constant_metrics['train']['min_val_epoch'] = epoch
            save_model(generator, checkpoint_dir+'train_best.pth')
    if lr_scheduler_on:
        vis.plot_many_stack(epoch+1, {'g_lr': generator_optimizer.param_groups[0]['lr'], 
            'd_lr': discriminator_optimizer.param_groups[0]['lr']}, xlabel='epoch')
        if config.use_lrschd == 'ronp':
            d_scheduler.step(_ade)
            g_scheduler.step(_ade)
        else:
            d_scheduler.step()
            g_scheduler.step()
spend = time.time() - since
print('finish !!! [peoch: %d, spend: %.3f]\n'
    '[gepoch: %d, spend: %.3f]\n'
    'total spend: %.3f' % (pepoch, spend1, gepoch, spend, spend1+spend))
save_model(generator, os.path.join(checkpoint_dir, '%s-lr%f-pepoch%d-gepoch%d.pth'
                                   % (model_name,
                                      config.d_learning_rate,
                                      pepoch,
                                      gepoch))) 
with open(checkpoint_dir+'constant_metrics.pkl', 'wb') as fp:
    pickle.dump(constant_metrics, fp)
import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import numpy as np
import pickle
# import multiprocessing
# import threading

from utils import *
from metrics import *
import config
from config import device
from model import *
import collections
from thop import profile

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='eth',
                    help='eth,hotel,univ,zara1,zara2')
parser.add_argument('--mw', default='train', help='model path')

args = parser.parse_args()
mw = args.mw

mask = config.mask
model_name = config.model_name
pred_type = config.pred_type

def kstep_test(generator, test_loader, ksteps=20, kde_samples=0,output_path="",n_samples=0, n_amdv_sample=0):
    ade_bigls = []
    fde_bigls = []
    aae_bigls = []
    total_smaple = 0
    total_peds = 0
    inference_time = 0
    sample_list = []
    mabs_loss = []
    eig_collect = []

    generator.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            batch = [tensor.to(device) for tensor in data]
            V_obs, A_obs, V_tr, loss_mask, obs_traj, num_peds = batch
            obs_mask = loss_mask[...,:config.obs_seq_len]
            loss_mask = loss_mask[...,config.obs_seq_len:]
            batch_size = len(V_obs)
            num_of_objs = int(num_peds[0])

            ade_ls = {}
            fde_ls = {}
            aae_ls = {}
            for n in range(num_of_objs):
                ade_ls[n] = []
                fde_ls[n] = []
                aae_ls[n] = []
            V_x = seq_to_nodes(obs_traj).to(device)
            V_y_rel_to_abs = nodes_rel_to_node_abs(V_tr[...,:2], V_x[:,-1,:,:])
            V_y_rel_to_abs = V_y_rel_to_abs.data.cpu().numpy()
            obs_traj = obs_traj.permute(0,1,3,2).data.cpu().numpy()

            # if i == 0:
            #     flops, params = profile(generator, inputs=(V_obs, A_obs))
            #     print("flops:", flops, "params:", params)

            if pred_type == 'sample':
                start = time.time()
                V_pred = generator(V_obs, A_obs, obs_mask if config.attn_mask else None)
                time_s = time.time()-start
                sx = torch.exp(V_pred[...,2])  # sx
                sy = torch.exp(V_pred[...,3])  # sy
                corr = torch.tanh(V_pred[...,4])  #corr
                cov = torch.zeros(V_pred.shape[0],V_pred.shape[1],V_pred.shape[2],2,2).to(V_pred.device)
                cov[...,0,0]= sx*sx
                cov[...,0,1]= corr*sx*sy
                cov[...,1,0]= corr*sx*sy
                cov[...,1,1]= sy*sy
                mean = V_pred[...,0:2]
                mvnormal = torchdist.MultivariateNormal(mean,cov)
            time_ = 0
            for _ in range(ksteps):
                if pred_type == 'sample':
                    V_pred = mvnormal.sample()
                else:
                    start = time.time()
                    V_pred = generator(V_obs, A_obs, obs_mask if config.attn_mask else None)
                    time_ += time.time()-start
                V_pred_rel_to_abs = nodes_rel_to_node_abs(V_pred[...,:2], V_x[:,-1,:,:])
                V_pred_rel_to_abs = V_pred_rel_to_abs.data.cpu().numpy()
                
                for n in range(num_of_objs):
                    pred = []
                    target = []
                    number_of = []
                    pred.append(V_pred_rel_to_abs[:,:,n:n+1,:])
                    target.append(V_y_rel_to_abs[:,:,n:n+1,:])
                    number_of.append(1)
                    ade_ls[n].append(ade_1(pred, target, number_of))
                    fde_ls[n].append(fde_1(pred, target, number_of))
                    aae_ls[n].append(aae_1(V_pred[:,:,n:n+1,:2], V_tr[:,:,n:n+1,:2]))
            for n in range(num_of_objs):
                ade_bigls.append(min(ade_ls[n]))
                fde_bigls.append(min(fde_ls[n]))
                aae_bigls.append(min(aae_ls[n]))
            
            total_smaple += batch_size
            total_peds += sum(num_peds)
            
            if pred_type == 'sample':
                inference_time += time_s
            else:
                inference_time += time_/ksteps

            if n_samples > 0:
                V_obs = V_obs[...,:num_of_objs, :2].squeeze()
                v_obs_rel = V_obs.cumsum(dim=0)
                gt_rel = V_tr[...,:num_of_objs, :2].squeeze().cumsum(dim=0) + v_obs_rel[-1]
                sample_ = {'obs': v_obs_rel, 'gt': gt_rel}
                sample_pred_list = []
                for _ in range(n_samples):
                    sample_pred_ = mvnormal.sample()
                    sample_pred_list.append(sample_pred_[...,:num_of_objs,:2].squeeze().cumsum(dim=0)+v_obs_rel[-1])
                sample_pred = torch.stack(sample_pred_list)  # [sample, seq, node, xy]
                sample_['pred'] = sample_pred
                sample_list.append(sample_)

            if n_amdv_sample > 0:
                amdv_sample_list = []
                for i in range(n_amdv_sample):
                    if pred_type == 'sample':
                        sample_pred_ = mvnormal.sample()
                    else:
                        sample_pred_ = generator(V_obs, A_obs, obs_mask if config.attn_mask else None)
                    sample_pred_abs = nodes_rel_to_node_abs(sample_pred_[...,:2], V_x[:,-1,:,:])
                    amdv_sample_list.append(sample_pred_abs.squeeze(0)[:,:num_of_objs,None,:])
                amdv_samples = torch.cat(amdv_sample_list, dim=2).data.cpu().numpy()
                gt_abs = V_y_rel_to_abs.squeeze(0)[:,:num_of_objs]
                m, _, _, _, eig = calc_amd_amv(gt_abs, amdv_samples)
                mabs_loss.append(m)
                eig_collect.append(eig)
    
    ade_ = sum(ade_bigls)/len(ade_bigls)
    fde_ = sum(fde_bigls)/len(fde_bigls)
    aae_ = sum(aae_bigls)/len(aae_bigls)
    if len(mabs_loss) > 0:
        amd_ = sum(mabs_loss) / len(mabs_loss) 
        amv_ = sum(eig_collect) / len(eig_collect)
    else:
        amd_ = 0.0
        amv_ = 0.0
    if len(sample_list) > 0:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        with open(output_path+'/sample%d.pkl'%n_samples, 'wb') as fp:
            pkl.dump(sample_list, fp)
        print(output_path+'/sample%d.pkl'%n_samples, 'saved')

    return ade_,fde_,aae_,amd_,amv_,inference_time/len(test_loader),


def process(path, testd, ksteps, f, dataset):
    print("*"*50)
    print('Evaluating model:', path)

    if mw == 'pre':
        model_path = path+'/pretrain_best.pth'
    elif mw == 'train':
        model_path = path+'/train_best.pth'
    elif mw == 'pref':
        model_path = path+'/pretrain_final.pth'
    else:
        model_path = path+'/%s-lr%f-pepoch%d-gepoch%d.pth' % (
            model_name, config.d_learning_rate, config.p_epoch, config.g_epoch)
    print('Model', model_path)
    f.write('%s\n' % model_path)

    stats = path+'/constant_metrics.pkl'
    if os.path.exists(stats):
        with open(stats, 'rb') as fp:
            cm = pickle.load(fp)
        print("Stats:", cm)

    obs_seq_len = config.obs_seq_len
    pred_seq_len = config.pred_seq_len
    data_set = './datasets/'+dataset+'/'

    dset_test = TrajectoryDataset(
        data_set + 'test/',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1, 
        norm_lap_matr=True,
        test='test', 
        max_nodes=config.max_node_num, 
        min_ped=1,
        cache_test_name='caches/cache_%s_test_%s.pkl' % (testd, config.cache_name))

    test_loader = DataLoader(
        dset_test,
        batch_size=1,
        shuffle=False,
        num_workers=0)

    generator, _ = get_model(model_name)
    print(generator)
    if len(config.device_ids) > 1:
        device_id = config.device_ids[0]
        generator.to(device_id)
        generator = DDP(generator, device_ids=[device_id])
    else:
        generator.to(device)
    checkpoint = torch.load(model_path, map_location=device)
    generator.load_state_dict(checkpoint)

    ade_,fde_,aae_,amd_,amv_,spend = kstep_test(generator, test_loader, ksteps=ksteps, 
        output_path=f'outputs/{dataset}',n_samples=config.n_samples, n_amdv_sample=config.n_amdv_sample)
    ade_all.append(ade_)
    fde_all.append(fde_)
    aae_all.append(aae_)
    amd_all.append(amd_)
    amv_all.append(amv_)

    outputs = f"{testd}\n"\
    f"ADE: {ade_}, FDE: {fde_}\n"\
    f"AAE: {aae_}\n"\
    f"AMD: {amd_}, AMV: {amv_}\n"\
    f"{mw},{model_name},{config.p_epoch},{config.g_epoch},{dataset},"\
        f"{ade_:.2f},{fde_:.2f},{aae_:.2f},{amd_:.2f},{amv_:.3f},{testd},{ksteps}\n" 

    print(outputs)
    f.write(outputs+'\n')
    # print('best train epoch:', cm['train']['min_val_epoch'])
    print(f'{dataset} spend: {spend}s')
    return spend


if __name__ == '__main__':
    test_datasets = config.datasets or [args.dataset]
    base_path = './checkpoint/{}-{}'

    fn = 'test_%s.txt' % ("_".join(test_datasets))
    fo = open(fn, 'a')
    fo.write('\n\n%s\n' % time.strftime('%Y-%m-%d %H:%M:%S'))
    ade_all = []
    fde_all = []
    aae_all = []
    amd_all = []
    amv_all = []

    total_spend = 0
    for testd in test_datasets:
        path = base_path.format(model_name, testd)
        total_spend += process(path, testd, config.ksteps, fo, testd)
    print("*"*50)
    avg_ade = sum(ade_all)/len(ade_all)
    avg_fde = sum(fde_all)/len(fde_all)
    avg_aae = sum(aae_all)/len(aae_all)
    avg_amd = sum(amd_all)/len(amd_all)
    avg_amv = sum(amv_all)/len(amv_all)
    if len(ade_all) > 1:
        outputs = "Avg\n"\
            f"Avg ADE: {avg_ade}, Avg FDE: {avg_fde}\n"\
            f"Avg AAE: {avg_aae}\n"\
            f"Avg AMD: {avg_amd}, Avg AMV: {avg_amv}\n"\
            f"{mw},{model_name},{config.p_epoch},{config.g_epoch},"\
                f"{avg_ade:.2f},{avg_fde:.2f},{avg_aae:.2f},{avg_amd:.2f},{avg_amv:.3f},{config.ksteps}\n" 
        print(outputs)
        fo.write(outputs)
    fo.close()
    print(f'avg inference time: {total_spend/len(test_datasets):.4f}s')
    
import os
import math
import numpy as np
from tqdm import tqdm
import pickle as pkl
import torch
from torch.utils.data import Dataset

from config import *


def anorm(p1, p2):
    if (np.abs(p1[0]) + np.abs(p1[1])) > 0 and (np.abs(p2[0]) + np.abs(p2[1])) > 0:
        return 1
    return 0

def to_theta(rel):
    """
    rel: ndarray, shape(2,)
    """
    if rel[0] == 0:
        if rel[1] > 0:
            return np.pi/2
        elif rel[1] < 0:
            return -np.pi/2
        else:
            return 0
    else:
        theta_ = np.arctan(float(rel[1]/rel[0]))
        if rel[0] < 0:
            if rel[1] >= 0:
                theta_ = theta_ + np.pi
            else:
                theta_ = theta_ - np.pi
        return theta_

def get_V(seq_, seq_rel):
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]
    V = np.zeros((seq_len, max_nodes, 2))

    for s in range(seq_len):
        step_ = seq_[:, :, s]
        step_rel = seq_rel[:, :, s]
        for h in range(len(step_)):
            V[s, h, :2] = step_rel[h, :2]
    return torch.from_numpy(V).type(torch.float)

def get_A(seq_, seq_rel):
    seq_ = seq_.squeeze()
    seq_rel  = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]
    A = np.zeros((seq_len, max_nodes, max_nodes))

    for s in range(seq_len):
        step_ = seq_[:, :, s]
        step_rel = seq_rel[:, :, s]
        for h in range(len(step_)):
            for k in range(h + 1, len(step_)):
                l2_norm = anorm(step_[h], step_[k])   #基于两点偏移量的距离计算权重
                A[s, h, k] = l2_norm
                A[s, k, h] = l2_norm
    return torch.from_numpy(A).type(torch.float)

def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)

def seq_to_nodes(seq_):
    """
    seq_: shape (batch, node, feat, seq)
    """
    batch_size = seq_.shape[0]
    max_nodes = seq_.shape[1]
    seq_len = seq_.shape[3]

    V = torch.zeros((batch_size, seq_len, max_nodes, 2))
    for s in range(seq_len):
        step_ = seq_[:, :, :, s]
        for h in range(step_.shape[1]):
            V[:, s, h, :] = step_[:, h, :]
    return V

def nodes_rel_to_node_abs(nodes, init_node):
    """
    nodes: shape (batch, seq, node, feat)
    init_node: shape (batch, node, feat)
    """
    init_node = init_node.unsqueeze(1).repeat(1,nodes.shape[1],1,1)
    cums = torch.cumsum(nodes, axis=1)
    nodes_ = init_node + cums
    return nodes_


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
            self, data_dir, obs_len=8, pred_len=8, skip=1, threshold=0.002,
            min_ped=1, delim='\t', norm_lap_matr=True, test='test', max_nodes=0, 
            cache_train_name=None, cache_test_name=None, cache_val_name=None, dataset='eth',
            cache_name_=None):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files

        """
        super(TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir  
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr
        self.test = (test == 'test' or test == 'val')
        
        if cache_train_name is None:
            cache_train_name = 'caches/cache_%s_%s.pkl' % (dataset, cache_name_)
        if cache_test_name is None:
            cache_test_name = 'caches/cache_%s_test_%s.pkl' % (dataset, cache_name_)
        if cache_val_name is None:
            cache_val_name = 'caches/cache_%s_val_%s.pkl' % (dataset, cache_name_)

        f = None
        if test == 'train' and os.path.exists(cache_train_name):
            print('%s,cached !!!' % cache_train_name)
            f = open(cache_train_name, 'rb')
        elif test == 'val' and os.path.exists(cache_val_name):
            print('%s, cached !!!' % cache_val_name)
            f = open(cache_val_name, 'rb')
        elif test == 'test' and os.path.exists(cache_test_name):
            print('%s,cached !!!' % cache_test_name)
            f = open(cache_test_name, 'rb')

        if f is not None:
            data_ = pkl.load(f)
            self.seq_start_end = data_['seq_start_end']
            self.v_obs = data_['v_obs']
            self.A_obs = data_['A_obs']
            self.v_pred = data_['v_pred']
            self.num_seq = data_['num_seq']
            self.obs_traj = data_['obs_traj']
            self.pred_traj = data_['pred_traj']
            self.obs_traj_rel = data_['obs_traj_rel']
            self.pred_traj_rel = data_['pred_traj_rel']
            self.loss_mask = data_['loss_mask']
            self.num_peds_in_seq = data_['num_peds_in_seq']
            f.close()
        else:
            all_files = os.listdir(self.data_dir)
            all_files = [os.path.join(self.data_dir, _path) for _path in all_files]  # 拼合路径名形成可访问的路径
            all_files = [path for path in all_files if os.path.isfile(path)]
            
            print("Processing Data .....")
            num_peds_in_seq = []
            seq_list = []
            seq_list_rel = []
            loss_mask_list = []

            frame_data_list = []
            frames_list = []
            num_seq_list = []

            for path in all_files:
                data = read_file(path, delim)
                frames = np.unique(data[:, 0]).tolist()
                frame_data = []
                for frame in frames:
                    frame_data.append(data[frame==data[:,0],:])
                num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))
                num_seq_list.append(num_sequences)
                frame_data_list.append(frame_data)
                frames_list.append(frames)

                max_peds_in_frame = 0
                for idx in range(0, num_sequences * self.skip + 1, skip):
                    curr_seq_data = np.concatenate(
                        frame_data[idx:idx + self.seq_len], axis=0)
                    peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                    num_peds_considered = 0
                    for _, ped_id in enumerate(peds_in_curr_seq):
                        curr_ped_seq = curr_seq_data[curr_seq_data[:, 1]==ped_id, :]
                        pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                        pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                        if pad_end - pad_front != self.seq_len:
                            continue
                        num_peds_considered += 1
                    max_peds_in_frame = max(max_peds_in_frame, num_peds_considered)
                print(path, max_peds_in_frame)
                self.max_peds_in_frame = max(self.max_peds_in_frame, max_peds_in_frame)
            if max_nodes != 0:
                self.max_peds_in_frame = max_nodes
            else:
                max_nodes = self.max_peds_in_frame
            print('max_peds_in_frame:', self.max_peds_in_frame)

            for i in range(len(num_seq_list)):
                num_sequences = num_seq_list[i]
                frame_data = frame_data_list[i]
                frames = frames_list[i]
                for idx in range(0, num_sequences * self.skip + 1, skip):
                    curr_seq_data = np.concatenate(
                        frame_data[idx:idx + self.seq_len], axis=0)
                    peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                    curr_seq_rel = np.zeros((self.max_peds_in_frame, 2,
                                             self.seq_len))  # [3,2,20]
                    curr_seq = np.zeros((self.max_peds_in_frame, 2, self.seq_len))
                    curr_loss_mask = np.zeros((self.max_peds_in_frame,
                                               self.seq_len))
                    num_peds_considered = 0
                    for _, ped_id in enumerate(peds_in_curr_seq):
                        curr_ped_seq = curr_seq_data[curr_seq_data[:, 1]==ped_id, :]
                        curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                        pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                        pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                        if pad_end - pad_front != self.seq_len:
                            continue
                        curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                        rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                        rel_curr_ped_seq[:, 1:] = \
                            curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                        _idx = num_peds_considered
                        curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                        curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                        curr_loss_mask[_idx, pad_front:pad_end] = 1  # 设置掩码
                
                        num_peds_considered += 1
                        if num_peds_considered >= max_nodes:
                            break

                    if num_peds_considered > min_ped:
                        num_peds_in_seq.append(num_peds_considered)
                        loss_mask_list.append(curr_loss_mask)
                        seq_list.append(curr_seq)
                        seq_list_rel.append(curr_seq_rel)
            self.num_peds_in_seq = torch.Tensor(num_peds_in_seq)
            self.num_seq = len(seq_list)
            print(len(num_peds_in_seq), max(num_peds_in_seq))
            seq_list = np.concatenate(seq_list, axis=0)
            seq_list_rel = np.concatenate(seq_list_rel, axis=0)
            loss_mask_list = np.concatenate(loss_mask_list, axis=0)

            # Convert numpy -> Torch Tensor
            self.obs_traj = torch.from_numpy(
                seq_list[:, :, :self.obs_len]).type(torch.float)
            self.pred_traj = torch.from_numpy(
                seq_list[:, :, self.obs_len:]).type(torch.float)
            self.obs_traj_rel = torch.from_numpy(
                seq_list_rel[:, :, :self.obs_len]).type(torch.float)
            self.pred_traj_rel = torch.from_numpy(
                seq_list_rel[:, :, self.obs_len:]).type(torch.float)
            self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
            cum_start_idx = [0] + np.cumsum([self.max_peds_in_frame]*len(num_peds_in_seq)).tolist()
            self.seq_start_end = [
                (start, end)
                for start, end in zip(cum_start_idx, cum_start_idx[1:])
            ]

            self.v_obs = []
            self.A_obs = []
            self.v_pred = []
            pbar = tqdm(total=len(self.seq_start_end))
            for ss in range(len(self.seq_start_end)):
                pbar.update(1)

                start, end = self.seq_start_end[ss]
                v_ = get_V(self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :])
                a_ = get_A(self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :])
                if v_.shape[1] != self.max_peds_in_frame:
                    print(v_.shape)
                self.v_obs.append(v_.clone())
                self.A_obs.append(a_.clone())
                v_ = get_V(self.pred_traj[start:end, :], self.pred_traj_rel[start:end, :])
                self.v_pred.append(v_.clone())

            pbar.close()

            if test == 'val':
                cache_name = cache_val_name
            elif test == 'test':
                cache_name = cache_test_name
            else:
                cache_name = cache_train_name
            if not os.path.exists('caches'):
                os.mkdir('caches')
            with open(cache_name, 'wb') as f:
                pkl.dump({'seq_start_end': self.seq_start_end,
                          'v_obs': self.v_obs,
                          'A_obs': self.A_obs,
                          'v_pred': self.v_pred,
                          'num_seq': self.num_seq,
                          'obs_traj': self.obs_traj,
                          'pred_traj': self.pred_traj,
                          'obs_traj_rel': self.obs_traj_rel,
                          'pred_traj_rel': self.pred_traj_rel,
                          'loss_mask': self.loss_mask,
                          'num_peds_in_seq': self.num_peds_in_seq,
                          }, f)

    def __len__(self):
        return self.num_seq
        
    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        if self.test:
            out = self.v_obs[index][...,:3], self.A_obs[index], self.v_pred[index][...,:3], \
            self.loss_mask[start:end], self.obs_traj[start:end], \
            self.num_peds_in_seq[index]
        else:
            out = self.v_obs[index][...,:3], self.A_obs[index], self.v_pred[index][...,:3], \
                self.loss_mask[start:end], self.num_peds_in_seq[index]
        return out
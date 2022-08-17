import torch
import numpy
import random
import json

def _set_up_seed(seed):
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
    print(torch.rand(1,3))

def _load_config(config_name, base_dir='config'):
    confs = ['default.json', config_name]
    config = {}
    for conf in confs:
        config_ = json.load(open(f'{base_dir}/{conf}'))
        for k in config_:
            config[k] = config_[k]
    print(config)
    for k in config:
        globals()[k] = config[k]

_set_up_seed(42)
_load_config('cnn-gat-lstm-bv.json')
 
device = torch.device('cuda:'+str(cuda) if torch.cuda.is_available() else "cpu")
p_epoch = pretrain_epoches
g_epoch = gan_epoches
datasets = []
device_ids = [cuda]
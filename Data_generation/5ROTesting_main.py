# %%
import os
import numpy as np
import torch
import torch.nn as nn
import random
from backend import *
from config import root

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU 환경에서 모든 GPU에 seed 설정
    torch.backends.cudnn.deterministic = True  # cuDNN을 determinstic 모드로 설정
    torch.backends.cudnn.benchmark = False  # cuDNN benchmarking을 off로 설정
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)
import preprocessing

preprocess = True
Title = f'RO_PE_PI_DA2'
hysteresis_data_dir = os.path.join(root,f'Dynamic_analysis/RO')
processed_data_dir = os.path.join(root,'Data_npz')
os.makedirs(hysteresis_data_dir, exist_ok=True)
os.makedirs(processed_data_dir, exist_ok=True)
with open(os.path.join(root,'Data_generation/EQ_list.txt'), 'r') as f:
    EQ_list_ = [line.strip() for line in f]
print(EQ_list_[:3])
target_data = os.path.join(root,'Data_generation/elcentro_NS.txt')
n_samples = 80
k0 = ((6.283)**2)
mat_type = 'RambergOsgoodSteel'
mat_props = [7.5, k0, 0.002, 5]
val_size = 0.2
test=True
normalize_gap = 0.1
if preprocess:
    preprocessing.preprocess(n_samples,
                                        hysteresis_data_dir,
                                        processed_data_dir,
                                        title=Title,
                                        val_size=val_size,
                                        normalize_gap=normalize_gap,
                                        test=test)
print('Data preprocessing completed.')
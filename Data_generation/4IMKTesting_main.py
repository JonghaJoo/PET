# %%
import os
import numpy as np
import torch
import torch.nn as nn
import random
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
Title = 'IMK_PE_PI_DA2'
hysteresis_data_dir = os.path.join(root,f'Dynamic_analysis/IMK')
processed_data_dir = os.path.join(root,'Data_npz')
os.makedirs(hysteresis_data_dir, exist_ok=True)
os.makedirs(processed_data_dir, exist_ok=True)
with open(os.path.join(root,'Data_generation/EQ_list.txt'), 'r') as f:
    EQ_list_ = [line.strip() for line in f]
print(EQ_list_[:3])

target_data = os.path.join(root,'Data_generation/elcentro_NS.txt')
n_samples = 80
mat_type = 'Bilin'

mat_scale_factor = 0.04


E, Fy, Ix, Zx, H, L, d, tw, bf, tf, Lb, ry, My_mod = 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800
E, Fy, Ix, Zx, H, L, d, tw, bf, tf, Lb, ry, My_mod = E * mat_scale_factor, Fy * mat_scale_factor, Ix * mat_scale_factor, Zx * mat_scale_factor, H * mat_scale_factor, L * mat_scale_factor, d * mat_scale_factor, tw * mat_scale_factor, bf * mat_scale_factor, tf * mat_scale_factor, Lb * mat_scale_factor, ry * mat_scale_factor, My_mod * mat_scale_factor 

n = 10.0
c1 = 25.4
c2 = 6.895
K = (n + 1) * 6 * E * Ix / H

theta_p = 0.318 * (d/tw)**-0.550 * (bf/2/tf)**-0.345 * (Lb/ry)**-0.023 * (L/d)**0.090 * (c1 * d/533)**-0.330 * (c2 * Fy/355)**-0.130
theta_pc = 7.5 * (d/tw)**-0.610 * (bf/2/tf)**-0.710 * (Lb/ry)**-0.110 * (c1 * d/533)**-0.161 * (c2 * Fy/355)**-0.320
Lmda = 536 * (d/tw)**-1.260 * (bf/2/tf)**-0.525 * (Lb/ry)**-0.130 * (c2 * Fy/355)**-0.291 / 1000

theta_u = 0.2
Res = 0.4
McMy = 1.1

My_P = My_mod
My_N = -1.0 * My_mod

as_mem_p = (McMy - 1) * My_P / (theta_p * 6 * E * Ix / H)
as_mem_n = -(McMy - 1) * My_N / (theta_p * 6 * E * Ix / H)
SH_mod_p = as_mem_p / (1.0 + n * (1.0 - as_mem_p))
SH_mod_n = as_mem_n / (1.0 + n * (1.0 - as_mem_n))

Lmda = Lmda / 2.0
L_S = Lmda
L_C = Lmda
L_A = Lmda
L_K = Lmda
c_S = 1.0
c_C = 1.0
c_A = 1.0
c_K = 1.0
D_P = 1.0
D_N = 1.0

mat_props = [K, SH_mod_p * 10, SH_mod_n * 10, My_P, My_N, L_S, L_C, L_A, L_K, c_S,
             c_C, c_A, c_K, theta_p, theta_p, theta_pc, theta_pc, Res, Res, theta_u, theta_u, D_P, D_N]

val_size = 0.2
test = True
normalize_gap = 0.1
if preprocess:
    preprocessing.preprocess(n_samples,
                                        hysteresis_data_dir,
                                        processed_data_dir,
                                        title=Title,
                                        val_size=val_size,
                                        normalize_gap=normalize_gap,
                                        test = test)
print('Data preprocessing completed.')

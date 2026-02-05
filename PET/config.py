# %%
import torch
from pathlib import Path
root = Path(__file__).resolve().parent.parent
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
embed_size = 20
num_layers = 5
forward_expansion = 2
heads = 5
MLP_size = 12
max_length = 4500
input_dim = 3
output_dim = 2

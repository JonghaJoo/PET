import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
from pathlib import Path
root = Path(__file__).resolve().parent.parent
import torch
device = torch.device("cuda")

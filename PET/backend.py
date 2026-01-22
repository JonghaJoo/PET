import torch
import numpy as np
import math
def preprocess_data(X,y,device,step=2):
    len_tensor= torch.tensor(
        [torch.count_nonzero(torch.from_numpy(X[i,:,1])) for i in range(X.shape[0])], 
        device = device
        )

    X_list, y_list = [], []
    for i in range(len(X)):
        X_cut = X[i,:len_tensor[i]:step, :1]
        y_cut = y[i,:len_tensor[i]:step]
        X_list.append(torch.tensor(X_cut).float().to(device))
        y_list.append(torch.tensor(y_cut).float().to(device))

    return X_list, y_list

def downsampling(X_list, y_list, max_length, start_ratio=0.75, interval_division=4):
    X_downsampled, y_downsampled = [], []
    for mm in range(int(max_length*start_ratio), max_length+1, max_length//interval_division):
        for X_seq, y_seq in zip(X_list, y_list):
            u_length = len(X_seq)
            if u_length <= mm:
                X_downsampled.append(X_seq)
                y_downsampled.append(y_seq)
            else:
                for i in range(int(u_length*start_ratio), u_length+1, int(u_length//interval_division)):
                    X_cut,y_cut = X_seq[:i], y_seq[:i]
                    if i<=mm:
                        X_downsampled.append(X_cut)
                        y_downsampled.append(y_cut)
                    else:
                        step = math.ceil((i-mm)/(mm-1))+1
                        n = i-((mm-1)*(step-1)+2)+1
                        X_front = X_cut[0:(step*n)+1:step,:]
                        X_back  = X_cut[(step*n)+(step-1):i:step-1,:]
                        y_front = y_cut[0:(step*n)+1:step]
                        y_back  = y_cut[(step*n)+(step-1):i:step-1]
                        X_downsampled.append(torch.cat((X_front, X_back)))
                        y_downsampled.append(torch.cat((y_front, y_back)))
                        
    return X_downsampled, y_downsampled

def add_intervals_X(X_list,device):
    zeros = [torch.zeros((1,x.shape[1]), device = device) for x in X_list]  
    X_interval_1 = [torch.cat((z,x[1:]-x[:-1]),dim=0) for z, x in zip(zeros,X_list)]
    X_interval_2 = [torch.cat((z,xi[:1]-xi[:-1]),dim=0) for z, xi in zip (zeros,X_interval_1)]
    return [torch.cat((x,xi1,xi2),dim=1) for x,xi1,xi2 in zip(X_list, X_interval_1, X_interval_2)]
                        
def add_intervals_y(y_list,device):
    for i in range(len(y_list)):
        y_list[i] = y_list[i].unsqueeze(1)
    zeros = [torch.zeros((1,y.shape[1]), device = device) for y in y_list]  
    y_interval_1 = [torch.cat((z,y[1:]-y[:-1]),dim=0) for z, y in zip(zeros,y_list)]
    return [torch.cat((y,yi1),dim=1) for y,yi1 in zip(y_list, y_interval_1)]

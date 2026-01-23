import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from config import device

class Loss:
    def __init__(self, min_y, max_y):
        self.min_y = min_y
        self.max_y = max_y

    def normalization_loss(src,output, y_true, energy, device, min_y, max_y):
        mse_loss_init = (torch.mean((output - y_true)**2 )).detach()  
        phys_loss_init, phys_loss_new_init = (Loss.drucker_loss(src,output, energy, min_y ,max_y))
        phys_loss_init = phys_loss_init.detach()
        phys_loss_new_init = phys_loss_new_init.detach()
        print('MSE Loss / Phys Loss normalization/ Phys Loss normalization_new {:.8f}/{:.8f}/{:.8f}'.format(mse_loss_init, phys_loss_init, phys_loss_new_init))
        return mse_loss_init, phys_loss_init, phys_loss_new_init

    def combined_loss(self, src, output, y_true, energy, alpha, beta, mse_loss_init, phys_loss_init, phys_loss_new_init):
        self.mse_loss = torch.mean((output - y_true)**2 )
        self.phys_loss, self.phys_loss_new = Loss.drucker_loss(src, output, energy, self.min_y, self.max_y)   
        self.total_loss = (1-(alpha+beta)) * self.mse_loss / mse_loss_init + alpha * (self.phys_loss) / phys_loss_init + beta * (self.phys_loss_new) / phys_loss_new_init
        print('Total Loss / MSE Loss /Phys Loss  / Phys Loss_new: {:.8f}/{:.8f}/{:.8f}/{:.8f}'.format(self.total_loss, (1-(alpha+beta)) * self.mse_loss / mse_loss_init, alpha * self.phys_loss / phys_loss_init, beta * self.phys_loss_new / phys_loss_new_init), end='\r')
        return self.total_loss  

            
    def drucker_loss(src, output, energy, min_y, max_y):
        batch_size, seq_len, _ = output.shape
        num_chops = 1000
        lines = torch.linspace(min_y, max_y, num_chops).to(device)
        line_gap = lines[1]-lines[0]   
        lines_expanded = lines.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, seq_len)
        force = output[:,:,0].to(device)
        force_expanded = force.unsqueeze(1).expand(-1, num_chops, -1) 
        disp = src[:,:,0].to(device)
        disp_expanded = disp.unsqueeze(1).expand(-1, num_chops, -1)
        disp_diff = src[:,:,1].to(device)
        disp_diff_expanded = disp_diff.unsqueeze(1).expand(-1, num_chops, -1)
        
        energy_expanded = energy[:,:,0].unsqueeze(1).expand(-1, num_chops, -1).to(device)
        deducted_line_force = lines_expanded - force_expanded
        deducted_line_force_within_gap = (abs(deducted_line_force)<=line_gap).int()
        deducted_line_force_binary = (deducted_line_force>=0).int()   
        lineidx_each_f = torch.diff(deducted_line_force_binary, dim=2, prepend = deducted_line_force_binary[:,:,:1]) 
        
        x = lineidx_each_f.clone()                       
        mask = (x == -1)
        x[mask] = 0
        prev_mask = F.pad(mask, (0, 1), value=False)[:, :, 1:]
        x[prev_mask] = -1
        lineidx_each_f_edited = x       
        
        change_bool = (lineidx_each_f_edited != 0).bool()
        change_idx = torch.nonzero(change_bool)
        selected_energy = energy_expanded[change_idx[:,0], change_idx[:,1], change_idx[:,2]]
        diff_e = selected_energy[1:]-selected_energy[:-1]
        selected_gap = deducted_line_force_within_gap[change_idx[:,0], change_idx[:,1], change_idx[:,2]]
        both_within_gap = selected_gap[1:]*selected_gap[:-1]
        selected_diff_input = disp_diff_expanded[change_idx[:,0], change_idx[:,1], change_idx[:,2]] 
        selected_disp_input = disp_expanded[change_idx[:,0], change_idx[:,1], change_idx[:,2]] 
        selected_force_input = force_expanded[change_idx[:,0], change_idx[:,1], change_idx[:,2]] 
        monoton_increase_input = selected_diff_input[1:]*selected_diff_input[:-1]
        diff_idx = change_idx[1:]-change_idx[:-1]
        pinn = diff_e*(diff_idx[:,0]==0)*(diff_idx[:,1]==0)*(both_within_gap>0)
        diff_e_neg =pinn[pinn<0]
        standard_e = (selected_force_input[1:]+selected_force_input[:-1])*0.5*(selected_disp_input[1:]-selected_disp_input[:-1] )
        pinn_new = abs(diff_e)-abs(standard_e)
        pinn_new =pinn_new*(diff_idx[:,0]==0)*(diff_idx[:,1]==0)*(monoton_increase_input>0)
        pinn_new_standard = pinn_new[pinn_new<0]
        denom = float(batch_size * seq_len)
        loss1 = - diff_e_neg.sum()/ denom
        loss2 = - pinn_new_standard.sum()/ denom
        return loss1, loss2
        

class RelativePositionalEncoding(nn.Module):
    def __init__(self, max_len, embed_size):
        super().__init__()
        self.max_len = max_len
        self.embed_size = embed_size
        self.relative_embeddings = nn.Embedding(2 * max_len, embed_size)
        
    def forward(self, seq_len, device):
        indices = torch.arange(seq_len, device=device).unsqueeze(0) - torch.arange(seq_len, device=device).unsqueeze(1)
        indices = torch.clamp(indices, min=-self.max_len, max=self.max_len) + self.max_len  
        return self.relative_embeddings(indices)  
#%%
# Self-Attention 모듈 (multi-head attention)
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads, max_length): 
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.values = nn.Linear(embed_size, embed_size)
        self.keys   = nn.Linear(embed_size, embed_size)
        self.queries= nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
        self.relative_pos_encoding = RelativePositionalEncoding(max_length, embed_size)

    def forward(self, values, keys, queries, mask, after_mask): 
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1] 
        values  = self.values(values)
        keys    = self.keys(keys)
        queries = self.queries(queries)
        values  = values.reshape(N, value_len, self.heads, self.head_dim)
        keys    = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)
        relative_encoding = self.relative_pos_encoding(seq_len=query_len, device=device)
        relative_encoding = relative_encoding.reshape(1, query_len, key_len, self.heads, self.head_dim)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        energy += torch.einsum("nqhd,qkhd->nhqk", [queries, relative_encoding.squeeze(0)])
        if mask is not None: 
            energy = energy.masked_fill(mask == 0, float('-1e20'))
            
        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=3)
        out = torch.einsum("nhqk,nkhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        out = out*after_mask
        out = self.fc_out(out)
        return out
#%%
# Transformer 블록 (Self-attention + Feed Forward)
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads,  forward_expansion,max_length):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads, max_length)
        self.norm1     = nn.LayerNorm(embed_size)
        self.norm2     = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.GELU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        
    def forward(self, value, key, query, mask, after_mask):
        attention = self.attention(value, key, query, mask, after_mask)
        x = self.norm1(attention + query)
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        return out
#%%
# Encoder: 변위(displacement) 데이터를 입력받아 선형투영 + 고정 positional encoding 적용
class Encoder(nn.Module):
    def __init__(self, input_dim, embed_size, num_layers, heads, device, forward_expansion, max_length, output_dim, MLP_size):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device     = device
        self.input_projection = nn.Linear(input_dim, embed_size) #input의 마지막 차원이 embed_size로 선형변환됨 
        self.layers = nn.ModuleList(
            [TransformerBlock(
                embed_size, heads, forward_expansion, max_length) 
             for _ in range(num_layers)]
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, MLP_size),
            nn.SiLU(),
            nn.Linear(MLP_size, output_dim)
        )
        self.linear = nn.Linear(embed_size, output_dim)
        
    def forward(self, x, mask, after_mask):
        x = self.input_projection(x)                    
        for layer in self.layers:
            x = layer(x, x, x, mask, after_mask)
        out = self.feed_forward(x)+self.linear(x)    
        return out

class Transformer_finetuning(nn.Module):

    def __init__(
        self, 
        input_dim, 
        output_dim, 
        forward_expansion, 
        embed_size, 
        num_layers, 
        heads,
        MLP_size,
        device="cuda", 
        max_length=None
    ):
        super(Transformer_finetuning, self).__init__()
        
        self.encoder = Encoder(
            input_dim, 
            embed_size, 
            num_layers, 
            heads, 
            device, 
            forward_expansion, 
            max_length,
            output_dim,
            MLP_size
        )
        
    def make_mask(self, x, data_length):
        N, L, _ = x.shape  
        pad_mask = torch.ones((N, L), dtype=torch.bool, device=x.device)
        for i in range(N):
            pad_mask[i, data_length[i]:] = False  # 각 배치별 유효한 길이 이후 
        pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)

        lookahead_mask = torch.tril(torch.ones(L, L, device=x.device)).unsqueeze(0).unsqueeze(0)
        return pad_mask * lookahead_mask 
          
 
    def make_after_mask(self, x, data_length):
        N, max_length, _ = x.shape  
        after_mask = torch.ones((N, max_length, 1), dtype=torch.bool, device=x.device)
        for i in range(N):
            after_mask[i, data_length[i]:, :] = False  
        return after_mask

    def energy(self, src, output):
        diff_disp = src[:,:,1:2] 
        avg_force = (output[:,1:,:1] + output[:,:-1,:1])
        avg_force = torch.cat((output[:,:1,:1], avg_force), dim=1)
        delta_energy = (diff_disp * avg_force)/2
        dissipated_energy= torch.cumsum(delta_energy, dim=1)
        return dissipated_energy

    def forward(self, src, data_length):
        mask = self.make_mask(src, data_length)
        after_mask = self.make_after_mask(src, data_length)
        out = self.encoder(src, mask, after_mask)
        energy = self.energy(src, out)
        return out, energy
    
class TransformerDataset(Dataset):
    def __init__(self, src_data, trg_data, data_lengths):
        self.src_data = src_data 
        self.trg_data = trg_data
        self.data_lengths = data_lengths
        
    def __len__(self):
        return len(self.src_data) 
    def __getitem__(self, idx):
        return {
            'src': self.src_data[idx],
            'trg': self.trg_data[idx],
            'data_length': self.data_lengths[idx],
            'index': idx
        }

# %%    
def batch_padding(batch):
        src_batch = [item['src'] for item in batch]
        trg_batch = [item['trg'] for item in batch]
        data_length = [item['data_length'] for item in batch]
        indices = [item['index'] for item in batch]
        max_length = max(s.shape[0] for s in src_batch)
        src_padded = torch.zeros(len(batch), max_length, src_batch[0].shape[-1], device=device)
        trg_padded = torch.zeros(len(batch), max_length, trg_batch[0].shape[-1], device=device)
    
        for i in range(len(batch)):
            src_padded[i, :src_batch[i].shape[0], :] = src_batch[i]  
            trg_padded[i, :trg_batch[i].shape[0], :] = trg_batch[i]
        true_trg = trg_padded
        src_padded = src_padded.float()    
        return {'src': src_padded, 
                'true': true_trg,
                'data_length': torch.tensor(data_length, device=src_padded.device),
                'index': torch.tensor(indices, device=src_padded.device)}
                   
# %%
class Trainer:
    def __init__(self, model, device, alpha, beta, mse_loss_init, phys_loss_init, phys_loss_new_init):
        self.model = model
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.mse_loss_init = mse_loss_init  
        self.phys_loss_init = phys_loss_init    
        self.phys_loss_new_init = phys_loss_new_init
        

    def evaluate(self, dataloader):
        self.model.eval()  
        self.criterion = nn.MSELoss()
        total_loss = 0
        X_val_predictions = []  
        val_predictions = []
        val_indices = []
        with torch.no_grad():
            for batch in dataloader:
                src = batch['src'].to(self.device).float()
                true = batch['true'].to(self.device).float()
                data_length = batch['data_length'].to(self.device)
                batch_indices = batch['index'].cpu().numpy()
                output,_= self.model(src,data_length)
                loss = self.criterion(output, true)  
                # loss = self.criterion.combined_loss(src, output, true, energy, self.alpha, self.beta, self.mse_loss_init, self.phys_loss_init, self.phys_loss_new_init)
                val_predictions.append(output.cpu())
                X_val_predictions.append(src.cpu())
                val_indices.append(batch_indices)
                total_loss += loss.item() 
        return float(total_loss) / len(dataloader), val_predictions, val_indices, X_val_predictions 
    

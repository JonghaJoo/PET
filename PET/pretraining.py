#%%
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from config import device

# %%
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
    
class Encoder(nn.Module):
    def __init__(self, input_dim, embed_size, num_layers, heads, device, forward_expansion, max_length, output_dim, MLP_size):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device     = device
        self.input_projection = nn.Linear(input_dim, embed_size) 
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
    
class Transformer_pretraining(nn.Module):
    def __init__(
        self, 
        input_dim, 
        output_dim, 
        embed_size, 
        num_layers, 
        forward_expansion, 
        heads,
        MLP_size,
        device="cuda", 
        max_length=None
    ):
        super(Transformer_pretraining, self).__init__()
        
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
            pad_mask[i, data_length[i]:] = False  
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
        return out

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
            src_padded[i, :src_batch[i].shape[0], :] = src_batch[i]  # 실제 값 채우기
            trg_padded[i, :trg_batch[i].shape[0], :] = trg_batch[i]
        true_trg = trg_padded
        src_padded = src_padded.float()    
        return {'src': src_padded, 
                'true': true_trg,
                'data_length': torch.tensor(data_length, device=src_padded.device),
                'index': torch.tensor(indices, device=src_padded.device)}
                   
# %%
class Trainer0:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        
    def train_epoch(self, dataloader):
        self.model.train()  
        total_loss = 0
        train_predictions = []
        X_predictions = []
        indices = []
        true_y = []
        for batch in dataloader: 
            src = batch['src'].to(self.device).float()   
            true = batch['true'].to(self.device).float()    
            data_length = batch['data_length'].to(self.device)
            batch_indices = batch['index'].cpu().numpy()
            self.optimizer.zero_grad()                     
            output = self.model(src, data_length)
            indices.append(batch_indices)
            train_predictions.append(output.cpu())
            X_predictions.append(src.cpu())   
            true_y.append(true.cpu())
            loss = self.criterion(output, true)      
            loss.backward()                                 
            self.optimizer.step()                           
            total_loss += loss.item()                     
        return total_loss / len(dataloader), train_predictions, indices , X_predictions, true_y
    
    def evaluate(self, dataloader):
        self.model.eval()  
        total_loss = 0
        X_val_predictions = []  
        val_predictions = []
        val_indices = []
        val_true_y= []
        with torch.no_grad():
            for batch in dataloader:
                src = batch['src'].to(self.device).float()
                true = batch['true'].to(self.device).float()
                data_length = batch['data_length'].to(self.device)
                batch_indices = batch['index'].cpu().numpy()
                output = self.model(src,data_length)
                val_predictions.append(output.cpu())
                X_val_predictions.append(src.cpu())
                val_indices.append(batch_indices)
                val_true_y.append(true.cpu())
                loss = self.criterion(output, true)
                total_loss += loss.item()
        return total_loss / len(dataloader), val_predictions, val_indices, X_val_predictions, val_true_y
    

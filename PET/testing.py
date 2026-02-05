import torch
import torch.nn as nn
from torch.utils.data import Dataset
from config import device

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
        
    def forward(self, values, keys, queries, mask): 
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
        out = out
        out = self.fc_out(out)
        return out
#%%
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
        
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.norm1(attention + query)
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        return out
#%%
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
    def forward(self, x, mask):
        x = self.input_projection(x)                    
        for layer in self.layers:
            x = layer(x, x, x, mask)
        out = self.feed_forward(x) + self.linear(x) 
        return out

class Transformer_predict(nn.Module):

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
        super(Transformer_predict, self).__init__()
        
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
        
    def make_mask(self, x):
        _, L, _ = x.shape  
        lookahead_mask = torch.tril(torch.ones(L, L, device=x.device)).unsqueeze(0).unsqueeze(0)
        return lookahead_mask 
          

    def forward(self, src):
        mask = self.make_mask(src)
        out = self.encoder(src, mask)
        return out

# %%
class Force_prediction:
    def __init__(self, model, device):
        self.model = model 
        self.device = device
        
    def predict_epoch(self, data):
        self.model.eval()  
        with torch.no_grad():
            output = self.model(data)
        return output

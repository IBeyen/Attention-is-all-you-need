import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import *

def positional_encoder(seq_len, d_model):
    PE = torch.arange(0, seq_len).repeat(d_model, 1).T.float()
    i_mat = torch.arange(0, d_model).repeat(seq_len, 1)
    PE[:,torch.arange(0, seq_len, 2)] = torch.sin(PE[:,torch.arange(0, seq_len, 2)]/10000**(i_mat[:,torch.arange(0, seq_len, 2)]/d_model))
    PE[:,torch.arange(1, seq_len, 2)] = torch.cos(PE[:,torch.arange(1, seq_len, 2)]/10000**(i_mat[:,torch.arange(1, seq_len, 2)]/d_model))
    return PE
    
class FFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model= d_model
        self.layer_1 = nn.Linear(self.d_model, self.d_model)
        self.layer_2 = nn.Linear(self.d_model, self.d_model)
        
    def forward(self, x):
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, h, d_model, d_k, d_v):
        super().__init__()
        self.num_head = h
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        
        self.multihead_att = Multihead_Attention(h, d_model, d_k, d_v)
        self.FF = FFN(d_model)
    
    def forward(self, x):
        x = x + self.multihead_att(x, x, x)
        x = F.normalize(x)
        x = x + self.FF(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, h, d_model, d_k, d_v):
        super().__init__()
        self.num_head = h
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        
        self.multihead_masked_att = Multihead_Attention(h, d_model, d_k, d_v, mask=True)
        self.multihead_att = Multihead_Attention(h, d_model, d_k, d_v)
        self.FF = FFN(d_model)
    
    def forward(self, x, encoder_output):
        x = self.multihead_masked_att(x, x, x)
        x = F.normalize(x)
        x = x + self.multihead_att(encoder_output, encoder_output, x)
        x = F.normalize(x)
        x = x + self.FF(x)
        x = F.normalize(x)
        return x
    
class Transformer(nn.Module):
    def __init__(self, h, d_model, d_k, d_v, num_tokens):
        super().__init__()
        self.num_head = h
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.num_tokens = num_tokens
        
        self.embeddings_1 = nn.Embedding(num_embeddings=num_tokens, embedding_dim=d_model, padding_idx=2)
        self.embeddings_2 = nn.Embedding(num_embeddings=num_tokens, embedding_dim=d_model, padding_idx=2)
        self.encoder = Encoder(h, d_model, d_k, d_v)
        self.decoder = Decoder(h, d_model, d_k, d_v)
        
        self.linear = nn.Linear(d_model, num_tokens)
        
    def forward(self, x_1, x_2):
        x_1 = self.embeddings_1(x_1)
        x_2 = self.embeddings_2(x_2)
        encoder_output = self.encoder(x_1)
        decoder_output = self.decoder(x_2, encoder_output)
        out = self.linear(decoder_output)
        out = F.softmax(out)
        return out
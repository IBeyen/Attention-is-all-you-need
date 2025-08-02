import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tokenizer import *
import torch

""" 
The train dataset is too big to load all at once as a pd.Dataframe so we will do it in chunks
"""
BOS_ID = 0
EOS_ID = 1
PAD_ID = 2

def collate_fn(batch):
    src, tgt_in, tgt_out = zip(*batch)
    src = torch.stack(list(src))
    tgt_in = torch.stack(list(tgt_in))
    tgt_out = torch.stack(list(tgt_out))
    return src, tgt_in, tgt_out
    

class CustomDataset(Dataset):
    def __init__(self, path, merges_en, merges_de, src_max_len, tgt_max_len):
        super().__init__()
        self.path = path
        self.merges_en, self.merges_de = merges_en, merges_de
        self.data = pd.read_csv(path)
        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x, y = self.data.iloc[idx]["de"], self.data.iloc[idx]["en"]
        x, y = encode_tokens(x, self.merges_de), encode_tokens(y, self.merges_en)
        src_tokens = x
        tgt_in = [BOS_ID] + y
        tgt_out = y + [EOS_ID]
        
        src_tokens += [PAD_ID for i in range(self.src_max_len-len(src_tokens))]
        tgt_in += [PAD_ID for i in range(self.tgt_max_len-len(tgt_in))]
        tgt_out += [PAD_ID for i in range(self.tgt_max_len-len(tgt_out))]
        src_tokens = src_tokens[:self.src_max_len]
        tgt_in = tgt_in[:self.tgt_max_len]
        tgt_out = tgt_out[:self.tgt_max_len]
        return torch.Tensor(src_tokens), torch.Tensor(tgt_in), torch.Tensor(tgt_out)
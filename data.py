import pandas as pd
from torch.utils.data import Dataset, DataLoader

train = pd.read_csv("WMT 2014 English-German\wmt14_translate_de-en_train.csv", chunksize=10)
test = pd.read_csv("WMT 2014 English-German\wmt14_translate_de-en_test.csv")
val = pd.read_csv("WMT 2014 English-German\wmt14_translate_de-en_validation.csv")

""" 
The train dataset is too big to load all at once as a pd.Dataframe so we will do it in chunks
"""
class TrainDataset(Dataset):
    def __init__(self, path):
        if train:
            self.chunk = 10000
        else:
            self.chunk = -1
            
        self.path = path
        self.size = -1
        self.iterator = pd.read_csv(path, chunksize=10000)
        self.data = self.iterator.get_chunk()
        
    def __len__(self):
        if self.size == -1:
            start = True
            reader = pd.read_csv(self.path, chunksize=10000)
            while True:
                data = reader.get_chunk()
                if data.index.values[0] == 0:
                    if not start:
                        return self.size
                self.size += len(data)
                start = False
        return self.size
    
    def __getitem__(self, idx):
        pass
            
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from data import *
from attention import *
from model import * 
from tokenizer import *

NUM_TOKENS = 500

data = pd.read_csv("WMT 2014 English-German\wmt14_translate_de-en_train.csv", nrows=1000)
str_en = ""
str_de = ""
for i in range(len(data)):
    str_en += str(data['en'].iloc[i]) + " " 
    str_de += str(data['de'].iloc[i]) + " "

merges_en, vocab_en = train_tokenizer(str_en, NUM_TOKENS)
merges_de, vocab_de = train_tokenizer(str_de, NUM_TOKENS)
del data

train = CustomDataset("WMT 2014 English-German\wmt14_translate_de-en_train.csv", src_max_len=500, tgt_max_len=500,  merges_de=merges_de, merges_en=merges_en)
test = CustomDataset("WMT 2014 English-German\wmt14_translate_de-en_test.csv", src_max_len=500, tgt_max_len=500,  merges_de=merges_de, merges_en=merges_en)
validation = CustomDataset("WMT 2014 English-German\wmt14_translate_de-en_validation.csv", src_max_len=500, tgt_max_len=500,  merges_de=merges_de, merges_en=merges_en)

train_loader = DataLoader(train, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test, collate_fn=collate_fn)
val_loader = DataLoader(validation, collate_fn=collate_fn)

transformer = Transformer(8, 512, 64, 64, NUM_TOKENS)
transformer.to("cpu")
optimizer = optim.Adam(transformer.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)  # Ignore padding in loss

for epoch in range(10):
    transformer.train()
    total_loss = 0

    with torch.autograd.set_detect_anomaly(True):
        for src_tokens, tgt_in_tokens, tgt_out_tokens in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            src_tokens = src_tokens.to("cpu").long()
            tgt_in_tokens = tgt_in_tokens.to("cpu").long()
            tgt_out_tokens = tgt_out_tokens.to("cpu").long()

            optimizer.zero_grad()
            outputs = transformer(src_tokens, tgt_in_tokens)
            
            outputs = outputs.view(-1, outputs.size(-1))  # [batch_size * seq_len, vocab_size]
            tgt_out_tokens = tgt_out_tokens.view(-1)      # [batch_size * seq_len]

            loss = criterion(outputs, tgt_out_tokens)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")
# ====================================================
# Library
# ====================================================
import os
import re
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()
import torch
import string
from utils import Tokenizer, text_clean, fix_error
# ====================================================
# Data Loading
# ====================================================
train = pd.read_csv('./train_files/train_captions.csv')
print(f'train.shape: {train.shape}')

for i, caption in enumerate(train['captions'].values):
    # caption = fix_error(caption)
    newcaption = text_clean(caption)
    train["captions"].iloc[i] = newcaption


# create tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train['captions'].values)
torch.save(tokenizer, 'tokenizer_vi_fix_error.pth')
# print('Saved tokenizer')

lengths = []
tk0 = tqdm(train['captions'].values, total=len(train))
for text in tk0:
    seq = tokenizer.text_to_sequence(text)
    length = len(seq) - 2
    lengths.append(length)
train['length'] = lengths
# train.to_pickle('train.pkl')
# print('Saved preprocessed train.pkl')


print(len(tokenizer))
# print(f"{tokenizer.stoi}")
# print(f"{tokenizer.stofreq}")
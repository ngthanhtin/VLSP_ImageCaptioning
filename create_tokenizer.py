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
# ====================================================
# Data Loading
# ====================================================
train = pd.read_csv('../data/train_captions.csv')
print(f'train.shape: {train.shape}')

# ====================================================
# Tokenizer
# ====================================================
class Tokenizer(object):
    
    def __init__(self):
        self.stoi = {}
        self.itos = {}

    def __len__(self):
        return len(self.stoi)
    
    def fit_on_texts(self, texts):
        vocab = set()
        for text in texts:
            vocab.update(text.split(' '))
        vocab = sorted(vocab)
        vocab.append('<sos>')
        vocab.append('<eos>')
        vocab.append('<pad>')
        for i, s in enumerate(vocab):
            self.stoi[s] = i
        self.itos = {item[1]: item[0] for item in self.stoi.items()}
        
    def text_to_sequence(self, text):
        sequence = []
        sequence.append(self.stoi['<sos>'])
        for s in text.split(' '):
            sequence.append(self.stoi[s])
        sequence.append(self.stoi['<eos>'])
        return sequence
    
    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            sequence = self.text_to_sequence(text)
            sequences.append(sequence)
        return sequences

    def sequence_to_text(self, sequence):
        return ''.join(list(map(lambda i: self.itos[i], sequence)))
    
    def sequences_to_texts(self, sequences):
        texts = []
        for sequence in sequences:
            text = self.sequence_to_text(sequence)
            texts.append(text)
        return texts
    
    def predict_caption(self, sequence):
        caption = ''
        for i in sequence:
            if i == self.stoi['<eos>'] or i == self.stoi['<pad>']:
                break
            caption += self.itos[i]
        return caption
    
    def predict_captions(self, sequences):
        captions = []
        for sequence in sequences:
            caption = self.predict_caption(sequence)
            captions.append(caption)
        return captions


# ====================================================
# Preprocess functions
# ====================================================

print("\nLower case..")
def lowercase(text_original):
    text_lower = text_original.lower()
    return(text_lower)

print("\nRemove punctuations..")
def remove_punctuation(text_original):
    text_no_punctuation = text_original.translate(str.maketrans('','',string.punctuation))
    return(text_no_punctuation)

print("\nRemove a single character word..")
def remove_single_character(text):
    text_len_more_than1 = ""
    for word in text.split():
        if len(word) > 1:
            text_len_more_than1 += " " + word
    return(text_len_more_than1)

print("\nRemove words with numeric values..")
def remove_numeric(text,printTF=False):
    text_no_numeric = ""
    for word in text.split():
        isalpha = word.isalpha()
        if printTF:
            print("    {:10} : {:}".format(word,isalpha))
        if isalpha:
            text_no_numeric += " " + word
    return(text_no_numeric)

def text_clean(text_original):
    text = lowercase(text_original)
    text = remove_punctuation(text)
    # text = remove_single_character(text)
    text = remove_numeric(text)
    return(text)

for i, caption in enumerate(train['captions'].values):
        newcaption = text_clean(caption)
        train["captions"].iloc[i] = newcaption


# create tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train['captions'].values)
torch.save(tokenizer, 'tokenizer_vi_fix.pth')
print('Saved tokenizer')

lengths = []
tk0 = tqdm(train['captions'].values, total=len(train))
for text in tk0:
    seq = tokenizer.text_to_sequence(text)
    length = len(seq) - 2
    lengths.append(length)
train['length'] = lengths
# train.to_pickle('train_vi.pkl')
# print('Saved preprocessed train.pkl')


print(len(tokenizer))
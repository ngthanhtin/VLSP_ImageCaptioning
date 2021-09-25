import os
from matplotlib import pyplot as plt

OUTPUT_DIR = './'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    
import numpy as np
import pandas as pd

import sys
sys.path.append('../input/timm-pytorch-image-models/pytorch-image-models-master')

import os
import time
import random


import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

import torch.fft
from config import CFG
from utils import *
from dataset import TrainDataset, TestDataset
from transformation import get_transforms
from models.model import CNN, DecoderWithAttention

device = CFG.device

def init_logger(log_file=OUTPUT_DIR+'train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = init_logger()


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(seed = CFG.seed)

def bms_collate(batch):
    imgs, labels, label_lengths = [], [], []
    for data_point in batch:
        imgs.append(data_point[0])
        labels.append(data_point[1])
        label_lengths.append(data_point[2])
    labels = pad_sequence(labels, batch_first = True, padding_value = tokenizer.stoi["<pad>"])
    return torch.stack(imgs), labels, torch.stack(label_lengths).reshape(-1, 1)

tokenizer = torch.load('tokenizer_vi.pth')
# print(f"tokenizer.stoi: {tokenizer.stoi}")

def inference(test_loader, encoder, decoder, tokenizer, device):
    
    encoder.eval()
    decoder.eval()
    
    text_preds = []
    tk0 = tqdm(test_loader, total = len(test_loader))
    
    for images in tk0:
        
        images = images.to(device)
        
        with torch.no_grad():
            features = encoder(images)
            predictions = decoder.predict(features, CFG.max_len, tokenizer)
            
        predicted_sequence = torch.argmax(predictions.detach().cpu(), -1).numpy()
        _text_preds = tokenizer.predict_captions(predicted_sequence)
        text_preds.append(_text_preds)
        
    text_preds = np.concatenate(text_preds)
    
    return text_preds





df = pd.read_csv('../data/vietcap4h-public-test/test_captions.csv')




def get_test_file_path(image_id):
    return CFG.test_path + "/images_public_test/{}".format(image_id)

def get_test_id(path_file):
    return path_file.split('/')[-1]
test = df

test['file_path'] = test['id'].apply(get_test_file_path)
print(f'test.shape: {test.shape}')


# ====================================================
# load model
# ====================================================
    
states = torch.load(CFG.prev_model, map_location = torch.device('cpu'))

encoder = CNN()
encoder.load_state_dict(states['encoder'])
encoder.to(device)

decoder = DecoderWithAttention(attention_dim = CFG.attention_dim, 
                               embed_dim     = CFG.embed_dim, 
                               encoder_dim   = CFG.enc_size,
                               decoder_dim   = CFG.decoder_dim,
                               num_layers    = CFG.decoder_layers,
                               vocab_size    = len(tokenizer), 
                               dropout       = CFG.dropout, 
                               device        = device)
decoder.load_state_dict(states['decoder'])
decoder.to(device)

del states
import gc
gc.collect()

# ====================================================
# inference
# ====================================================

test_dataset = TestDataset(test, transform = get_transforms(data = 'valid'))
test_loader  = DataLoader(test_dataset, batch_size = 32, shuffle = False, num_workers = CFG.num_workers)
predictions  = inference(test_loader, encoder, decoder, tokenizer, device)


# ====================================================
#  submission to json and csv
# ====================================================

test['id'] = test['file_path'].apply(get_test_id)
test['captions'] = [f"{text}" for text in predictions]
# test[['id', 'captions']].to_csv('submission.csv', index=False)


# json
import json
data = []
for index, row in test.iterrows():
    captions, id = row['captions'], row['id']
    data.append({'id': id, 'captions': captions})
with open('submission.json', 'w') as outfile:
    json.dump(data, outfile, ensure_ascii=False, indent=4)
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

from sklearn.model_selection import StratifiedKFold

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



def train_fn(train_loader, encoder, decoder, criterion, 
             encoder_optimizer, decoder_optimizer, epoch,
             encoder_scheduler, decoder_scheduler, device):
    
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()

    # switch to train mode
    encoder.train()
    decoder.train()
    
    start = end = time.time()
    global_step = 0
    
    for step, (images, labels, label_lengths) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        images        = images.to(device)
        labels        = labels.to(device)
        label_lengths = label_lengths.to(device)
        batch_size    = images.size(0)
        
        features = encoder(images)
        predictions, caps_sorted, decode_lengths, alphas, sort_ind = decoder(features, labels, label_lengths)
        targets     = caps_sorted[:, 1:]
        predictions = pack_padded_sequence(predictions, decode_lengths, batch_first=True).data
        targets     = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
        
        loss        = criterion(predictions, targets)

        # record loss
        losses.update(loss.item(), batch_size)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
            
        if CFG.apex:
            with amp.scale_loss(loss, decoder_optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
            
        encoder_grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), CFG.max_grad_norm)
        decoder_grad_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), CFG.max_grad_norm)
        
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            encoder_optimizer.step()
            decoder_optimizer.step()
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            global_step += 1
            
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Encoder Grad: {encoder_grad_norm:.4f}  '
                  'Decoder Grad: {decoder_grad_norm:.4f}  '
                  #'Encoder LR: {encoder_lr:.6f}  '
                  #'Decoder LR: {decoder_lr:.6f}  '
                  .format(
                   epoch+1, step, len(train_loader), 
                   batch_time        = batch_time,
                   data_time         = data_time, 
                   loss              = losses,
                   remain            = timeSince(start, float(step+1)/len(train_loader)),
                   encoder_grad_norm = encoder_grad_norm,
                   decoder_grad_norm = decoder_grad_norm,
                   #encoder_lr=encoder_scheduler.get_lr()[0],
                   #decoder_lr=decoder_scheduler.get_lr()[0],
                   ))
    
    return losses.avg


def valid_fn(valid_loader, encoder, decoder, tokenizer, criterion, device):
    
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    
    # switch to evaluation mode
    encoder.eval()
    decoder.eval()
    
    text_preds = []
    start = end = time.time()
    
    for step, (images) in enumerate(valid_loader):

        # measure data loading time
        data_time.update(time.time() - end)
        
        images     = images.to(device)
        batch_size = images.size(0)
        
        with torch.no_grad():
            features    = encoder(images)
            predictions = decoder.predict(features, CFG.max_len, tokenizer)
        predicted_sequence = torch.argmax(predictions.detach().cpu(), -1).numpy()
        _text_preds        = tokenizer.predict_captions(predicted_sequence)
        
        text_preds.append(_text_preds)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  .format(
                   step, len(valid_loader), 
                   batch_time = batch_time,
                   data_time  = data_time,
                   remain     = timeSince(start, float(step+1)/len(valid_loader)),
                   ))
            
    text_preds = np.concatenate(text_preds)
    return text_preds


# ====================================================
# Train loop
# ====================================================

def bms_collate(batch):
    imgs, labels, label_lengths = [], [], []
    for data_point in batch:
        imgs.append(data_point[0])
        labels.append(data_point[1])
        label_lengths.append(data_point[2])
    labels = pad_sequence(labels, batch_first = True, padding_value = tokenizer.stoi["<pad>"])
    return torch.stack(imgs), labels, torch.stack(label_lengths).reshape(-1, 1)

tokenizer = torch.load('./tokenizers/tokenizer_vi_not_remove_single_character.pth')
# print(f"tokenizer.stoi: {tokenizer.stoi}")

def train_loop(folds, fold):

    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index
   
    train_folds  = folds.loc[trn_idx].reset_index(drop = True)
    
    valid_folds  = folds.loc[val_idx].reset_index(drop = True)
    valid_labels = valid_folds['captions'].values

    train_dataset = TrainDataset(train_folds, tokenizer, transform = get_transforms(data = 'train'))
    valid_dataset = TestDataset(valid_folds, transform = get_transforms(data = 'valid'))
  
    train_loader = DataLoader(train_dataset, 
                              batch_size  = CFG.batch_size, 
                              shuffle     = True, 
                              num_workers = CFG.num_workers, 
                              pin_memory  = True,
                              drop_last   = True,
                              collate_fn  = bms_collate)
    valid_loader = DataLoader(valid_dataset, 
                              batch_size  = CFG.batch_size, 
                              shuffle     = False, 
                              num_workers = CFG.num_workers,
                              pin_memory  = True, 
                              drop_last   = False)
    
    # ====================================================
    # scheduler 
    # ====================================================
    def get_scheduler(optimizer):
        if CFG.scheduler=='ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, 
                                          mode     = 'min', 
                                          factor   = CFG.factor, 
                                          patience = CFG.patience, 
                                          verbose  = True, 
                                          eps      = CFG.eps)
        elif CFG.scheduler=='CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, 
                                          T_max      = CFG.T_max, 
                                          eta_min    = CFG.min_lr, 
                                          last_epoch = -1)
        elif CFG.scheduler=='CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer, 
                                                    T_0        = CFG.T_0, 
                                                    T_mult     = 1, 
                                                    eta_min    = CFG.min_lr, 
                                                    last_epoch = -1)
        return scheduler

    # ====================================================
    # model & optimizer
    # ====================================================

#    states = torch.load(CFG.prev_model,  map_location=torch.device('cpu'))

#     encoder = Encoder(CFG.model_name, 
#                       pretrained = True)
    encoder = CNN(is_pretrained=True, type_=CFG.model_name)
#    encoder.load_state_dict(states['encoder'])
    
    encoder.to(device)
    encoder_optimizer = Adam(encoder.parameters(), 
                             lr           = CFG.encoder_lr, 
                             weight_decay = CFG.weight_decay, 
                             amsgrad      = False)
#    encoder_optimizer.load_state_dict(states['encoder_optimizer'])
    encoder_scheduler = get_scheduler(encoder_optimizer)
#    encoder_scheduler.load_state_dict(states['encoder_scheduler'])
    
    decoder = DecoderWithAttention(attention_dim = CFG.attention_dim, 
                                   embed_dim     = CFG.embed_dim, 
                                   encoder_dim   = CFG.enc_size,
                                   decoder_dim   = CFG.decoder_dim,
                                   num_layers    = CFG.decoder_layers,
                                   vocab_size    = len(tokenizer), 
                                   dropout       = CFG.dropout, 
                                   device        = device)
#    decoder.load_state_dict(states['decoder'])
    decoder.to(device)
    decoder_optimizer = Adam(decoder.parameters(), 
                             lr           = CFG.decoder_lr, 
                             weight_decay = CFG.weight_decay, 
                             amsgrad      = False)
#    decoder_optimizer.load_state_dict(states['decoder_optimizer'])

    decoder_scheduler = get_scheduler(decoder_optimizer)
 #   decoder_scheduler.load_state_dict(states['decoder_scheduler'])

    # ====================================================
    # loop
    # ====================================================
    criterion = nn.CrossEntropyLoss(ignore_index = tokenizer.stoi["<pad>"])
#     criterion = seq_anti_focal_cross_entropy_loss()

    best_score = 0 #np.inf
    best_loss  = np.inf
    
    for epoch in range(CFG.epochs):
        print("Epoch: ", epoch)
        start_time = time.time()
        # train
        avg_loss = train_fn(train_loader, encoder, decoder, criterion, 
                            encoder_optimizer, decoder_optimizer, epoch, 
                            encoder_scheduler, decoder_scheduler, device)

        # eval
        text_preds = valid_fn(valid_loader, encoder, decoder, tokenizer, criterion, device)
        text_preds = [f"{text}" for text in text_preds]
        LOGGER.info(f"labels: {valid_labels[:5]}")
        LOGGER.info(f"preds: {text_preds[:5]}")
        
        # scoring
        # score = get_score_levenshtein(valid_labels, text_preds)
        score = get_score_bleu(valid_labels, text_preds)
        
        if isinstance(encoder_scheduler, ReduceLROnPlateau):
            encoder_scheduler.step(score)
        elif isinstance(encoder_scheduler, CosineAnnealingLR):
            encoder_scheduler.step()
        elif isinstance(encoder_scheduler, CosineAnnealingWarmRestarts):
            encoder_scheduler.step()
            
        if isinstance(decoder_scheduler, ReduceLROnPlateau):
            decoder_scheduler.step(score)
        elif isinstance(decoder_scheduler, CosineAnnealingLR):
            decoder_scheduler.step()
        elif isinstance(decoder_scheduler, CosineAnnealingWarmRestarts):
            decoder_scheduler.step()

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f} - Best Score: {best_score:.4f}')
        
        if score > best_score: # < for Levenhstein, > for BLEU
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'encoder': encoder.state_dict(), 
                        'encoder_optimizer': encoder_optimizer.state_dict(), 
                        'encoder_scheduler': encoder_scheduler.state_dict(), 
                        'decoder': decoder.state_dict(), 
                        'decoder_optimizer': decoder_optimizer.state_dict(), 
                        'decoder_scheduler': decoder_scheduler.state_dict(), 
                        'text_preds': text_preds,
                       },
                        OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best.pth')

    print("End train loop")



#---------READ DATA--------------------

df = pd.read_csv('../data/train_captions.csv')

def read_data(data_frame):
    for i, caption in enumerate(data_frame['captions'].values):
        newcaption = text_clean(caption)
        data_frame["captions"].iloc[i] = newcaption
        
    lengths = []
    tk0 = tqdm(data_frame['captions'].values, total=len(data_frame))
    for text in tk0:
        seq = tokenizer.text_to_sequence(text)
        length = len(seq)
        lengths.append(length)
    
    data_frame['length'] = lengths
    print("Max Length: ",data_frame['length'].max())
    return data_frame


def get_train_file_path(image_id):
    return CFG.train_path + "/images_train/{}".format(image_id)

train = read_data(df)

train['file_path'] = train['id'].apply(get_train_file_path)
print(train['length'].min())
print(f'train.shape: {train.shape}')

# ---------------- CALCULATE MEAN, STD---------------------
def calculate_mean_std(): # should use Imagenet mean and std
    train_ds = TrainDataset(train, tokenizer, transform = get_transforms(data = 'train'))

    train_loader = DataLoader(train_ds, 
                              batch_size  = CFG.batch_size, 
                              shuffle     = True, 
                              num_workers = CFG.num_workers, 
                              pin_memory  = True,
                              drop_last   = True,
                              collate_fn  = bms_collate)

    print('==> Computing mean and std..')
    mean = 0.
    std = 0.
    for (images, labels, label_lengths) in tqdm(train_loader):
        inputs = images.float()
        # Rearrange batch to be the shape of [B, , C, W * H]
        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
        # Compute mean and std here
        mean += inputs.mean(2).sum(0)/255.0
        std += inputs.std(2).sum(0)/255.0
    mean /= len(train_ds)
    std /= len(train_ds)
    
    print(mean, std)

# ------------------- TRAIN ------------------------
if CFG.debug:
    CFG.epochs = 1
    train = train.sample(n = CFG.samp_size, random_state = CFG.seed).reset_index(drop = True)

train_dataset = TrainDataset(train, tokenizer, transform = get_transforms(data='train'))

folds = train.copy()
Fold = StratifiedKFold(n_splits = CFG.n_fold, shuffle = True, random_state = CFG.seed)
for n, (train_index, val_index) in enumerate(Fold.split(folds, folds['length'])):
    folds.loc[val_index, 'fold'] = int(n)
folds['fold'] = folds['fold'].astype(int)

for train_fold in CFG.trn_fold:
    train_loop(folds, train_fold)
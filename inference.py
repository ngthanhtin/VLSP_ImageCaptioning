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
import random


import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

import torch.fft
from config import CFG
from utils import *
from dataset import TestDataset, TrainDataset
from transformation import get_transforms
from models.model import CNN, Net
from beam_search import Beam
from typing import Tuple, Dict, Optional
from models.ensemble import EnsembleNet

device = CFG.device

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(seed = CFG.seed)

tokenizer = torch.load(CFG.tokenizer_path)

# def decode_beamsearch(self, memory, max_length, beamsize, memory_key_padding_mask=None):
#         # type: (torch.Tensor, int, int, Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]
#         batch_size = memory.size(0)
#         sos_inputs = torch.full((batch_size, 1), self.vocab.SOS_IDX,
#                                 dtype=torch.long, device=memory.device)                 # [B, L=1]
 
#         end_flag = torch.zeros(batch_size, dtype=torch.bool,
#                                device=memory.device)                                    # [B]
 
#         inputs = (sos_inputs, torch.zeros(batch_size))                                                             # [B, K, L=1]
#         batch_decoded = []
#         all_candidates = [inputs]
 
#         for _ in range(max_length + 1):
#             next_candidate = []
#             for (inputs, score) in all_candidates:
#                 # inputs: [B, L]
#                 # score: [B]
#                 end_flag = end_flag | (inputs[:, -1] == self.vocab.EOS_IDX)
#                 if end_flag.all():
#                     batch_decoded.append((inputs, score))
#                     continue
 
#                 outputs = self.model(memory, inputs)           # [B, L, V]
#                 outputs = outputs[:, -1]                                                # [B, V]
#                 log_probs = F.log_softmax(outputs, dim=-1)                              # [B, V]
#                 scores, indices = log_probs.topk(k=beamsize, dim=-1)                    # [B, K]
 
#                 scores = scores.masked_fill(end_flag.unsqueeze(-1), 0.0)
#                 indices = indices.masked_fill(end_flag.unsqueeze(-1), self.vocab.PAD_IDX)
 
#                 inputs = inputs.unsqueeze(-1)       # [B, L, 1]
#                 indices = indices.unsqueeze(1)      # [B, 1, K]
#                 inputs = torch.cat((inputs.repeat(1, 1, beamsize), indices), dim=1)     # [B, L, K]
 
#                 scores = score + scores                                                 # [B, K]
 
#                 for beam in range(beamsize):
#                     next_candidate.append((inputs[..., beam], scores[..., beam]))
 
#             next_candidate = sorted(next_candidate, key=lambda x: x[1], reverse=True)
#             all_candidates = next_candidate[:beamsize]
 
#         batch_decoded = batch_decoded + all_candidates
#         batch_decoded = sorted(batch_decoded, key=lambda x: x[1], reverse=True)[:beamsize]
#         batch, scores = zip(*batch_decoded)
#         batch = [x.transpose(0, 1) for x in batch]  # [L, 1]
#         batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=self.vocab.PAD_IDX)  # [K, L, 1]
#         batch = batch.permute(2, 0, 1)
#         scores = torch.stack(scores, dim=-1)

def batch_translate_beam_search(test_loader, encoder, decoder, tokenizer, device, beam_size=2, candidates=1, max_seq_length=CFG.max_len):
    # img: NxCxHxW
    encoder.eval()
    decoder.eval()

    sos_token = tokenizer.stoi['<sos>']
    eos_token = tokenizer.stoi['<eos>']

    text_preds = []

    tk0 = tqdm(test_loader, total = len(test_loader))
    for images in tk0:
        images = images.to(device)
        with torch.no_grad():
            enc_inputs = encoder(images)
            for i in range(enc_inputs.size(0)):#batch size
                enc_input = enc_inputs[i,:,:].repeat(beam_size, 1, 1) 
                sent = beamsearch(enc_input, decoder, beam_size, candidates, max_seq_length, sos_token, eos_token)
                _text_preds = tokenizer.predict_caption(sent)
                text_preds.append([_text_preds])

    text_preds = np.concatenate(text_preds)

    return text_preds

def translate_beam_search(test_loader, encoder, decoder, tokenizer, device, beam_size=2, candidates=1, max_seq_length=CFG.max_len):
    # img: 1xCxHxW
    encoder.eval()
    decoder.eval()

    sos_token = tokenizer.stoi['<sos>']
    eos_token = tokenizer.stoi['<eos>']

    text_preds = []
    tk0 = tqdm(test_loader, total = len(test_loader))
    for images in tk0:
        images = images.to(device)
        with torch.no_grad():
            enc_input = encoder(images)
            sent = beamsearch(enc_input, decoder, beam_size, candidates, max_seq_length, sos_token, eos_token)
            _text_preds = tokenizer.predict_caption(sent)
            
            text_preds.append([_text_preds])

    text_preds = np.concatenate(text_preds)
    return text_preds
        
def beamsearch(memory, decoder, beam_size, candidates, max_seq_length, sos_token, eos_token):    
    decoder.eval()
    device = memory.device

    beam = Beam(beam_size=beam_size, min_length=3, n_top=1, ranker=None, \
                start_token_id=sos_token, end_token_id=eos_token)

    with torch.no_grad():
        memory = memory.repeat(beam_size, 1, 1) # batch_size*beamsize, img_max_len, image_dim

        for i in range(max_seq_length):
            tgt_inp = beam.get_current_state().to(device)
            decoder_outputs = decoder.forward_one(memory, tgt_inp)
            log_prob = F.log_softmax(decoder_outputs[:, -1, :].squeeze(0), dim=-1)
            beam.advance(log_prob.cpu())
            
            if beam.done():
                break
                
        scores, ks = beam.sort_finished(minimum=1)
    
        hypothesises = []
        for i, (times, k) in enumerate(ks[:candidates]):
            hypothesis = beam.get_hypothesis(times, k)
            hypothesises.append(hypothesis)
    
    return [int(i) for i in hypothesises[0]]

def inference(test_loader, encoder, decoder, tokenizer, device):
    
    encoder.eval()
    decoder.eval()

    text_preds = []
    tk0 = tqdm(test_loader, total = len(test_loader))
    
    for images in tk0:
        
        images = images.to(device)
        
        with torch.no_grad():
            features = encoder(images)
            predictions, _ = decoder.predict(features, tokenizer)
        
        _text_preds        = tokenizer.predict_captions(predictions.detach().cpu().numpy())
        text_preds.append(_text_preds)
        
    text_preds = np.concatenate(text_preds)
    
    return text_preds


def ensemble_inference(test_loader, tokenizer, device):
    path1 = './swin_transformerdeocder_fold0_best.pth'
    path2 = './swin_transformerdeocder_fold1_best.pth'
    net = EnsembleNet(path1, path2, tokenizer, device)

    text_preds = []
    tk0 = tqdm(test_loader, total = len(test_loader))
    
    for images in tk0:
        images = images.to(device)
        with torch.no_grad():
            predictions = net.predict(images)
        
        _text_preds        = tokenizer.predict_captions(predictions.detach().cpu().numpy())
        text_preds.append(_text_preds)
        
    text_preds = np.concatenate(text_preds)
    
    return text_preds

def inference_test_data(used_beam=False):
    # ------------------READ DATA---------------
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

    encoder = CNN(is_pretrained=False, type_=CFG.model_name)
    encoder.load_state_dict(states['encoder'])
    encoder.to(device)

    decoder = Net(  max_length                   = CFG.max_len,
                        embed_dim                    = CFG.embed_dim,
                        vocab_size                   = len(tokenizer),
                        decoder_dim                  = CFG.decoder_dim,
                        ff_dim                       = CFG.ff_dim,
                        num_head                     = CFG.num_head,
                        num_layer                    = CFG.num_layer,
                        device                       = device)
    decoder.load_state_dict(states['decoder'])
    decoder.to(device)

    del states
    import gc
    gc.collect()

    # ====================================================
    # inference
    # ====================================================

    test_dataset = TestDataset(test, transform = get_transforms(data = 'valid'))
    test_loader  = DataLoader(test_dataset, batch_size = 32 if used_beam else 64, shuffle = False, num_workers = CFG.num_workers)
    t1 = time.time()
    if used_beam:
        predictions  = batch_translate_beam_search(test_loader, encoder, decoder, tokenizer, device, beam_size=2)
    else:
        predictions  = inference(test_loader, encoder, decoder, tokenizer, device)
        # predictions  = ensemble_inference(test_loader, tokenizer, device)
    print("Time to inference test data: ", time.time()-t1)
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
        data.append({'id': id, 'captions': captions[1:]})
    with open('results.json', 'w') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)


def inference_train_data():
    # ------------------READ DATA---------------
    df = pd.read_csv('./train_files/train_captions.csv')

    def read_data(data_frame):
        for i, caption in enumerate(data_frame['captions'].values):
            # caption = fix_error(caption)
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


    train = read_data(df)

    def get_train_file_path(image_id):
        return CFG.train_path + "/images_train/{}".format(image_id)

    train['file_path'] = train['id'].apply(get_train_file_path)
    print(f'train.shape: {train.shape}')

    # ====================================================
    # load model
    # ====================================================
        
    states = torch.load(CFG.prev_model, map_location = torch.device('cpu'))

    encoder = CNN(is_pretrained=False, type_=CFG.model_name)
    encoder.load_state_dict(states['encoder'])
    encoder.to(device)

    decoder = Net(  max_length                   = CFG.max_len,
                        embed_dim                    = CFG.embed_dim,
                        vocab_size                   = len(tokenizer),
                        decoder_dim                  = CFG.decoder_dim,
                        ff_dim                       = CFG.ff_dim,
                        num_head                     = CFG.num_head,
                        num_layer                    = CFG.num_layer,
                        device                       = device)
    decoder.load_state_dict(states['decoder'])
    decoder.to(device)

    del states
    import gc
    gc.collect()

    # ====================================================
    # inference
    # ====================================================

    train_dataset = TestDataset(train, transform = get_transforms(data = 'train'))
    train_loader  = DataLoader(train_dataset, batch_size = 64, shuffle = False, num_workers = CFG.num_workers)
    t1 = time.time()
    predictions  = inference(train_loader, encoder, decoder, tokenizer, device)
    predictions = [f"{text[1:]}" for text in predictions]
    print("Time to inference train data: ", time.time()-t1)
    print(get_score_bleu(train['captions'].values, predictions))

if __name__ == "__main__":
    inference_test_data(used_beam=False)
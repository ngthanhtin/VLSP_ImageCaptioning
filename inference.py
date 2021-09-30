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
from dataset import TestDataset
from transformation import get_transforms
from models.model import CNN, DecoderWithAttention

device = CFG.device
tokenizer = torch.load('./tokenizers/tokenizer_vi_fix_spelling.pth')

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(seed = CFG.seed)

# def beam_search_predictions(image, beam_index = 3):
#     start = [wordtoix["startseq"]]
#     start_word = [[start, 0.0]]
#     while len(start_word[0][0]) < max_length:
#         temp = []
#         for s in start_word:
#             par_caps = sequence.pad_sequences([s[0]], maxlen=max_length, padding='post')
#             preds = model.predict([image,par_caps], verbose=0)
#             word_preds = np.argsort(preds[0])[-beam_index:]
#             # Getting the top <beam_index>(n) predictions and creating a 
#             # new list so as to put them via the model again
#             for w in word_preds:
#                 next_cap, prob = s[0][:], s[1]
#                 next_cap.append(w)
#                 prob += preds[0][w]
#                 temp.append([next_cap, prob])
                    
#         start_word = temp
#         # Sorting according to the probabilities
#         start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
#         # Getting the top words
#         start_word = start_word[-beam_index:]
    
#     start_word = start_word[-1][0]
#     intermediate_caption = [ixtoword[i] for i in start_word]
#     final_caption = []
    
#     for i in intermediate_caption:
#         if i != 'endseq':
#             final_caption.append(i)
#         else:
#             break
    
#     final_caption = ' '.join(final_caption[1:])
#     return final_caption

def inference_with_batched_beam_search(test_loader, encoder, decoder, tokenizer, device, beam_size=3):

    encoder.eval()
    decoder.eval()

    # Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

    hypotheses = list()

    tk0 = tqdm(test_loader,desc="EVALUATING AT BEAM SIZE " + str(beam_size), total = len(test_loader))
    
    # For each image
    for image in tk0:
        k = beam_size

        # Move to GPU device, if available
        image = image.to(device)  # (1, 3, 256, 256)
        
        # Encode
        encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(2)

        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[tokenizer.stoi['<sos>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = decoder.init_hidden_state(encoder_out)
        h, c = [hi.squeeze(0) for hi in h], [ci.squeeze(0) for ci in c]
        
        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(k_prev_words)  # (s, embed_dim)
            if embeddings.dim() == 3:
                embeddings = embeddings.squeeze(1)

            awe, _ = decoder.attention(encoder_out, h[-1])  # (s, encoder_dim), (s, num_pixels)

            gate = decoder.sigmoid(decoder.f_beta(h[-1]))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            input = torch.cat([embeddings, awe], dim = 1)
            for j, rnn in enumerate(decoder.decode_step):
                at_h, at_c = rnn(input, (h[j], c[j]))  # (s, decoder_dim)
                input = decoder.dropout(at_h)
                h[j]  = at_h
                c[j]  = at_c

            # h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = decoder.fc(h[-1])  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / len(tokenizer)  # (s)
            next_word_inds = top_k_words % len(tokenizer)  # (s)
            prev_word_inds = prev_word_inds.long()
            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != tokenizer.stoi['<eos>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            incomplete_inds = torch.Tensor(incomplete_inds).to(device)
            incomplete_inds = incomplete_inds.long()
            h[0] = h[0][prev_word_inds[incomplete_inds]]
            h[-1] = h[-1][prev_word_inds[incomplete_inds]]
            c[0] = c[0][prev_word_inds[incomplete_inds]]
            c[-1] = c[-1][prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
            
            # Break if things have been going on too long
            if step > 50:
                break
            step += 1
        
    
        if len(complete_seqs_scores) == 0:
            print("Can not predict!......")
            hypotheses.append([""])
            continue

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        pred_index = [w for w in seq if w not in {tokenizer.stoi['<sos>'], tokenizer.stoi['<eos>'], tokenizer.stoi['<pad>']}]
        pred_text = [tokenizer.predict_caption(pred_index)]

        # Hypotheses
        hypotheses.append(pred_text)

    hypotheses = np.concatenate(hypotheses)
    
    return hypotheses

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

def ensemble_inference(test_loader, encoder1, decoder1, encoder2, decoder2, tokenizer, device):
    
    encoder1.eval()
    decoder1.eval()
    encoder2.eval()
    decoder2.eval()

    text_preds = []
    tk0 = tqdm(test_loader, total = len(test_loader))
    
    # prepare an array of exponentially decreasing weights
    alpha = 2.0
    weights = [np.exp(-i/alpha) for i in range(1, 3)]

    for images in tk0:
        
        images = images.to(device)
        
        with torch.no_grad():
            features1 = encoder1(images)
            predictions1 = decoder1.predict(features1, CFG.max_len, tokenizer)

            features2 = encoder2(images)
            predictions2 = decoder2.predict(features2, CFG.max_len, tokenizer)

            predictions = (weights[0]*predictions1 + weights[1]*predictions2)/2.

        predicted_sequence = torch.argmax(predictions.detach().cpu(), -1).numpy()
        _text_preds = tokenizer.predict_captions(predicted_sequence)
        text_preds.append(_text_preds)
        
    text_preds = np.concatenate(text_preds)
    
    return text_preds
    
# ------------------READ DATA---------------
df = pd.read_csv('../data/vietcap4h-public-test/test_captions.csv')

def get_test_file_path(image_id):
    return CFG.test_path + "/images_public_test/{}".format(image_id)

def get_test_id(path_file):
    return path_file.split('/')[-1]
test = df

test['file_path'] = test['id'].apply(get_test_file_path)
print(f'test.shape: {test.shape}')

test_dataset = TestDataset(test, transform = get_transforms(data = 'valid'))
test_loader  = DataLoader(test_dataset, batch_size = 1, shuffle = False, num_workers = CFG.num_workers)
# ====================================================
# load model
# ====================================================

if CFG.ensemble == False:
    states = torch.load(CFG.prev_model, map_location = torch.device('cpu'))

    encoder = CNN(is_pretrained=False, type_=CFG.model_name)
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

    # Inference
    predictions  = inference_with_batched_beam_search(test_loader, encoder, decoder, tokenizer, device, beam_size=2)
else:
    print("Predicting with Ensemble.....")
    model1 = './pretrained_models/swin_fold1_best.pth'
    model2 = './pretrained_models/swin_fold0_best.pth'

    states1 = torch.load(model1, map_location = torch.device('cpu'))

    encoder1 = CNN(is_pretrained=False, type_='swin')
    encoder1.load_state_dict(states1['encoder'])
    encoder1.to(device)

    decoder1 = DecoderWithAttention(attention_dim = CFG.attention_dim, 
                                embed_dim     = CFG.embed_dim, 
                                encoder_dim   = 1024,
                                decoder_dim   = CFG.decoder_dim,
                                num_layers    = CFG.decoder_layers,
                                vocab_size    = len(tokenizer), 
                                dropout       = CFG.dropout, 
                                device        = device)
    decoder1.load_state_dict(states1['decoder'])
    decoder1.to(device)

    del states1
    import gc
    gc.collect()

    states2 = torch.load(model2, map_location = torch.device('cpu'))

    encoder2 = CNN(is_pretrained=False, type_='swin')
    encoder2.load_state_dict(states2['encoder'])
    encoder2.to(device)

    decoder2 = DecoderWithAttention(attention_dim = CFG.attention_dim, 
                                embed_dim     = CFG.embed_dim, 
                                encoder_dim   = 1024,
                                decoder_dim   = CFG.decoder_dim,
                                num_layers    = CFG.decoder_layers,
                                vocab_size    = len(tokenizer), 
                                dropout       = CFG.dropout, 
                                device        = device)
    decoder2.load_state_dict(states2['decoder'])
    decoder2.to(device)

    del states2
    import gc
    gc.collect()

    # Inference
    predictions  = ensemble_inference(test_loader, encoder1, decoder1, encoder2, decoder2, tokenizer, device)

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
    data.append({'id': id, 'captions': captions[2:]})
with open('results.json', 'w') as outfile:
    json.dump(data, outfile, ensure_ascii=False, indent=4)
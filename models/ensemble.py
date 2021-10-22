import torch.nn as nn
import torch

from models.model import CNN, DecoderWithAttention
from typing import Tuple, Dict, Optional
from config import CFG
from tqdm.auto import tqdm
import numpy as np

class EnsembleNet(nn.Module):
    def __init__(self, model1_path, model2_path, tokenizer, device):
        # 1: swin, 2: vit
        super(EnsembleNet, self).__init__()
        self.device = device
        self.tokenizer = tokenizer

        # ====================================================
        # load model 1
        # ====================================================
            
        states1 = torch.load(model1_path, map_location = torch.device('cpu'))

        self.encoder1 = CNN(is_pretrained=False, type_='swin')
        self.encoder1.load_state_dict(states1['encoder'])
        self.encoder1.to(device)

        self.decoder1 = DecoderWithAttention(attention_dim = CFG.attention_dim, 
                                            embed_dim     = CFG.embed_dim, 
                                            encoder_dim   = CFG.enc_size,
                                            decoder_dim   = CFG.decoder_dim,
                                            num_layers    = CFG.decoder_layers,
                                            vocab_size    = len(tokenizer), 
                                            dropout       = CFG.dropout, 
                                            device        = device)

        self.decoder1.load_state_dict(states1['decoder'])
        self.decoder1.to(device)

        del states1
        import gc
        gc.collect()

        # ====================================================
        # load model 1
        # ====================================================

        states2 = torch.load(model2_path, map_location = torch.device('cpu'))

        self.encoder2 = CNN(is_pretrained=False, type_='swin')
        self.encoder2.load_state_dict(states2['encoder'])
        self.encoder2.to(device)

        self.decoder2 = DecoderWithAttention(attention_dim = CFG.attention_dim, 
                                            embed_dim     = CFG.embed_dim, 
                                            encoder_dim   = CFG.enc_size,
                                            decoder_dim   = CFG.decoder_dim,
                                            num_layers    = CFG.decoder_layers,
                                            vocab_size    = len(tokenizer), 
                                            dropout       = CFG.dropout, 
                                            device        = device)
        self.decoder2.load_state_dict(states2['decoder'])
        self.decoder2.to(device)

        del states2
        import gc
        gc.collect()

    def predict(self, image):
        return 1
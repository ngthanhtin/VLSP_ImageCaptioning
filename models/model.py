import torch
import torch.nn as nn

import timm
from fairseq.models import *
from fairseq.modules import *

import math
import numpy as np
from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from fairseq.models import *
from fairseq.modules import *

from models.vit import vit_base_patch16_224
from models.swin import swin_base_patch4_window7_224, swin_large_patch4_window7_224

class CNN(nn.Module):
    def __init__(self, is_pretrained=True, type_='vit'):
        super(CNN, self).__init__()
        
        if type_ == 'vit':
            self.e = vit_base_patch16_224(pretrained=is_pretrained)
        elif type_ == 'swin':
            self.e = swin_base_patch4_window7_224(pretrained=is_pretrained)

        for p in self.e.parameters():
            p.requires_grad = True#False

    def forward(self, image):      
        x = self.e.forward_features(image)

        return x


#https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute
class Namespace(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

# ------------------------------------------------------
class PositionEncode1D(nn.Module):
    def __init__(self, dim, max_length):
        super().__init__()
        assert (dim % 2 == 0)
        self.max_length = max_length

        d = torch.exp(torch.arange(0., dim, 2) * (-math.log(10000.0) / dim))
        position = torch.arange(0., max_length).unsqueeze(1)
        pos = torch.zeros(1, max_length, dim)
        #pos.require_grad = False
        pos[0, :, 0::2] = torch.sin(position * d)
        pos[0, :, 1::2] = torch.cos(position * d)
        self.register_buffer('pos', pos)

    def forward(self, x):
        batch_size, T, dim = x.shape
        x = x + self.pos[:,:T]
        return x


# https://github.com/pytorch/fairseq/blob/master/fairseq/modules/transformer_layer.py
class TransformerEncode(FairseqEncoder):

    def __init__(self, dim, ff_dim, num_head, num_layer):
        super().__init__({})
        #print('my TransformerEncode()')

        self.layer = nn.ModuleList([
            TransformerEncoderLayer(Namespace({
                'encoder_embed_dim': dim,
                'encoder_attention_heads': num_head,
                'attention_dropout': 0.5,
                'dropout': 0.5,
                'encoder_normalize_before': True,
                'encoder_ffn_embed_dim': ff_dim,
            })) for i in range(num_layer)
        ])
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):# T x B x C
        #print('my TransformerEncode forward()')
        for layer in self.layer:
            x = layer(x)
        x = self.layer_norm(x)
        return x

class TransformerDecode(FairseqIncrementalDecoder):
    def __init__(self, dim, ff_dim, num_head, num_layer):
        super().__init__({})

        self.layer = nn.ModuleList([
            TransformerDecoderLayer(Namespace({
                'decoder_embed_dim': dim,
                'decoder_attention_heads': num_head,
                'attention_dropout': 0.5, #0.5
                'dropout': 0.5,# 0.5
                'decoder_normalize_before': True,
                'decoder_ffn_embed_dim': ff_dim,
                # 'decoder_learned_pos': True,
                # 'cross_self_attention': True,
                'activation-fn': 'gelu',
            })) for i in range(num_layer)
        ])
        self.layer_norm = nn.LayerNorm(dim)


    def forward(self, x, mem, x_mask):
        for layer in self.layer:
            x = layer(x, mem, self_attn_mask=x_mask)[0]
        x = self.layer_norm(x)
        
        return x  # T x B x C

    def forward_one(self,
            x   : torch.Tensor,
            mem : torch.Tensor,
            incremental_state : Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]]
    )-> torch.Tensor:
        x = x[-1:]

        for layer in self.layer:
            x = layer(x, mem, incremental_state=incremental_state)[0]
        x = self.layer_norm(x)
     
        return x


class Net(nn.Module):

    def __init__(self,max_length,embed_dim,vocab_size,decoder_dim,ff_dim,num_head,num_layer, device):
        super(Net, self).__init__()
        self.device = device
        self.max_len = max_length
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.decoder_dim = decoder_dim
        self.ff_dim = ff_dim
        self.num_head = num_head
        self.num_layer = num_layer
        
        self.image_encode = nn.Identity()
        # ---
        self.text_pos = PositionEncode1D(embed_dim, max_length)
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.text_decode = TransformerDecode(decoder_dim, ff_dim, num_head, num_layer)

        # ---
        self.logit = nn.Linear(decoder_dim, vocab_size) # embeded_dim, vocab_size moi dung!~!!!!!!!
        self.dropout = nn.Dropout(p=0.5)

        # ----
        # initialization
        self.token_embed.weight.data.uniform_(-0.1, 0.1)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-0.1, 0.1)

    def forward_one(self, encoder_out, encoded_captions):

        image_embed = self.image_encode(encoder_out).permute(1, 0, 2).contiguous() # img_len, batch_size, channels
        
        text_embed = self.token_embed(encoded_captions)
        text_embed = self.text_pos(text_embed).permute(1, 0, 2).contiguous() # text_length,batch_size, text_embedding

        text_mask = np.triu(np.ones((text_embed.size(0), text_embed.size(0))), k=1).astype(np.uint8) # text_length, text_length
        text_mask = torch.autograd.Variable(torch.from_numpy(text_mask) == 1).to(self.device)

        x = self.text_decode(text_embed[:text_embed.size(0)], image_embed, text_mask) # text_length,batch_size, text_embedding
        
        x = x.permute(1, 0, 2).contiguous() # batchsize, text_length, embedding_dim

        logit = self.logit(x) # batchsize, text_length, vocab_size
    
        return logit

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim = 0, descending = True)
        encoder_out      = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        decode_lengths = (caption_lengths - 1).tolist()

        image_embed = self.image_encode(encoder_out).permute(1, 0, 2).contiguous() # img_len, batch_size, channels
        
        text_embed = self.token_embed(encoded_captions)
        text_embed = self.text_pos(text_embed).permute(1, 0, 2).contiguous() # text_length,batch_size, text_embedding

        text_mask = np.triu(np.ones((text_embed.size(0), text_embed.size(0))), k=1).astype(np.uint8) # text_length, text_length
        text_mask = torch.autograd.Variable(torch.from_numpy(text_mask) == 1).to(self.device)

        x = self.text_decode(text_embed, image_embed, text_mask) # text_length,batch_size, text_embedding
        
        x = x.permute(1, 0, 2).contiguous() # batchsize, text_length, embedding_dim
        x = self.dropout(x)

        logit = self.logit(x) # batchsize, text_length, vocab_size
    
        return logit, encoded_captions, decode_lengths

    def predict(self, encoder_out, tokenizer):
        # argmax decode
        batch_size = encoder_out.size(0)
        
        image_embed = self.image_encode(encoder_out).permute(1, 0, 2).contiguous() # (img_len,bs,image_dim)

        token = torch.full((batch_size, self.max_len), tokenizer.stoi['<pad>'], dtype=torch.long, device=self.device) # (batch_size,max_len) 
        text_pos = self.text_pos.pos #(1,sequence_len,text_dim) torch.zeros(1, max_length, dim)
        token[:, 0] = tokenizer.stoi['<sos>']
        
        # -------------------------------------
        eos = tokenizer.stoi['<eos>']
        pad = tokenizer.stoi['<pad>']

        # fast version
        if 1:
            # incremental_state = {}
            incremental_state = torch.jit.annotate(
                Dict[str, Dict[str, Optional[torch.Tensor]]],
                torch.jit.annotate(Dict[str, Dict[str, Optional[torch.Tensor]]], {}),
            )
            for t in range(self.max_len - 1):
                last_token = token[:, t] # take the whole batch's t'th token
                text_embed = self.token_embed(last_token) #[bs,text_dim] Generate embedding for the t'th token
                text_embed = text_embed + text_pos[:, t]  #[bs,text_dim] Combine with pos embed for t'th token

                text_embed = text_embed.reshape(1, batch_size, self.embed_dim)
                
                #text_embed ---> 1,bs,text_dim
                #image_embed ---> img_pos_embed,bs,image_dim
                x = self.text_decode.forward_one(text_embed, image_embed, incremental_state)
                ## x -----> (1,bs,text_dim)
                x = x.reshape(batch_size, self.decoder_dim)
                ## x -----> (bs,decoder_dim)
                l = self.logit(x)
                ## l -----> (bs,num_classes)
                k = torch.argmax(l, -1)
                token[:, t + 1] = k
                if ((k == eos) | (k == pad)).all():
                    break
    
        predict = token[:, 1:]
        return predict, l
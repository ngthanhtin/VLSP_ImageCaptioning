import torch.nn as nn
import torch

from models.model import CNN, Net
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

        ######################################################
        image_dim1 = 1536#1024
        text_dim1 = 1536#1024
        decoder_dim1 = 1536#1024

        image_dim2 = 1536#768
        text_dim2 = 1536#768
        decoder_dim2 = 1536#768
        # num_layer = 4 #3
        # num_head = 8
        # ff_dim = 2048#1024
        # ====================================================
        # load model 1
        # ====================================================
            
        states1 = torch.load(model1_path, map_location = torch.device('cpu'))

        self.encoder1 = CNN(is_pretrained=False, type_='swin')
        self.encoder1.load_state_dict(states1['encoder'])
        self.encoder1.to(device)

        self.decoder1 = Net(max_length                   = CFG.max_len,
                            embed_dim                    = text_dim1,
                            vocab_size                   = len(tokenizer),
                            decoder_dim                  = decoder_dim1,
                            ff_dim                       = CFG.ff_dim,
                            num_head                     = CFG.num_head,
                            num_layer                    = CFG.num_layer,
                            device                       = device)
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

        self.decoder2 = Net(     max_length              = CFG.max_len,
                            embed_dim                    = text_dim2,
                            vocab_size                   = len(tokenizer),
                            decoder_dim                  = decoder_dim2,
                            ff_dim                       = CFG.ff_dim,
                            num_head                     = CFG.num_head,
                            num_layer                    = CFG.num_layer,
                            device                       = device)
        self.decoder2.load_state_dict(states2['decoder'])
        self.decoder2.to(device)

        del states2
        import gc
        gc.collect()

    def predict(self, image):
        # argmax decode
        max_length = CFG.max_len
        batch_size = len(image)
        
        image_enc1 = self.encoder1(image)
        image_embed1 = self.decoder1.image_encode(image_enc1).permute(1, 0, 2).contiguous() # (img_len,bs,image_dim)

        image_enc2 = self.encoder2(image)
        image_embed2 = self.decoder2.image_encode(image_enc2).permute(1, 0, 2).contiguous() # (img_len,bs,image_dim)

        token = torch.full((batch_size, max_length), self.tokenizer.stoi['<pad>'], dtype=torch.long, device=self.device) # (batch_size,max_len) 
        text_pos_1 = self.decoder1.text_pos.pos #(1,sequence_len,text_dim) torch.zeros(1, max_length, dim)
        text_pos_2 = self.decoder2.text_pos.pos #(1,sequence_len,text_dim) torch.zeros(1, max_length, dim)
        token[:, 0] = self.tokenizer.stoi['<sos>']

        eos = self.tokenizer.stoi['<eos>']
        pad = self.tokenizer.stoi['<pad>']

        # fast version
        if 1:
            # incremental_state = {}
            incremental_state1 = torch.jit.annotate(
                Dict[str, Dict[str, Optional[torch.Tensor]]],
                torch.jit.annotate(Dict[str, Dict[str, Optional[torch.Tensor]]], {}),
            )
            incremental_state2 = torch.jit.annotate(
                Dict[str, Dict[str, Optional[torch.Tensor]]],
                torch.jit.annotate(Dict[str, Dict[str, Optional[torch.Tensor]]], {}),
            )
            
            for t in range(max_length - 1):
                last_token_1 = token[:, t] # take the whole batch's t'th token
                text_embed_1 = self.decoder1.token_embed(last_token_1) #[bs,text_dim] Generate embedding for the t'th token
                text_embed_1 = text_embed_1 + text_pos_1[:, t]  #[bs,text_dim] Combine with pos embed for t'th token
                text_embed_1 = text_embed_1.reshape(1, batch_size, self.decoder1.embed_dim)

                last_token_2 = token[:, t] # take the whole batch's t'th token
                text_embed_2 = self.decoder2.token_embed(last_token_2) #[bs,text_dim] Generate embedding for the t'th token
                text_embed_2 = text_embed_2 + text_pos_2[:, t]  #[bs,text_dim] Combine with pos embed for t'th token
                text_embed_2 = text_embed_2.reshape(1, batch_size, self.decoder2.embed_dim)

                #text_embed ---> 1,bs,text_dim(768)
                #image_embed ---> img_pos_embed,bs,image_im
                x_1 = self.decoder1.text_decode.forward_one(text_embed_1, image_embed1, incremental_state1)
                x_2 = self.decoder2.text_decode.forward_one(text_embed_2, image_embed2, incremental_state2)
                ## x -----> (1,bs,text_dim)
                x_1 = x_1.reshape(batch_size, self.decoder1.embed_dim)
                x_2 = x_2.reshape(batch_size, self.decoder2.embed_dim)
                ## x -----> (bs,decoder_dim)
                l_1 = self.decoder1.logit(x_1)
                l_2 = self.decoder2.logit(x_2)

                l = 0.45*l_1 + 0.55*l_2
                ## l -----> (bs,num_classes)
                k = torch.argmax(l, -1)
                token[:, t + 1] = k
                if ((k == eos) | (k == pad)).all():
                    break

        predict = token[:, 1:]
        return predict
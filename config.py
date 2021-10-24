#  n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
#   'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
#   'efficientnet-b6': 2304, 'efficientnet-b7': 2560}

# This is not, to put it mildly, the most elegant solution ever - but I ran into some trouble 
# with checking the size of feature spaces programmatically inside the CFG definition.
import torch


# == 'vit':

# image_dim = 768
# text_dim = 768
# decoder_dim = 768
# num_layer = 4 #3
# num_head = 8
# ff_dim = 2048#1024
        
# 'swin'
# image_dim = 1024
# text_dim = 1024
# decoder_dim = 1024
# num_layer = 6 #3
# num_head = 8
# ff_dim = 2048#1024

class CFG:
    debug          = False
    apex           = False
    teacher_forcing= False
    max_len        = 40
    print_freq     = 100
    num_workers    = 4
    model_name     = 'swin'
    enc_size       = 1024 #768 #1536
    samp_size      = 10
    size           = 224 # 288 image size
    ff_dim         = 2048
    num_head       = 8
    num_layer      = 5 # 5
    scheduler      = 'CosineAnnealingWarmRestarts' 
    epochs         = 12
    T_0            = 50
    T_max          = 4  
    encoder_lr     = 1e-4
    decoder_lr     = 1e-4
    min_lr         = 1e-6
    batch_size     = 12
    weight_decay   = 1e-5
    gradient_accumulation_steps = 1
    max_grad_norm  = 100
    attention_dim  = 256
    embed_dim      = 1024
    decoder_dim    = 1024
    decoder_layers = 2     # number of LSTM layers
    dropout        = 0.5
    seed           = 42
    n_fold         = 4
    trn_fold       = [0]
    train          = True
    train_path     = '../data/viecap4h-public-train/viecap4h-public-train/'
    test_path      = '../data/vietcap4h-public-test/'
    prep_path      = './preprocessed-stuff/'
    prev_model     =  './swin_transformerdeocder_fold0_best.pth'
    tokenizer_path = './tokenizers/tokenizer_vi_fix_error_english2.pth'
    device         = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
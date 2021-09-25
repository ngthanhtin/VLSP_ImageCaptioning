#  n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
#   'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
#   'efficientnet-b6': 2304, 'efficientnet-b7': 2560}

# This is not, to put it mildly, the most elegant solution ever - but I ran into some trouble 
# with checking the size of feature spaces programmatically inside the CFG definition.
import torch

class CFG:
    debug          = False
    apex           = False
    max_len        = 36
    print_freq     = 100
    num_workers    = 4
    model_name     = 'swintransformer'
    enc_size       = 1024 #1408
    samp_size      = 10
    size           = 224 # 288 image size
    scheduler      = 'CosineAnnealingWarmRestarts' 
    epochs         = 25
    T_0            = 50
    T_max          = 4  
    encoder_lr     = 1e-4
    decoder_lr     = 4e-4
    min_lr         = 1e-6
    batch_size     = 32
    weight_decay   = 1e-6
    gradient_accumulation_steps = 1
    max_grad_norm  = 100
    attention_dim  = 256
    embed_dim      = 512
    decoder_dim    = 512
    decoder_layers = 2     # number of LSTM layers
    dropout        = 0.5
    seed           = 42
    n_fold         = 4
    trn_fold       = 0
    train          = True
    train_path     = '../data/viecap4h-public-train/viecap4h-public-train/'
    test_path      = '../data/vietcap4h-public-test/'
    prep_path      = './preprocessed-stuff/'
    prev_model     = './swintransformer_b2_fold0_best2.pth'

    device         = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
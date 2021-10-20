# efficientnetv2-m:2152, #efficientnetb2: 1408, #efficientnetv2-s: 1792 
# vit_base_patch16_224: 768, vit_base_patch32_224_in21k: 768 =>vit_base_patch16_224 better
# swin_base_patch4_window7_224_in22k: 1024

import torch

class CFG:
    ensemble       = False
    noise_injection= True
    debug          = False
    apex           = False
    max_len        = 40 # 40
    print_freq     = 100
    num_workers    = 4
    model_name     = 'swin'
    enc_size       =  1536 #1024
    
    samp_size      = 10
    size           = 224 # 288 image size
    scheduler      = 'CosineAnnealingWarmRestarts' 
    epochs         = 15
    T_0            = 50
    T_max          = 4  
    encoder_lr     = 1e-4
    decoder_lr     = 4e-4
    min_lr         = 1e-6
    batch_size     = 16
    weight_decay   = 1e-6
    gradient_accumulation_steps = 1
    max_grad_norm  = 100
    attention_dim  = 256
    embed_dim      = 512 # 512
    decoder_dim    = 512 # 512
    decoder_layers = 2     # number of LSTM layers
    dropout        = 0.5
    seed           = 42
    n_fold         = 4
    trn_fold       = [0] #0 is best for tokenizer_vi_fix_error_english2
    train          = True
    train_path     = '../data/viecap4h-public-train/viecap4h-public-train/'
    test_path      = '../data/vietcap4h-private-test/vietcap4h-private-test/'
    # test_path      = '../data/vietcap4h-public-test/'
    prep_path      = './preprocessed-stuff/'
    tokenizer_path = './tokenizers/tokenizer_vi_fix_error_english2.pth'
    prev_model     =  '/home/tinvn/TIN/VLSP_ImageCaptioning/vlsp_code/pretrained_models/swin_fold0_epoch14_best_remove_english_0298.pth'

    device         = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
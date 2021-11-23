# VLSP2021 vieCap4H Challenge: Automatic image caption generation for healthcare domains in Vietnamese
# Dataset
https://aihub.vn/competitions/40#participate

![Drag Racing](https://github.com/ngthanhtin/VLSP_ImageCaptioning/blob/master/image/image_captioning_vlsp.png?raw=true)

# Dependencies
`pip install -r requirements.txt` <br/>
Download tokenizer from: [tokenizer link](https://drive.google.com/file/d/1de3lxn78g4OFWFwpDaFkg6L4jg4xaXpu/view?usp=sharing) and put it in: `tokenizers` folder. <br/>
Download pretrained weights from: [pretrained link](https://drive.google.com/file/d/1pI4h_REpyWQzcOvGJP7dQCf4lG4oCrwK/view?usp=sharing) and put it in `pretrained_models` folder.

# Prepare data
## Conversion
In this repo, I convert the json to csv format for simple processing.</br>
In `json2txt.py`, specify the caption `file_path` in `json` format, and caption destination `dest_path` in `csv` format. <br/>
The `file_path` will be the path to the `train_captions.json` or `sample_submission.json` or `private_sample_sub.json`.  <br/>
The `dest_path` will be the path containing the generated csv file, it is in the `train_files` folder. Noted that: the `train_captions.json` will generate `train_captions.csv`, the `sample_submission.json` will generate `test_captions.csv`, and the `private_sample_sub.json` will generate `private_captions.csv`.

## Config
In `config.py`, we specify these path to the provided data: <br/>
```
train_path     = '../data/viecap4h-public-train/viecap4h-public-train/'
# test_path      = '../data/vietcap4h-public-test/'
test_path      = '../data/vietcap4h-private-test/vietcap4h-private-test/'
tokenizer_path = './tokenizers/tokenizer_vi_fix_error_english2.pth'
prev_model     =  './pretrained_models/swin_fold1_epoch11_best_remove_english_292646.pth'
```
`train_path` is the path to the `images_train` folder of the train data. <br/>
`test_path` is the path to the `images_public_test` folder of the public test data or `images` folder of the private data. <br/>

`tokenizer_path` is the path to the tokenizer. <br/>
`prev_model` is the path to the pretrained model.

# Model
This code uses Swin Transformer (you can change to other types of Transformer) as the Encoder and LSTM Attention as the Decoder.

# Training
Run `train.py` to train with the train data. Make sure that you already have the tokenizers file.

To reproduce our result on the private board, please make sure to train `fold 2` which is specified in `config.py`, and get the weight at the `epoch 11`.

# Inference
Run `inference.py`. Noted that we currently dont support the ensemble version, but if you want to do it, set the `ensemble` parameter in `config.py` to True.

# Results
Achieved 0.302 BLEU4 for Public Dataset.</br>
Achieved 0.293 BLEU4 for Private Dataset. As a result, I achived the 3rd on the leaderboard.</br>

# Citation
If you find this code useful for your work, please consider citing:
```
@inproceedings{tin2021vieCap4H,
  title={Image Captioning Using Swin Transformer Encoder and LSTM Attention Decoder},
  author={Thanh Tin Nguyen},
  booktitle={VLSP2021 vieCap4H Challenge: Automatic image caption generation for healthcare domains in Vietnamese},
  year={2021}
}
```

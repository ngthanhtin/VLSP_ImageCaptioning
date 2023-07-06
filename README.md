# VLSP2021 vieCap4H Challenge: Automatic image caption generation for healthcare domains in Vietnamese
# Dataset
https://aihub.vn/competitions/40#participate

![Drag Racing](https://github.com/ngthanhtin/VLSP_ImageCaptioning/blob/master/image/image_captioning_vlsp.png?raw=true)

# Dependencies
`pip install torch torchvision torchaudio` <br/>
`pip install -r requirements.txt` <br/>
Download the tokenizer from: [tokenizer link](https://drive.google.com/file/d/1de3lxn78g4OFWFwpDaFkg6L4jg4xaXpu/view?usp=sharing) and put it in: `tokenizers` folder. <br/>
Download pre-trained weights from [pretrained link](https://drive.google.com/file/d/1pI4h_REpyWQzcOvGJP7dQCf4lG4oCrwK/view?usp=sharing) and put it in `pretrained_models` folder.

# Prepare data
## Conversion
In this repo, I convert the JSON to CSV format for simple processing.</br>
In `json2txt.py`, specify the caption `file_path` in `json` format, and caption destination `dest_path` in `csv` format. <br/>
The `file_path` will be the path to the `train_captions.json` or `sample_submission.json` or `private_sample_sub.json`.  <br/>
The `dest_path` will be the path containing the generated CSV file, it is in the `train_files` folder. Noted that: the `train_captions.json` will generate `train_captions.csv`, the `sample_submission.json` will generate `test_captions.csv`, and the `private_sample_sub.json` will generate `private_captions.csv`.

## Config
In `config.py`, we specify these paths to the provided data: <br/>
```
train_path     = '../data/viecap4h-public-train/viecap4h-public-train/'
# test_path      = '../data/vietcap4h-public-test/'
test_path      = '../data/vietcap4h-private-test/vietcap4h-private-test/'
tokenizer_path = './tokenizers/tokenizer_vi_fix_error_english2.pth'
prev_model     =  './pretrained_models/swin_fold1_epoch11_best_remove_english_292646.pth'
```
`train_path` is the path to the `images_train` folder of the train data. <br/>
`test_path` is the path to the `images_public_test` folder of the public test data or the `images` folder of the private data. <br/>

`tokenizer_path` is the path to the tokenizer. <br/>
`prev_model` is the path to the pre-trained model.

# Model
This code uses Swin Transformer (you can change to other types of Transformer) as the Encoder and LSTM Attention as the Decoder.

# Training
Run `train.py` to train with the train data. Make sure that you already have the tokenizer file.

To reproduce our result on the private board, please make sure to train `fold 2` which is specified in `config.py`, and get the weight at the `epoch 11`.

# Inference
Run `inference.py`. Noted that we currently dont support the ensemble version, but if you want to do it, set the `ensemble` parameter in `config.py` to True.

# Results
Achieved 0.302 BLEU4 for Public Dataset.</br>
Achieved 0.293 BLEU4 for Private Dataset. As a result, I achived the 3rd on the leaderboard.</br>

# Citation
If you find this code useful for your work, please consider citing:
## Preprint version
```
@article{DBLP:journals/corr/abs-2209-01304,
  author    = {Thanh Tin Nguyen and
               Long H. Nguyen and
               Nhat Truong Pham and
               Liu Tai Nguyen and
               Van Huong Do and
               Hai Nguyen and
               Ngoc Duy Nguyen},
  title     = {vieCap4H-VLSP 2021: Vietnamese Image Captioning for Healthcare Domain
               using Swin Transformer and Attention-based {LSTM}},
  journal   = {CoRR},
  volume    = {abs/2209.01304},
  year      = {2022},
  url       = {https://doi.org/10.48550/arXiv.2209.01304},
  doi       = {10.48550/arXiv.2209.01304},
  eprinttype = {arXiv},
  eprint    = {2209.01304},
  timestamp = {Mon, 26 Sep 2022 18:12:06 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2209-01304.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
## Published version
```
@article{JCSCE,
	author = {Nguyen Tin and Nguyen H. and Pham Truong and Nguyen Tai and Do Huong and Nguyen Hai and Nguyen Duy},
	title = {vieCap4H Challenge 2021: Vietnamese Image Captioning for Healthcare Domain using Swin Transformer and Attention-based LSTM},
	journal = {VNU Journal of Science: Computer Science and Communication Engineering},
	volume = {38},
	number = {2},
	year = {2022},
	keywords = {},
	abstract = {This study presents our approach to automatic Vietnamese image captioning for the healthcare domain in text processing tasks of Vietnamese Language and Speech Processing (VLSP) Challenge 2021, as shown in Figure~\ref\{fig:example\}. In recent years, image captioning often employs a convolutional neural network-based architecture as an encoder and a long short-term memory (LSTM) as a decoder to generate sentences. These models perform remarkably well in different datasets. Our proposed model also has an encoder and a decoder, but we instead use a Swin Transformer in the encoder, and a LSTM combined with an attention module in the decoder. The study presents our training experiments and techniques used during the competition. Our model achieves a BLEU4 score of 0.293 on the vietCap4H dataset, and the score is ranked the 3\$^\{rd\}\$ place on the private leaderboard. Our code can be found at \url\{https://git.io/JDdJm\}.},
	issn = {2588-1086},	doi = {10.25073/2588-1086/vnucsce.369},
	url = {//jcsce.vnu.edu.vn/index.php/jcsce/article/view/369}
}
```

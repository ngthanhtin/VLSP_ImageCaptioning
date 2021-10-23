# VLSP2021 vieCap4H Challenge: Automatic image caption generation for healthcare domains in Vietnamese
# Dataset
https://aihub.vn/competitions/40#participate

![Drag Racing](https://github.com/ngthanhtin/VLSP_ImageCaptioning/blob/master/image/image_captioning_vlsp.png?raw=true)

# Prepare data
In this repo, I convert the json to csv format for easy processing.</br>
In `json2txt.py`, specify the caption `file_path` in `json` format, and caption destination `dest_path` in `csv` format.

# Model
This code uses Swin Transformer (you can change to other types of Transformer) as the Encoder and LSTM Attention as the Decoder.

# Training
Run `train.py`.
# Inference
Run `inference.py`.

# Results
Achieved 0.302 BLEU4 for Public Dataset.</br>
Achieved 0.293 BLEU4 for Private Dataset. As a result, I achived the 4th on the leaderboard.</br>

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
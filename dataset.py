import os
OUTPUT_DIR = './'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader, Dataset


class TrainDataset(Dataset):
    def __init__(self, df, tokenizer, transform=None):
        super().__init__()
        self.df         = df
        self.tokenizer  = tokenizer
        self.file_paths = df['file_path'].values
        self.labels     = df['captions'].values
        self.transform  = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = cv2.imread(file_path)

        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        except:
            print('Error returned')
            
        if self.transform:
            augmented = self.transform(image = image)
            image     = augmented['image']
        label = self.labels[idx]
        label = self.tokenizer.text_to_sequence(label)

        label_length = len(label)
        label_length = torch.LongTensor([label_length])
        return image, torch.LongTensor(label), label_length
    

class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.df = df
        self.file_paths = df['file_path'].values
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = cv2.imread(file_path)
        
        # trick to overcome dead files
        if image is None:
            #check if jpg or png
            decode_type = file_path[-3:]
            new_path = ''
            if decode_type == 'jpg':
                new_path = file_path[:-4] + '.png'
                image = cv2.imread(new_path)
            else:
                new_path = file_path[:-4] + '.jpg'
                image = cv2.imread(new_path)
        
        if image is None:
            print(file_path)
            print(new_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image
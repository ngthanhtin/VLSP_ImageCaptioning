from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip, 
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout, 
    IAAAdditiveGaussianNoise, Transpose, Blur
    )
from albumentations.pytorch import ToTensorV2

from config import CFG

# transformations
def get_transforms(*, data):
    # tensor([0.5137, 0.4916, 0.4835]) tensor([0.2398, 0.2337, 0.2372])
    if data == 'train':
        return Compose([
            Resize(CFG.size, CFG.size),
            # HorizontalFlip(p=0.5),                  
#             Transpose(p=0.5),
#             HorizontalFlip(p=0.5),
#             VerticalFlip(p=0.5),
#             ShiftScaleRotate(p=0.5),   
            # Normalize(
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225],
            # ),
            Normalize(
                mean=[0.5137, 0.4916, 0.4835],
                std=[0.2398, 0.2337, 0.2372],
            ),
            ToTensorV2(),
        ])
    
    elif data == 'valid':
        return Compose([
            Resize(CFG.size, CFG.size),
            # Normalize(
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225],
            # ),
            Normalize(
                mean=[0.5137, 0.4916, 0.4835],
                std=[0.2398, 0.2337, 0.2372],
            ),
            ToTensorV2(),
        ])
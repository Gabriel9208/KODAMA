import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.utils.LetterBox import LetterBox

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]
train_transform = A.Compose(
    [
        LetterBox(target_size=(640, 640), fill_color=(255, 255, 255)),

        A.HorizontalFlip(
            p=0.5,                                    
        ),

        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(0.02, 0.2),              
            hole_width_range=(0.02, 0.2),               
            fill=255,                                     
            p=0.4,                                      
        ),

        A.HueSaturationValue(
            hue_shift_limit=round(0.015 * 360),
            sat_shift_limit=round(0.7   * 255),
            val_shift_limit=round(0.4   * 255),
            p=0.5,
        ),

        A.Affine(
            translate_percent={
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
            },
            scale=(1.0 - 0.5, 1.0 + 0.5),  
            rotate=(-15, 15),            
            interpolation=1,                           
            border_mode=0,           
            fill_mask=255,                  
            p=0.5,
        ), 
        A.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ToTensorV2(),                                   
    ]
)

val_transform = A.Compose([
    LetterBox(target_size=(640, 640), fill_color=(255, 255, 255)),
    A.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ToTensorV2(), 
])
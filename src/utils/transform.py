import random
import albumentations as A
from albumentations.pytorch import ToTensorV2

# YOLO default augmentation

train_transform = A.Compose(
    [
        A.Affine(
            translate_percent={
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
            },
            scale=(1.0 - 0.5, 1.0 + 0.5),              
            interpolation=1,                           
            border_mode=0,                             
            p=1.0,
        ),

        A.HorizontalFlip(
            p=0.5,                                    
        ),

        A.Lambda(
            image=lambda img, **kw: (
                img[:, :, ::-1].copy()                 
                if random.random() < 0.0                
                else img
            ),
            p=1.0,
        ),

        A.HueSaturationValue(
            hue_shift_limit=round(0.015 * 360),
            sat_shift_limit=round(0.7   * 255),
            val_shift_limit=round(0.4   * 255),
            p=1.0,
        ),

        A.RandAugment(
            n=2, m=9,
            p=1.0,
        ),

        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(0.02, 0.2),              
            hole_width_range=(0.02, 0.2),               
            fill=255,                                     
            p=0.4,                                      
        ),

        ToTensorV2(),                                   
    ]
)
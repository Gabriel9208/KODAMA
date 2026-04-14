import os

from ultralytics import YOLO
import wandb


wandb.login()

check_point = "runs/segment/KODAMA/yolo26n-seg_100_default_0/weights/last.pt"
if os.path.exists(check_point):
    model = YOLO(check_point)
    model.train(resume=True)
else:
    model = YOLO("model/yolo26n-seg.pt")

    train_config = {
        "epochs": 100
    }   

    data_aug = {   
        "mixup": 0.1,         
        "copy_paste": 0.2,    
        "degrees": 10.0,   
    }   

    # name: 100_64 -> 100 epochs, batch size 64
    model.train(
        name="yolo26n-seg_100_default_0",
        project="KODAMA",
        data="configs/sanpo.yaml",  

        cos_lr=True,    

        epochs=train_config["epochs"],

        imgsz=640,
        workers=8,
        patience=15,
        seed=42,
        device=[0, 1],  

        # data augmentation
        mixup=data_aug["mixup"],
        copy_paste=data_aug["copy_paste"],
        degrees=data_aug["degrees"],    

        exist_ok=True,
        pretrained=True,
        save=True,
        verbose=True,
        resume=True
    )

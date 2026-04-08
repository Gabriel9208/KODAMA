from ultralytics import YOLO
import wandb

wandb.login()

model = YOLO("model/yolo26n-seg.pt")

train_config = {
    "epochs": 100,
    "batch": 64,
    "lr0": 1e-3,
    "lrf": 1e-2,
    "weight_decay": 0.0005
}

data_aug = {   
    "mixup": 0.1,         
    "copy_paste": 0.2,    
    "degrees": 10.0,   
}

# name: 100_64 -> 100 epochs, batch size 64
model.train(
    name="yolo26n-seg_100_64",
    project="KODAMA",
    data="configs/sanpo.yaml",

    cos_lr=True,

    lr0=train_config["lr0"],
    lrf=train_config["lrf"],
    weight_decay=train_config["weight_decay"],
    epochs=train_config["epochs"],
    batch=train_config["batch"],

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

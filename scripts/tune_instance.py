from ultralytics import YOLO

model = YOLO("model/yolo26n-seg.pt")

search_space = {
    "lr0": (1e-5, 1e-2),
    "lrf": (1e-4, 1e-2),
    "weight_decay": (1e-3, 1e-2),
    "momentum": (0.7, 0.98),
    "batch": [32, 64, 128]
}

model.tune(
    data="configs/sanpo.yaml", 
    epochs=3, 
    iterations=20, 
    optimizer="AdamW", 
    space=search_space,

    device=[0, 1],
    imgsz=640,
    workers=8,

    project="KODAMA",
    name="yolo26n-seg_tune",
    seed=42,
    plots=False,
    save=False,
    val=False,
    resume=True
)
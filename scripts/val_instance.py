from ultralytics import YOLO

model = YOLO("runs/segment/KODAMA/yolo26n-seg_100_64/weights/best.pt")

metrics = model.val(
    data="configs/sanpo.yaml",
    split="val",
    batch=64,
    imgsz=640,
    device=[1],
    verbose=True,
)
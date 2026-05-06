import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from src.models.semantic_decoder import SementicDecoder
from src.models.feature_extractor import FeatureExtractor
from src.engine.trainer import Trainer, TrainerConfig
from src.engine.callbacks import CheckpointCallback, LoggingCallback
from src.datasets.SANPO_dataset import SANPO_dataset

cfg = TrainerConfig(
    max_epochs=20,
    device="cuda",
    grad_clip=1.0,
    log_every_n_steps=50,
    grad_accum_steps=4,
    amp_dtype="float16",
    checkpoint_dir="runs/semantic_decoder",
    resume_from="runs/semantic_decoder/last.pt",
)

rgb_dir = "data/images/train"
seg_dir = "data/labels/train_semantic"
#depth_dir = "data/images/train/depth_maps"
train_dataset = SANPO_dataset(rgb_dir, seg_dir)

rgb_dir = "data/images/val"
seg_dir = "data/labels/val_semantic"
#depth_dir = "data/images/val/depth_maps"

val_dataset = SANPO_dataset(rgb_dir, seg_dir)

train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=cfg.batch_size, 
    shuffle=True, 
    num_workers=cfg.num_workers
)

val_loader = torch.utils.data.DataLoader(
    val_dataset, 
    batch_size=cfg.batch_size, 
    shuffle=False, 
    num_workers=cfg.num_workers
)

model = YOLO("model/yolo26n-seg.pt")
check_point = "runs/segment/KODAMA/yolo26n-seg_100_64/weights/best.pt"
if os.path.exists(check_point):
    model = YOLO(check_point)
    print(f"Loading best checkpoint of instance seg network from {check_point}")

feature_extractor = FeatureExtractor(model=model, layers=[16, 19, 22])
decoder = SementicDecoder(num_classes=15, p3_channel=64, p4_channel=128, p5_channel=256)


optimizer = AdamW(
    decoder.parameters(),
    lr=cfg.learning_rate,
    weight_decay=cfg.weight_decay,
    betas=(0.9, 0.999),
)

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=10,
    eta_min=1e-6,
)


trainer = Trainer(
    feature_extractor, 
    decoder, 
    optimizer, 
    scheduler, 
    train_loader, 
    val_loader, 
    cfg,
    callbacks=[CheckpointCallback(cfg.checkpoint_dir),
    LoggingCallback(cfg.max_epochs)]
)
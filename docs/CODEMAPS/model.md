<!-- Generated: 2026-04-08 | Files scanned: 4 | Token estimate: ~600 -->

# Model Architecture & Feature Extraction

**Last Updated:** 2026-04-08  
**Entry Points:** `src/models/feature_extractor.py`, `model/semantic_decoder.py`

## Architecture Overview

```
YOLO Backbone (Frozen)
  ├─ Layers 0-15: Stem + Early blocks
  ├─ Layer 16 [P3]: 64ch  × 80×60    ──┐
  ├─ Layer 19 [P4]: 128ch × 40×30    ──┤ Feature Extraction
  ├─ Layer 22 [P5]: 256ch × 20×15    ──┘
  └─ Layers 23+: Detection head (not used in this pipeline)

Feature Extraction (Hook-based)
  └─ FeatureExtractor registers forward hooks on layers [16, 19, 22]
       └─ Captures P3, P4, P5 features → {16: tensor, 19: tensor, 22: tensor}

Semantic Decoder (NEW)
  └─ SementicDecoder(num_classes, 64, 128, 256)
       ├─ P3 pathway:  Conv(3×3)
       ├─ P4 pathway:  UpsampleBlock(128→64)
       ├─ P5 pathway:  UpsampleBlock(256→128) → UpsampleBlock(128→64)
       ├─ Fusion:      Element-wise sum
       └─ Head:        Conv(1×1) → Upsample(8×) → (num_classes, 640, 480)
```

## src/models/ Modules

### feature_extractor.py (35 lines)

```python
class FeatureExtractor(nn.Module):
    def __init__(self, model, layers: list[int]):
        """Register forward hooks on specified layers.
        
        Args:
            model: YOLO instance (ultralytics)
            layers: list of layer indices to hook (e.g., [16, 19, 22])
        """
        
    def _hook_fn(self, name) -> Callable:
        """Returns hook function that saves output to self.features[name]"""
        
    def forward(self, X: Tensor) -> dict[int, Tensor]:
        """Execute model and return {layer_id: output_tensor}"""
        return self.features  # {16: P3, 19: P4, 22: P5}
        
    # Context manager support
    def __enter__(self) / __exit__(): 
        """Auto-cleanup of hooks on context exit"""
```

**Usage:**

```python
from ultralytics import YOLO
from src.models.feature_extractor import FeatureExtractor

yolo = YOLO('yolo26n-seg.pt')

with FeatureExtractor(yolo, layers=[16, 19, 22]) as extractor:
    features = extractor(image)  # {16: P3, 19: P4, 22: P5}
    # P3: (1, 64, 80, 60)
    # P4: (1, 128, 40, 30)
    # P5: (1, 256, 20, 15)
```

## model/ Modules

### semantic_decoder.py (71 lines)

**Classes:**

1. **UpsampleBlock** — Conv → GroupNorm → ReLU → Bilinear Upsample(2×)
   ```python
   def forward(self, x: Tensor) -> Tensor:
       x = self.conv(x)           # 3×3 conv
       x = self.norm(x)           # GroupNorm(out_channels // 16)
       x = self.relu(x)
       x = self.upsample(x)       # 2× bilinear, align_corners=False
       return x
   ```

2. **SementicDecoder** — FPN fusion → semantic segmentation head
   ```python
   def __init__(self, 
                num_classes: int,      # output channels
                p3_channel: int = 64,  # YOLO layer 16
                p4_channel: int = 128, # YOLO layer 19
                p5_channel: int = 256):# YOLO layer 22
       
       # P3 pathway: pass-through conv
       self.p3_up = nn.Sequential(Conv2d(64, 64, 3×3))
       
       # P4 pathway: 1× upsample block (40×30 → 80×60)
       self.p4_up = nn.Sequential(UpsampleBlock(128→64))
       
       # P5 pathway: 2× upsample blocks (20×15 → 40×30 → 80×60)
       self.p5_up = nn.Sequential(
           UpsampleBlock(256→128),
           UpsampleBlock(128→64)
       )
       
       # Output head: fuse → 1×1 conv → 8× upsample
       self.out = nn.Sequential(
           Conv2d(64, num_classes, 1×1),
           Upsample(8×, bilinear)
       )
   
   def forward(self, p3, p4, p5):
       p3 = self.p3_up(p3)       # (1, 64, 80, 60)
       p4 = self.p4_up(p4)       # (1, 64, 80, 60)
       p5 = self.p5_up(p5)       # (1, 64, 80, 60)
       
       fused = p3 + p4 + p5      # Element-wise sum
       out = self.out(fused)     # (1, num_classes, 640, 480)
       
       return out
   ```

**Input/Output:**
- Inputs: P3 (80×60), P4 (40×30), P5 (20×15) from YOLO
- Output: Semantic logits (num_classes, 640, 480)

## Integration Flow

```
run_stream.py / train()
  │
  ├─ Load YOLO checkpoint: model/yolo26n-seg.pt
  │
  ├─ with FeatureExtractor(yolo, layers=[16, 19, 22]) as extractor:
  │    └─ Hook layers 16 (P3), 19 (P4), 22 (P5)
  │
  ├─ Load WebDataset shards from Google Drive (rclone mount)
  │
  ├─ For each image in DataLoader:
  │    │
  │    ├─ features = extractor(image)  # Extract P3, P4, P5
  │    │
  │    ├─ decoder = SementicDecoder(num_classes=30, p3_ch=64, p4_ch=128, p5_ch=256)
  │    │
  │    └─ sem_logits = decoder(features[16], features[19], features[22])
  │         └─ Shape: (batch, 30, 640, 480)
  │
  ├─ Compute semantic loss vs ground truth
  │
  └─ (Panoptic fusion layer: combine instance + semantic — not yet implemented)
```

## Key Design Decisions

- **Frozen backbone:** YOLO weights not updated during training
- **Hook-based extraction:** Non-invasive, works with ultralytics YOLO out-of-the-box
- **FPN fusion:** Sum pooling (not concatenation) keeps parameter count low
- **8× output upsampling:** Aligns with 640×480 crop patch size (from image_crop)
- **GroupNorm in decoder:** More stable than BatchNorm for training on large feature maps

## Dependencies

| Module | Purpose |
|--------|---------|
| `torch.nn` | PyTorch layers (Conv2d, Upsample, GroupNorm) |
| `ultralytics` | YOLO model loading + inference |
| `src.models.feature_extractor` | Hook registration |
| `model.semantic_decoder` | FPN decoder |

## Related Codemaps

- [architecture.md](./architecture.md) — Overall system design
- [pipeline.md](./pipeline.md) — Data pipeline + model integration
- [data.md](./data.md) — Preprocessing (image_crop → 640×480 patches)

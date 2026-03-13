from dataclasses import dataclass
import torch

@dataclass
class PanopticQuality:
    PQ: float
    SQ: float
    RQ: float
    TP: int
    FP: int
    FN: int
    
@dataclass
class PanopticQualityResult:
    mPQ: float
    mSQ: float
    mRQ: float
    classes: dict[int, PanopticQuality]
    

def calculate_pq(g: torch.Tensor, p: torch.Tensor) -> PanopticQualityResult:
    return PanopticQualityResult(mPQ=0.0, mSQ=0.0, mRQ=0.0, classes={})
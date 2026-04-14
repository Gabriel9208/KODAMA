# KODAMA

**K**eypoint-free **O**bject-aware **D**epth-**A**ware **M**apping **A**rchitecture

A lightweight, real-time panoptic segmentation system for egocentric scene understanding, targeting pedestrian navigation assistance for the visually impaired.

---

## Overview

KODAMA builds on YOLO26n-seg as a frozen/lightly fine-tuned backbone and adds three original modules on top: a **Multi-Scale Semantic Decoder** (P3/P4/P5 FPN fusion), a **Panoptic Fusion Module** (thing/stuff conflict resolution), and a **Depth-aware Risk Scoring** system (2 m obstacle recall > 95%).

The core design principle is *principled lightweight*: instead of rewriting YOLO's detection head or calling a pretrained decoder as a black box, KODAMA borrows YOLO's mature instance segmentation capability while contributing an independently trained semantic branch — keeping the total parameter count to **~3.7 M** (vs. kMaX-DeepLab ResNet-50 at ~55 M).

**Dataset:** [SANPO](https://github.com/google-research-datasets/sanpo) (WACV 2025) — egocentric pedestrian navigation, 31 classes (15 things + 15 stuff + 1 void), trained on SANPO-Real only.

**Baseline comparison:** kMaX-DeepLab (ResNet-50, PQ 34.6 on SANPO-Real, ~55 M params).

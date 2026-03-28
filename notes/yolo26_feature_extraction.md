# YOLO26-seg feature extraction

## Model Architecture

| Layer    | Type      |
| -------- | --------- |
| Layer 0  | Conv      |
| Layer 1  | Conv      |
| Layer 2  | C3k2      |
| Layer 3  | Conv      |
| Layer 4  | C3k2      |
| Layer 5  | Conv      |
| Layer 6  | C3k2      |
| Layer 7  | Conv      |
| Layer 8  | C3k2      |
| Layer 9  | SPPF      |
| Layer 10 | C2PSA     |
| Layer 11 | Upsample  |
| Layer 12 | Concat    |
| Layer 13 | C3k2      |
| Layer 14 | Upsample  |
| Layer 15 | Concat    |
| Layer 16 | C3k2      |
| Layer 17 | Conv      |
| Layer 18 | Concat    |
| Layer 19 | C3k2      |
| Layer 20 | Conv      |
| Layer 21 | Concat    |
| Layer 22 | C3k2      |
| Layer 23 | Segment26 |

## Feature Extraction

| Layer         | Shape                        | Channel |
| ------------- | ---------------------------- | ------- |
| P3 (Layer 16) | torch.Size([1, 64, 80, 60])  | 64      |
| P4 (Layer 19) | torch.Size([1, 128, 40, 30]) | 128     |
| P5 (Layer 22) | torch.Size([1, 256, 20, 15]) | 256     |

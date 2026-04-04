# Literature Review

## YOLO v26

### Problem Solved

### Relation to KODAMA

### Design Decision in KODAMA

---

## Panoptic Segmentation

### Problem Solved

- Unifies sementic and instance segmentation tasks into a single framework, namely panoptic segmentation.

- Define the PQ metric as $PQ = SQ \times RQ$, where $SQ$ is the segmentation quality and $RQ$ is the recognition quality. Details as follows:
  - $$SQ = \frac{\sum_{(p, g) \in TP} \text{IoU}(p, g)}{|TP|}$$

  - $$RQ = \frac{|TP|}{|TP| + \frac{1}{2}|FP| + \frac{1}{2}|FN|}$$

  - Each pixel is assigned to a sementic label and an instance ID.

- The PQ of Things and stuff is calculated together.

- Unlabeled pixels are not considered in the evaluation.

### Relation to KODAMA

KODAMA aims to utilize panoptic segmentation to achieve semantic understanding of the scene for egocentric navigation. This paper defines the PQ metric to evaluate the performance of panoptic segmentation.

### Design Decision in KODAMA

The PQ metric can be directly used in KODAMA to evaluate the performance of panoptic segmentation results.

---

## YOLO-based Panoptic Segmentation Network 2021

### Problem Solved

### Relation to KODAMA

### Design Decision in KODAMA

---

## SANPO

### Problem Solved

Propose a new dataset for panoptic segmentation, namely SANPO.

- SANPO benchmark utilizes Kmax-Deeplab as model and is trained on SANPO-Train and evaluated on SANPO-RealTest(evaluate only on the human annotated subset).

- Kmax-Deeplab with resnet50 as encoder have PQ 34.6,
  - trained on pure SANPO-Real-Train(Human annotated GT only)
  - trained on SANPO-Synthetic-Train fine-tuned on SANPO-Real-Train(Human annotated GT only)
  - trained on SANPO-Synthetic-Train and SANPO-Real-Train(50% synthetic, 50% real).

### Relation to KODAMA

- KODAMA can use SANPO as a benchmark to evaluate the performance of panoptic
  segmentation.
- kMaX-DeepLab with ResNet-50 (~55M params) (PQ 34.6) is the smallest baseline available. KODAMA (~3.7M params) aims to achieve comparable PQ with ~1/15 the parameters and significantly faster inference.

### KODAMA Goal

Use Kmax-Deeplab with resnet50 as encoder as baseline model to compare.

1. **Utilize YOLOv26's mature capability, only train semantic segmentation branch**
   - kMaX-DeepLab trains the whole system (encoder + decoder) from scratch, which requires 32 TPU and a large amount of data. In contrast, YOLOv26 has already learned detection + instance segmentation capabilities on COCO, so we only need to train a lightweight semantic decoder. This significantly reduces training costs.
2. **Real-time inference and deployability**
   - YOLO26n is designed for edge deployment (NMS-free, nano-scale parameters), and with a decoder of less than 1M parameters, the entire system can run on a mobile phone or Jetson. kMaX-DeepLab cannot achieve this.
3. **Modular design**
   - Instance segmentation and semantic segmentation are separated, so one can be trained independently of the other. kMaX-DeepLab is end-to-end, so it is difficult to debug when something goes wrong.

---

## Panoptic Feature Pyramid Networks

### Problem Solved

Propose a simple, single-network baseline for the joint task of panoptic segmentation.

- Using FPN as backbone, apply Mask R-CNN onto it. (Mask R-CNN with FPN)

#### Model Design

- **FPN**: ResNet as a bottom-up pathway, and adds a light top-down pathway. Adding
  in transformed versions of higher-resolution features from the bottom-up pathway (skip connections). Generate high-resolution, rich, multi-scale features that are critical for sementic segmentation task.

- **Instance segmentation branch**: The use of the **same channel dimension** for all pyramid levels in FPN makes it easy to attach a region-based object detector (Mask R-CNN in this case) to it.

- **Semantic segmentation branch**:
  - Starting from the deepest FPN level (at 1/32 scale), we perform three upsampling stages to yield a feature map at 1/4 scale, where each upsampling stage consists of 3×3 convolution, group norm, ReLU, and 2× bilinear upsampling. This strategy is repeated for FPN scales 1/16, 1/8, and 1/4 (with progressively fewer upsampling stages). The result is a set of feature maps at the same 1/4 scale, which are then element-wise summed.
  - A final 1×1 convolution, 4× bilinear upsampling, and softmax are used to generate the per-pixel class labels at the original image resolution.
  - In addition to stuff classes, this branch also outputs a special ‘other’ class for all pixels belonging to objects (to avoid predicting stuff classes for such pixels).

#### Training Issues

- **Deal with conflict between instance and semantic segmentation**:
  1. Resolving overlaps between different instances based on their confidence scores,
  2. Resolving overlaps between instance and semantic segmentation outputs in favor of instances,
  3. Removing any stuff regions labeled ‘other’ or under a given area threshold.

- **Train Loss Design**:

  Total loss: $$L = \lambda_i (L_c + L_b + L_m) + \lambda_s L_s$$, cannot be purely add without weighting them. The optimal λs and λi is selected from {0.5, 0.75, 1.0}.
  - Instance segmentation branch
    - $$L_c$$ (classification loss) (noramlized by the number of sampled RoIs)
    - $$L_b$$ (bounding-box loss) (noramlized by the number of sampled RoIs)
    - $$L_m$$ (mask loss) (noramlized by the number of foreground RoIs)
  - Semantic segmentation branch
    - $$L_s$$ (semantic segmentation loss) (per-pixel cross-entropy loss between the predicted and the ground-truth labels, normalized by the number of labeled image pixels)

### Relation to KODAMA

KODAMA use YOLOv26 (which also have FPN structure) as instance segmentation model, so I can then add a semantic segmentation branch to it, similar to this paper.

### Design Decision in KODAMA

- Add semantic segmentation branch to YOLOv26, similar to this paper. While the paper utilize P2 ~ P5 for sementic segmentation branch, KODAMA only utilize **P3 ~ P5**. (P2 in PANet neck is not in the path in top-down pathway)

- Maybe use P2 as skip connection for semantic segmentation branch (ablation experement).

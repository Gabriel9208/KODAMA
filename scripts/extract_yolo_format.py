import cv2
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple
import yaml
import tarfile 
import os
from collections import Counter

pixel_counts = Counter()    
instance_counts = Counter()

with open("configs/sanpo_yolo_label_mapping.yaml") as f:
    sanpo_to_yolo = yaml.safe_load(f)["sanpo_to_yolo"]

lookup_table = np.full(31, -1, dtype=np.int32)

for k, v in sanpo_to_yolo.items():
    lookup_table[k] = v

################################################
#               Pixel to Polygon               #
################################################
def instance_isolation(mask: np.ndarray) -> Dict[Tuple[int, int], np.ndarray]:
    """
        Input: SANPO mask (3, H, W)
        Output: instance_masks with 0's and 1's (dict) (class_id, instance_id) -> (H, W)
    """
    if mask is None:
        print("Error: mask is None")
        return

    class_id = mask[0, :, :]
    instance_id = mask[1, :, :] * 256 + mask[2, :, :]

    c_flat = class_id.ravel()
    i_flat = instance_id.ravel()

    thing_mask = (class_id >= 0)

    pairs = np.column_stack((c_flat[thing_mask.ravel()], i_flat[thing_mask.ravel()]))
    unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)

    instance_masks = {}
    for (c_id, i_id), count in zip(unique_pairs, counts):
        pixel_counts[c_id] += count
        instance_counts[c_id] += 1

        condition = (class_id == c_id) & (instance_id == i_id)
        condition = condition.astype(np.uint8)
        instance_masks[(c_id, i_id)] = condition

    return instance_masks

def contour_extraction(instance_masks: Dict[Tuple[int, int], np.ndarray]) -> Dict[int, list]:
    """
        Input: instance_masks with 0's and 1's (dict) (class_id, instance_id) -> (H, W)
        Output: contours with class id (dict) class_id -> list of contours
    """

    H, W = next(iter(instance_masks.values())).shape

    contours = defaultdict(list)
    for (c_id, i_id), mask in instance_masks.items():
        cnts, _ = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # YOLO seg need 3 pts
            if len(approx) >= 3:
                approx = approx.squeeze(1) # (N, 2)
                approx_norm = approx.astype(np.float32) / np.array([W, H])
                contours[c_id].append(approx_norm)

    return contours
    

def pixel_to_polygon(mask: np.ndarray, output_path: str):
    instance_masks = instance_isolation(mask)

    if not instance_masks:
        print(f"No instance masks found for {output_path}")
        open(output_path, 'w').close()
        return

    contours = contour_extraction(instance_masks)

    with open(output_path, "w") as f:
        for c_id, cnts in contours.items():
            for cnt in cnts:
                f.write(f"{c_id} ")
                for x, y in cnt:
                    f.write(f"{x} {y} ")
                f.write("\n")            

################################################
#             SANPO to YOLO label              #
################################################

def sanpo_to_yolo_label(mask: np.ndarray) -> np.ndarray:    
    mask[0, :, :] = lookup_table[mask[0, :, :]]

    return mask

################################################
#              extract_yolo_format             #
################################################

def extract_yolo_format(mask: np.ndarray, output_path: str):
    mask = sanpo_to_yolo_label(mask)
    pixel_to_polygon(mask, output_path)
      

if __name__ == "__main__":
    shard_dir = "data/shards"
    
    if not os.path.exists("data/images/train"):
        os.makedirs("data/images/train", exist_ok=True)
    if not os.path.exists("data/labels/train"):
        os.makedirs("data/labels/train", exist_ok=True)

    for i in os.listdir(shard_dir):

        with tarfile.open(os.path.join(shard_dir, i), "r") as tar:
            names = tar.getnames()
            for name in names:
                if name.endswith(".seg.png"):
                    f = tar.extractfile(name)
                    buf = np.frombuffer(f.read(), dtype=np.uint8)
                    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    mask = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.int32)
                    extract_yolo_format(mask, f"data/labels/train/{name.replace('.seg.png', '.txt')}")
                
                elif name.endswith(".png"):
                    raw = tar.extractfile(name).read()
                    with open(f"data/images/train/{name}", "wb") as out:
                        out.write(raw)

    yolo_name = yaml.safe_load(open("configs/sanpo_yolo_label_mapping.yaml"))
    yolo_name = yolo_name["sanpo_panoptic_to_yolo_id"]
    
    print(f"{'ID':<4} {'Class':<20} {'Pixels':>12} {'Instances':>10}")
    print("-" * 50)
    for c_id in yolo_name.keys():
        class_name = yolo_name[c_id]
        count = pixel_counts[c_id]
        instance_count = instance_counts[c_id]
        print(f"{c_id:<4} {class_name:<20} {count:>12} {instance_count:>10}")
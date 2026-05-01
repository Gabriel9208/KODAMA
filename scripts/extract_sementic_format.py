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

with open("configs/sanpo_sementic_label_mapping.yaml") as f:
    sanpo_to_decoder = yaml.safe_load(f)["sanpo_to_decoder"]

lookup_table = np.full(31, -1, dtype=np.int32)

for k, v in sanpo_to_decoder.items():
    lookup_table[k] = v

def count_pixel(mask: np.ndarray):
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

    for (c_id, i_id), count in zip(unique_pairs, counts):
        pixel_counts[c_id] += count
        instance_counts[c_id] += 1


if __name__ == "__main__":
    shard_dir = "data/train_shards"
    
    if not os.path.exists("data/labels/train_semantic"):
        os.makedirs("data/labels/train_semantic", exist_ok=True)

    for i in os.listdir(shard_dir):

        with tarfile.open(os.path.join(shard_dir, i), "r") as tar:
            names = tar.getnames()
            for name in names:
                if name.endswith(".seg.png"):
                    raw = tar.extractfile(name).read()

                    f = tar.extractfile(name)
                    buf = np.frombuffer(f.read(), dtype=np.uint8)
                    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    mask = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.int32)
                    count_pixel(mask)

                    with open(f"data/labels/train_semantic/{name}", "wb") as out:
                        out.write(raw)

    decoder_to_name = yaml.safe_load(open("configs/sanpo_sementic_label_mapping.yaml"))
    decoder_to_name = decoder_to_name["sanpo_sementic_to_name"]
    
    print(f"{'ID':<4} {'Class':<20} {'Pixels':>12} {'Instances':>10}")
    print("-" * 50)
    for c_id in decoder_to_name.keys():
        class_name = decoder_to_name[c_id]
        count = pixel_counts[c_id]
        instance_count = instance_counts[c_id]
        print(f"{c_id:<4} {class_name:<20} {count:>12} {instance_count:>10}")
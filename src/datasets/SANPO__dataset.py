from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import torch

class SANPO__dataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.rgb_dir = os.path.join(data_dir, "video_frames")
        self.seg_dir = os.path.join(data_dir, "segmentation_masks")
        self.depth_dir = os.path.join(data_dir, "depth_maps")
        self.rgb_files = sorted([f for f in os.listdir(self.rgb_dir) if f.endswith(".png")])
        self.seg_files = sorted([f for f in os.listdir(self.seg_dir) if f.endswith(".png")])
        self.depth_files = sorted([f for f in os.listdir(self.depth_dir) if f.endswith(".npy")])

        if not (len(self.rgb_files) == len(self.seg_files) == len(self.depth_files)):
            raise ValueError("video_frames, segmentation_masks, and depth_maps should have the same length")
    
    def __len__(self):
        return len(self.rgb_files)
    
    def __getitem__(self, idx):
        rgb = cv2.imread(os.path.join(self.rgb_dir, self.rgb_files[idx]))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32)

        seg = cv2.imread(os.path.join(self.seg_dir, self.seg_files[idx]))
        seg = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY).astype(np.int64)

        depth = np.load(os.path.join(self.depth_dir, self.depth_files[idx])).astype(np.float32)

        return torch.from_numpy(rgb), torch.from_numpy(seg), torch.from_numpy(depth)
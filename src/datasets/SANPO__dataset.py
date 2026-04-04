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
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.int32)

        seg_bgr = cv2.imread(os.path.join(self.seg_dir, self.seg_files[idx]))
        seg_bgr = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.int32)
        semantic = seg_bgr[0, :, :]
        instance = seg_bgr[1, :, :] * 256 + seg_bgr[2, :, :]
        seg = np.stack([semantic, instance], axis=0).astype(np.int32)

        depth = np.load(os.path.join(self.depth_dir, self.depth_files[idx])).astype(np.float32)
 
        return torch.from_numpy(rgb), torch.from_numpy(seg), torch.from_numpy(depth)


if __name__ == "__main__":
    seg_dir=r'C:\Users\yen08\Desktop\KODAMA\data\processed\-5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG\camera_chest\left\segmentation_masks'
    files=os.listdir(seg_dir)
    seg=cv2.imread(os.path.join(seg_dir, files[0]))
    print('Channel B (0) unique:', np.unique(seg[:,:,0]))
    print('Channel G (1) unique:', np.unique(seg[:,:,1]))
    print('Channel R (2) unique:', np.unique(seg[:,:,2]))
    print('Are all channels strictly identical?', np.array_equal(seg[:,:,0], seg[:,:,1]) and np.array_equal(seg[:,:,1], seg[:,:,2]))
import gzip
import numpy as np
import cv2
import os
import math

class SANPO_data_processor:
    @staticmethod
    def image_crop(image_path: str, save_dir: str, file_prefix: str, interpolate: int = cv2.INTER_LINEAR):
        """
        Read SANPO RGB image, crop into three overlapping patches and resize.
        """
        H, W = 1242, 2208
        TARGET_SIZE = (640, 640)

        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image from path: {image_path}")
        except Exception as e:
            raise IOError(f"Fail to read {image_path}: {e}")

        height, width, _ = image.shape
        
        if not math.isclose(width/height, W/H):
            raise ValueError(f"Image resolution aspect ratio does not match expected ratio. Expected {W/H}, got {width/height}")


        patches = {
            "left": image[:, 0:height],
            "center": image[:, width//2-height//2:width//2+height//2],
            "right": image[:, width-height:width]
        }

        resized_patches = {}
        for position, patch in patches.items():
            resized_patches[position] = cv2.resize(patch, TARGET_SIZE, interpolation=interpolate)
            
            save_path = os.path.join(save_dir, f"{file_prefix}_{position}.png")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            print(f"Saving cropped image to: {save_path}")
            
            cv2.imwrite(save_path, resized_patches[position])


    @staticmethod
    def process_and_patch_sanpo_depth(gz_file_path: str, save_dir: str, file_prefix: str):
        """
        Read SANPO float16.gz deepth map.
        """
        DEPTH_H, DEPTH_W = 720, 1280 
        TARGET_SIZE = (640, 640)

        try:
            with gzip.open(gz_file_path, 'rb') as f:
                raw_bytes = f.read()

            # First 4 bytes are header, the rest are pixel data
            pixel_bytes = raw_bytes[4:]

            depth_f16 = np.frombuffer(pixel_bytes, dtype=np.float16).reshape(DEPTH_H, DEPTH_W)

        except Exception as e:
            raise IOError(f"Fail to read {gz_file_path}: {e}")

        # cv2 cannot handle float16 well -> convert to 32
        depth_f32 = depth_f16.astype(np.float32)
        depth_f32 = np.nan_to_num(depth_f32, nan=0.0)

        height, width = depth_f32.shape

        patches = {
            "left": depth_f32[:, 0:height],
            "center": depth_f32[:, width//2-height//2:width//2+height//2],
            "right": depth_f32[:, width-height:width]
        }

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        resized_patches = {}
        for position, patch in patches.items():
            resized_patches[position] = cv2.resize(patch, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

            save_path = os.path.join(save_dir, f"{file_prefix}_{position}_float16.npy")
            np.save(save_path, resized_patches[position].astype(np.float16))

            
import gzip
import numpy as np
import cv2
import os

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
        assert height == H and width == W, \
            f"Fatal Error: Image resolution does not match expected size. Expected {(H, W)}, got {(height, width)}"

        patches = {
            "left": image[:, 0:1242],
            "center": image[:, 483:1725],
            "right": image[:, 966:W]
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

            expected_bytes = DEPTH_H * DEPTH_W * 2 + 4 
            assert len(raw_bytes) == expected_bytes, \
                f"Fatal Error: Incorrect Byte Count. Expected {expected_bytes}, got {len(raw_bytes)}"

            # First 4 bytes are header, the rest are pixel data
            pixel_bytes = raw_bytes[4:]

            depth_f16 = np.frombuffer(pixel_bytes, dtype=np.float16).reshape(DEPTH_H, DEPTH_W)

        except Exception as e:
            raise IOError(f"Fail to read {gz_file_path}: {e}")

        # cv2 cannot handle float16 well -> convert to 32
        depth_f32 = depth_f16.astype(np.float32)
        depth_f32 = np.nan_to_num(depth_f32, nan=0.0)


        patches = {
            "left": depth_f32[0:720, 0:720],
            "center": depth_f32[0:720, 280:1000],
            "right": depth_f32[0:720, 560:1280]
        }

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        resized_patches = {}
        for position, patch in patches.items():
            resized_patches[position] = cv2.resize(patch, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)

            save_path = os.path.join(save_dir, f"{file_prefix}_{position}_float16.npy")
            np.save(save_path, resized_patches[position].astype(np.float16))

            
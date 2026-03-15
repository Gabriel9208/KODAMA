import os
import cv2
from SANPO_data_processor import SANPO_data_processor

def process_directory(input_base_dir, output_base_dir):
    """Processes a single directory."""
    tasks = {
        "video_frames": {
            "function": SANPO_data_processor.image_crop,
            "params": {"interpolate": cv2.INTER_LINEAR},
            "extension": ".png"
        },
        "segmentation_masks": {
            "function": SANPO_data_processor.image_crop,
            "params": {"interpolate": cv2.INTER_NEAREST},
            "extension": ".png"
        },
        "depth_maps": {
            "function": SANPO_data_processor.process_and_patch_sanpo_depth,
            "params": {},
            "extension": ".float16.gz"
        }
    }

    for data_type, config in tasks.items():
        input_dir = os.path.join(input_base_dir, data_type)
        output_dir = os.path.join(output_base_dir, data_type)

        if not os.path.exists(input_dir):
            print(f"Input directory not found: {input_dir}")
            continue

        for filename in os.listdir(input_dir):
            if filename.endswith(config["extension"]):
                file_path = os.path.join(input_dir, filename)
                file_prefix = os.path.splitext(os.path.splitext(filename)[0])[0]

                print(f"Processing {file_path}...")
                try:
                    if data_type == 'depth_maps':
                        config["function"](gz_file_path=file_path, save_dir=output_dir, file_prefix=file_prefix)
                    else:
                        config["function"](image_path=file_path, save_dir=output_dir, file_prefix=file_prefix, **config["params"])
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

def preprocess_sanpo_data_generalized():
    data_root = r"C:\Users\yen08\Desktop\KODAMA\data"
    processed_root = os.path.join(data_root, "processed")

    for item in os.listdir(data_root):
        item_path = os.path.join(data_root, item)
        if os.path.isdir(item_path) and item != "processed":
            # Assuming the structure is always {item}/camera_chest/left
            input_base = os.path.join(item_path, "camera_chest", "left")
            output_base = os.path.join(processed_root, item, "camera_chest", "left")
            
            if os.path.exists(input_base):
                print(f"--- Processing directory: {input_base} ---")
                process_directory(input_base, output_base)
            else:
                print(f"--- Skipping {item_path}, 'camera_chest/left' not found ---")

if __name__ == "__main__":
    preprocess_sanpo_data_generalized()

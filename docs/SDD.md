# SDD for SANPO dataset off-line processing: Phase 1

## 1. Requirements

It's like(analogy only) moving the folders under **data/** into **data/processed/**. But do not move them. The folder structure may look like:
```
data/
    - -5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG/
        - camera_chest/
            - left/
                - depth_maps/
                - segmentation_masks/
                - video_frames/
                - frame_segmentation_annotation_type.json
            - ~~right/~~
                - ...
            - fixed_camera_poses.csv
            - description.json
        - ~~...~~
        
    - ...
```

The files or folder around with ~~ is the ones need t be ignored, and ... means all other files/folder under the directory.
~~...~~ meand all these files/folder are ignored.

The goal:
1. use SANPO_data_processor.image_crop to process images under **video_frames** folder, the interpolation must use **cv2.INTER_LINEAR**.
2. use SANPO_data_processor.image_crop to process images under **segmentation_masks** folder, the interpolation must use **cv2.INTER_NEAREST**.
3. use SANPO_data_processor.process_and_patch_sanpo_depth to process images under **depth_maps** folder, the interpolation must use **cv2.INTER_NEAREST**.
4. The above processed files must have the same prefix as the original one, so 000000.png will be 000000_left.png..., and 000000.float16.gz must be 000000_left_float16.npy.


## 2. Output
The output processed files must be placed inside a **processed/** folder, which must be directly under **data/** folder.
The folder structure may look like:
```
data/processed/
    - -5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG/
        - camera_chest/
            - left/
                - depth_maps/
                - segmentation_masks/
                - video_frames/
                - frame_segmentation_annotation_type.json
            - fixed_camera_poses.csv
            - description.json        
    - ...
```

## 3. Steps
1. Read SANPO_data_processor to understand each function
2. Apply them to the files that just mentioned under "C:\Users\yen08\Desktop\KODAMA\data\-5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG\camera_chest\left"

---

# SDD for SANPO dataset off-line processing: Phase 2

## 1. Requirements
1. Generalization: The dir name "-5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG" may vary. The code have to auto detect folders under **data/** folder(ignore the json file and the processed/ folder), the structure will all be like:
C:\Users\yen08\Desktop\KODAMA\data\\{placeholder}\...
2. All thing stays quite the same as phase1, just make the code more generalized.

## 2. Output
The output processed files must be placed inside a **processed/** folder, which must be directly under **data/** folder.
The folder structure may look like:
```
data/processed/
    - -5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG/
        - camera_chest/
            - left/
                - depth_maps/
                - segmentation_masks/
                - video_frames/
                - frame_segmentation_annotation_type.json
            - fixed_camera_poses.csv
            - description.json        
    - ... (The key of this phase --> automatically handle other folder with same structure as -5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG)
```

## 3. Steps
1. Read SANPO_data_processor, SANPO_data_preprocess_script to understand what have done
2. Apply them to the folders under **data/** folder (ignore the processed/ folder)
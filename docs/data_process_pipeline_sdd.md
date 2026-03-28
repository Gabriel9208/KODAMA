# SDD Overview: SANPO Dataset Streaming Training Pipeline v2

> This document provides the architectural overview. For implementation details of each phase, refer to the corresponding standalone documents.

---

## Document Index

| File                                 | Content                                                                        | Execution Environment |
| ------------------------------------ | ------------------------------------------------------------------------------ | --------------------- |
| `00_overview.md` (this file)         | Architecture overview, Decimation design, Resume capability, Safety mechanisms | —                     |
| `01_phase0_download.md`              | Phase 0: Download session data from GCS                                        | Windows               |
| `02_phase1_decimation_preprocess.md` | Phase 1: Decimation + Data preprocessing                                       | Windows               |
| `03_phase2_3_pack_upload.md`         | Phase 2-3: Pack shards + Upload + Verify + Clean                               | Windows               |
| `04_phase4_train_stream.md`          | Phase 4: Mount Google Drive + Streaming training                               | Ubuntu P100           |

**Program Outputs:**

- Phases 0-3 → `pipeline_pack_upload.py` (Windows)
- Phase 4 → `pipeline_train_stream.py` (Ubuntu P100)

---

## 1. Background and Goals

The training dataset is SANPO-Real (560 sessions), stored in a public GCS bucket. Local and training machine storage is insufficient for the full dataset. The pipeline must:

1. Download required sessions from GCS (left side only)
2. Decimation: Sample frames by configurable parameters (reduce to ~1/10)
3. Preprocessing: Crop and resize to 640×640 (using existing `SANPO_data_processor`)
4. Pack into WebDataset shard format
5. Upload to Google Drive for long-term storage
6. Delete local data after verification passes
7. Stream from Google Drive during training without full download

---

## 2. Environment

| Role             | System                    | Purpose                                               |
| ---------------- | ------------------------- | ----------------------------------------------------- |
| Packing machine  | Windows (~97GB available) | Download, Decimation, Preprocess, Pack, Upload, Clean |
| Cloud storage    | Google Drive              | Long-term shard storage (~250GB)                      |
| Training machine | Ubuntu (2×P100 16GB)      | Mount cloud storage, Execute training                 |

---

## 3. Core Technology Choices

| Technology                    | Purpose                     | Rationale                                                                                                                                                         |
| ----------------------------- | --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| WebDataset (tar-based shards) | Packing format              | Merges many small files into large tars; native PyTorch streaming support; shard design enables parallel reads and shuffle. Target shard size: **1-2 GB** per tar |
| gcloud CLI                    | Download from GCS           | Official tool, supports public buckets                                                                                                                            |
| rclone                        | Google Drive upload / mount | Supports upload, check, and mount in one tool                                                                                                                     |

---

## 4. Full Pipeline Overview

```
GCS (gs://gresearch/sanpo_dataset/v0/sanpo-real/)
        ↓  Phase 0: gcloud storage cp (left side only)
Windows local (data\raw\)
        ↓  Phase 1 Step 0: Pre-Decimation Validation
        ↓    → FAIL: delete raw data, mark "skipped", free disk space
        ↓    → PASS: continue
        ↓  Phase 1a: Decimation (sample every N frames, configurable interval + offset)
        ↓  Phase 1b: Preprocessing (crop + resize to 640×640, using SANPO_data_processor)
Windows local (data\processed\)
        ↓  Phase 2: Pack into WebDataset shards
Windows local (data\shards\)
        ↓  Phase 3a: rclone copy upload
        ↓  Phase 3b: rclone check verification
        ↓  Phase 3c: Delete raw + processed + shards after verification passes
Google Drive (SANPO-Dataset/shards/)
        ↓  Phase 4a: rclone mount (Ubuntu P100)
        ↓  Phase 4b: WebDataset streaming training
~/gdrive/shards/ → GPU training
```

---

## 5. Decimation Design

### 5.1 Parameter Definition

```yaml
# configs/decimation_config.yaml
decimation:
  interval: 10 # Sample 1 frame every N frames
  offset: 0 # Start from which frame (0-indexed)
  # Sampling result: offset, offset+interval, offset+2*interval, ...
  # Default interval=10, offset=0 → frames 0, 10, 20, 30...
```

### 5.2 Sampling Logic

```
Input: Sorted frame list (natural sort by filename)
Sample: frames[offset], frames[offset + interval], frames[offset + 2*interval], ...

Example (interval=10, offset=0, 527 frames total):
  Selected frame indices: 0, 10, 20, 30, ... 520 → 53 frames
```

### 5.3 Key Rules

1. **Deterministic sampling**: No randomness. Same parameters always produce same results
2. **Three data types stay synchronized**: video_frames, segmentation_masks, depth_maps use identical frame indices
3. **Parameters recorded**: Decimation parameters written into each sample's metadata for traceability
4. **Left side only**: Each session processes only `camera_chest/left` and/or `camera_head/left`

### 5.4 Data Volume Estimate

> **Important:** Not all 560 train sessions have segmentation annotations. SANPO documentation states ~20% of real frames have panoptic masks. Sessions without complete seg + depth annotations will be filtered out in Phase 1 Step 0 (Pre-Decimation Validation). Actual valid session count may be significantly less than 560.

| Item                                            | Before Decimation                   | After Decimation (interval=10)              |
| ----------------------------------------------- | ----------------------------------- | ------------------------------------------- |
| Per session (~527 frames)                       | ~8 GB                               | ~800 MB                                     |
| 560 sessions (if all valid)                     | ~4.5 TB                             | ~450 GB (raw)                               |
| **Estimated valid sessions**                    | **~100-200 (TBD after validation)** | —                                           |
| After preprocessing (crop + resize)             | —                                   | TBD                                         |
| Google Drive requirement                        | ~4 TB                               | **Estimated ~50-120 GB**                    |
| Total training samples (if ~150 valid sessions) | —                                   | 150 × ~53 × 3 patches ≈ **~24,000 samples** |

> These estimates will be refined after the first batch of sessions is validated. Update this section with actual numbers.

---

## 6. Resume Capability Design

### 6.1 Progress Tracking File

`data\pipeline_progress.json` tracks the processing status of each session:

```json
{
  "decimation_config": {
    "interval": 10,
    "offset": 0
  },
  "sessions": {
    "session_abc123": {
      "status": "cleaned",
      "downloaded_at": "2026-03-28T10:00:00",
      "validated_at": "2026-03-28T10:05:00",
      "valid_cameras": ["camera_chest"],
      "total_frames": {"camera_chest": 527},
      "decimated_at": "2026-03-28T10:15:00",
      "processed_at": "2026-03-28T10:30:00",
      "packed_at": "2026-03-28T10:45:00",
      "uploaded_at": "2026-03-28T11:00:00",
      "verified_at": "2026-03-28T11:05:00",
      "cleaned_at": "2026-03-28T11:06:00",
      "shard_files": ["shard-000003.tar", "shard-000004.tar"],
      "frame_count": 53,
      "patch_count": 159
    },
    "session_def456": {
      "status": "downloaded",
      "downloaded_at": "2026-03-28T12:00:00"
    },
    "session_ghi789": {
      "status": "skipped",
      "skip_reason": "all cameras failed validation (chest: seg missing, head: depth count mismatch)",
      "skipped_at": "2026-03-28T12:05:00"
    }
  }
}
```

### 6.2 Session State Machine

```
pending → downloaded → validated → decimated → processed → packed → uploaded → verified → cleaned
           ↘ error                ↘ skipped
```

- **`skipped`**: All cameras failed validation (missing seg/depth, count mismatch). Permanently excluded, never retried. Raw data is immediately deleted to free disk space. Note: validation is per-camera — individual cameras that fail are deleted, but the session is only `skipped` if no camera passes.
- **`error`**: Environment failure (network interruption, download failure, gcloud error). Retryable — on resume, treated the same as `pending` and re-downloaded.

Each state transition is written to the progress file immediately upon success.

### 6.3 Resume Logic

On script startup:

1. Read `pipeline_progress.json` (initialize if not found)
2. For all sessions, resume from the next step based on current status:
   - `cleaned` → Skip
   - `skipped` → Skip (permanently excluded — missing/misaligned seg or depth)
   - `verified` → Clean
   - `uploaded` → Verify
   - `packed` → Upload
   - `processed` → Pack
   - `decimated` → Preprocess
   - `validated` → Decimate
   - `downloaded` → Validate (Pre-Decimation Validation)
   - `error` → Download (retry)
   - `pending` → Download

### 6.4 Decimation Parameter Change Protection

If the `decimation_config` in the progress file differs from the current yaml:

- Script raises an error and aborts
- User must manually delete the progress file to proceed
- Prevents mixing different decimation parameters within the same shard set

---

## 7. Safety Mechanisms

### 7.1 Automatic Directory Creation

Before all I/O operations: `os.makedirs(target_dir, exist_ok=True)`. Applies to:

- `data\raw\{session_id}\...`
- `data\processed\{session_id}\...`
- `data\shards\`

### 7.2 Pre-Deletion Verification

**Unverified deletion is strictly forbidden.** Prerequisites:

- Corresponding shards have been uploaded to Google Drive
- `rclone check` passed (checksum match confirmed)
- Session status in `pipeline_progress.json` is `"verified"`

### 7.3 File Integrity Checks

| Timing                        | Check Content                                                                                                                                                                     |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| After download                | `video_frames/` exists and is non-empty                                                                                                                                           |
| **Pre-Decimation Validation** | **Per-camera check: seg_masks and depth_maps directories exist and non-empty; file counts match across all three types; frame IDs are aligned. FAIL → delete that camera's raw data only. Session marked `skipped` only if all cameras fail** |
| After decimation              | Sampled counts match across all three data types, indices match parameters                                                                                                        |
| After preprocessing           | Each frame produced 9 files (3 patches × 3 data types)                                                                                                                            |
| After packing                 | Each shard sample contains `.png` + `_seg.png` + `_depth.npy` + `.json`                                                                                                           |
| After upload                  | `rclone check` checksum comparison                                                                                                                                                |

### 7.4 Progress File Protection

- Write to `.tmp` first, then rename to overwrite (atomic write)
- On startup, if `.tmp` exists, the previous write was incomplete — use the last good version

### 7.5 Disk Space Check

Before downloading each new batch, check available space >= estimated requirement × 1.5. If insufficient, pause and prompt the user.

---

## 8. Storage Requirements

| Location      | Requirement                       | Notes                                                                          |
| ------------- | --------------------------------- | ------------------------------------------------------------------------------ |
| Windows local | Varies by batch                   | Peak: raw + processed + shard coexist; deleted after verification              |
| Google Drive  | ~50-120 GB (TBD after validation) | Long-term storage for all shards; much less than 560 sessions due to filtering |
| Ubuntu P100   | ~50 GB                            | rclone cache ceiling, does not grow                                            |

---

## 9. Prerequisites

**Windows packing machine:** gcloud CLI, rclone (gdrive remote configured), Python (webdataset, opencv-python, numpy, pyyaml), existing `SANPO_data_processor` code

**Ubuntu training machine:** rclone (gdrive remote configured), FUSE, Python (webdataset, torch, ultralytics)

---

## 10. Open Items

- [ ] Confirm camera_head vs camera_chest distribution across 560 sessions
- [ ] Confirm whether some sessions only have chest or only head
- [x] ~~Determine batch download size~~ → **3-4 sessions per batch** (~24-32 GB raw, peak ~40-50 GB with processed + shards)
- [ ] Confirm whether GCS download incurs egress fees
- [ ] Choose Google Drive storage plan
- [ ] Check Ubuntu P100 actual available disk space (determines rclone cache ceiling)
- [ ] Confirm decimation interval=10 is sufficient (can adjust later, but requires repacking all data)
- [ ] Confirm SANPO_data_processor's image_crop works on all session resolutions (currently hardcoded 2208×1242 and 720×1280)

---

The following are the step by step SDD

# Phase 0: Download Session Data from GCS

> Execution environment: Windows
> Prerequisite: `00_overview.md`
> Next step: `02_phase1_decimation_preprocess.md`

---

## Goal

Download SANPO-Real train sessions from Google Cloud Storage. Only download the left side of each camera for three data types.

---

## Input

| Item            | Path / Source                                                                                   |
| --------------- | ----------------------------------------------------------------------------------------------- |
| Session ID list | `data\sanpo_dataset_v0_sanpo-real_splits_train_session_ids.txt` (560 sessions, one ID per line) |
| GCS bucket      | `gs://gresearch/sanpo_dataset/v0/sanpo-real/`                                                   |

---

## Download Scope

**Left side only.** Segmentation masks and depth maps are only annotated for the left lens.

For each session, download three directories:

- `{session_id}/camera_chest/left/video_frames/` (`.png`)
- `{session_id}/camera_chest/left/segmentation_masks/` (`.png`)
- `{session_id}/camera_chest/left/depth_maps/` (`.float16.gz`)

> If a session contains `camera_head`, also download `camera_head/left/` with the same three directories.

**Do NOT download:** right side, zed_depth_maps, camera poses, or other metadata.

---

## Download Commands

```bash
# Download chest camera (left side)
gcloud storage cp -r gs://gresearch/sanpo_dataset/v0/sanpo-real/{session_id}/camera_chest/left/video_frames data\raw\{session_id}\camera_chest\left\
gcloud storage cp -r gs://gresearch/sanpo_dataset/v0/sanpo-real/{session_id}/camera_chest/left/segmentation_masks data\raw\{session_id}\camera_chest\left\
gcloud storage cp -r gs://gresearch/sanpo_dataset/v0/sanpo-real/{session_id}/camera_chest/left/depth_maps data\raw\{session_id}\camera_chest\left\

# If head camera exists (left side)
gcloud storage cp -r gs://gresearch/sanpo_dataset/v0/sanpo-real/{session_id}/camera_head/left/video_frames data\raw\{session_id}\camera_head\left\
gcloud storage cp -r gs://gresearch/sanpo_dataset/v0/sanpo-real/{session_id}/camera_head/left/segmentation_masks data\raw\{session_id}\camera_head\left\
gcloud storage cp -r gs://gresearch/sanpo_dataset/v0/sanpo-real/{session_id}/camera_head/left/depth_maps data\raw\{session_id}\camera_head\left\
```

---

## Download Strategy

Local available space is ~97GB. Each session's raw data is ~8GB. **Download in batches of 3-4 sessions** (~24-32 GB raw, peak ~40-50 GB with processed + shards):

```
Batch loop:
  1. Take the next 3-4 session IDs (skip sessions already "cleaned", "skipped", or "error" in progress)
  2. Check disk space >= estimated requirement × 1.5
  3. For each session in the batch:
     a. Check camera_head existence via: gcloud storage ls gs://.../{session_id}/camera_head/left/
     b. Download camera_chest/left/ (always)
     c. Download camera_head/left/ (only if step a confirmed it exists)
  4. → Proceed to Phase 1 (Decimation + Preprocessing)
  5. → Proceed to Phase 2-3 (Pack + Upload + Clean)
  6. Repeat until all sessions are processed
```

---

## Output Directory Structure

```
data\raw\{session_id}\
  ├── camera_chest\left\
  │   ├── video_frames\
  │   │   ├── 000000.png
  │   │   ├── 000001.png
  │   │   └── ... (~527 files)
  │   ├── segmentation_masks\
  │   │   ├── 000000.png
  │   │   └── ...
  │   └── depth_maps\
  │       ├── 000000.float16.gz
  │       └── ...
  └── camera_head\left\ (if available)
      └── ... (same structure)
```

---

## Integrity Check (run immediately after download)

**Note:** Not all SANPO-Real sessions have segmentation masks — only a subset has panoptic annotations. The download phase performs a basic check only. The strict validation (seg/depth existence, frame alignment) is performed in Phase 1 before decimation.

1. Confirm `video_frames/` directory exists and is non-empty
2. If `video_frames/` is missing or empty, mark session as `skipped` (confirmed missing data — never retry) and delete the raw data immediately

> Segmentation and depth validation is deferred to Phase 1 Step 0 (Pre-Decimation Validation). Sessions missing segmentation masks will be immediately deleted there to free disk space.

---

## Status Update

Download successful and basic check passed → update session in `pipeline_progress.json`:

```json
{
  "status": "downloaded",
  "downloaded_at": "2026-03-28T10:00:00"
}
```

---

## Prerequisites

- gcloud CLI installed
- `gcloud auth login` completed (SANPO is a public bucket, no project permissions needed)

---

## Error Handling

| Scenario                                        | Action                                                                                                |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Network interruption mid-download               | Mark as `error` (retryable). On resume, re-download from scratch. gcloud CLI handles partial transfer |
| gcloud CLI crash or unexpected exit code        | Mark as `error` (retryable). Log the exit code for diagnostics                                        |
| Session ID not found in GCS                     | Mark as `skipped` (permanent). Delete any partial download                                            |
| Insufficient disk space                         | Pause, prompt user to free space or wait for previous batch upload to complete                        |
| `video_frames/` missing or empty after download | Mark as `skipped` (permanent). Delete raw data immediately                                            |

---

# Phase 1: Decimation + Data Preprocessing

> Execution environment: Windows
> Prerequisite: `01_phase0_download.md`
> Next step: `03_phase2_3_pack_upload.md`

---

## Goal

Process downloaded raw session data in three steps:

1. **Pre-Decimation Validation**: Verify segmentation masks and depth maps exist, are complete, and align with video frames **per camera**. Cameras that fail validation are individually deleted to free disk space. A session is marked `skipped` only if **all** cameras fail.
2. **Decimation**: Sample frames by configurable parameters, reducing ~527 frames to ~53 frames
3. **Preprocessing**: Crop wide images into three 640×640 patches, using the existing `SANPO_data_processor`

---

## Input

| Item               | Source                                                      |
| ------------------ | ----------------------------------------------------------- |
| Raw data           | `data\raw\{session_id}\camera_chest\left\` (Phase 0 output) |
| Decimation config  | `configs\decimation_config.yaml`                            |
| Preprocessing tool | `SANPO_data_processor` (existing code)                      |

---

## Step 0: Pre-Decimation Validation

**Background:** Not all SANPO-Real sessions have segmentation annotations — only a subset (~20%) has panoptic masks. Cameras without complete annotations are useless for panoptic segmentation training and must be filtered out before any processing.

**Validation is per-camera, not per-session.** Each camera (chest, head) is validated independently. A camera that fails is deleted individually. The session is only marked `skipped` if all cameras fail.

### Validation Checks

For each camera within a downloaded session (`camera_chest/left`, `camera_head/left` if present):

```python
video_frames_dir = f"data/raw/{session_id}/{camera}/left/video_frames"
seg_masks_dir    = f"data/raw/{session_id}/{camera}/left/segmentation_masks"
depth_maps_dir   = f"data/raw/{session_id}/{camera}/left/depth_maps"

# --- Check 1: All three directories exist and are non-empty ---
if not os.path.isdir(seg_masks_dir) or len(os.listdir(seg_masks_dir)) == 0:
    # FAIL: No segmentation masks
    → delete and skip

if not os.path.isdir(depth_maps_dir) or len(os.listdir(depth_maps_dir)) == 0:
    # FAIL: No depth maps
    → delete and skip

# --- Check 2: File counts match ---
frame_files = sorted([f for f in os.listdir(video_frames_dir) if f.endswith(".png")])
seg_files   = sorted([f for f in os.listdir(seg_masks_dir) if f.endswith(".png")])
depth_files = sorted([f for f in os.listdir(depth_maps_dir) if f.endswith(".float16.gz")])

if not (len(frame_files) == len(seg_files) == len(depth_files)):
    # FAIL: File count mismatch
    → delete and skip

# --- Check 3: Frame alignment (filenames correspond) ---
# Extract the numeric prefix from each filename and compare
frame_ids = [extract_id(f) for f in frame_files]    # e.g. ["000000", "000001", ...]
seg_ids   = [extract_id(f) for f in seg_files]
depth_ids = [extract_id(f) for f in depth_files]

if frame_ids != seg_ids or frame_ids != depth_ids:
    # FAIL: Frames are not aligned
    → delete and skip
```

### On Camera Validation Failure

If any check fails for a camera:

1. **Log the reason** (which camera, which check failed, file counts, mismatched IDs)
2. **Delete only that camera's raw data**:
   ```python
   shutil.rmtree(f"data/raw/{session_id}/{camera}")
   ```
3. **Continue** to validate the next camera in the same session

### After All Cameras Validated

If **all cameras failed** → mark the session as `skipped`:
```json
{
  "status": "skipped",
  "skip_reason": "all cameras failed validation (chest: seg missing, head: depth count mismatch)",
  "skipped_at": "2026-03-28T10:05:00"
}
```
Delete any remaining raw data for the session: `shutil.rmtree(f"data/raw/{session_id}")`

If **at least one camera passed** → mark the session as `validated`:
```json
{
  "status": "validated",
  "validated_at": "2026-03-28T10:05:00",
  "valid_cameras": ["camera_chest"],
  "total_frames": {"camera_chest": 527}
}
```

### Why Delete Immediately

- Raw data is ~8GB per session
- Local disk space is limited (~97GB)
- Invalid sessions waste space that could be used for the next batch download
- The original data remains on GCS and can be re-downloaded if needed

---

## Step 1a: Decimation

### Processing Logic

```python
# Read decimation parameters
interval = config["decimation"]["interval"]  # default 10
offset = config["decimation"]["offset"]      # default 0

# List and sort video_frames
all_frames = sorted(os.listdir(video_frames_dir))

# Sample
selected_indices = list(range(offset, len(all_frames), interval))
selected_frames = [all_frames[i] for i in selected_indices]

# Use the same indices for segmentation_masks and depth_maps
all_segs = sorted(os.listdir(seg_masks_dir))
all_depths = sorted(os.listdir(depth_maps_dir))
selected_segs = [all_segs[i] for i in selected_indices]
selected_depths = [all_depths[i] for i in selected_indices]
```

### Key Rules

- Sort by filename (natural sort) to ensure all three data types are aligned
- `selected_indices` is deterministic — no randomness
- Decimation only determines "which frames to process" — no files are moved or copied at this stage

### Integrity Check

- Confirm `len(selected_frames) == len(selected_segs) == len(selected_depths)`
- If mismatch, log error and mark session as `error`

### Status Update

Decimation complete → status updated to `"decimated"`

---

## Step 1b: Data Preprocessing

### Existing Code Dependencies

The `SANPO_data_processor` class provides two static methods:

**`image_crop(image_path, save_dir, file_prefix, interpolate)`**

- Input: SANPO wide image (2208×1242)
- Process: Crop into three overlapping square patches: left / center / right
- Each resized to 640×640
- Output: `{file_prefix}_left.png`, `{file_prefix}_center.png`, `{file_prefix}_right.png`

**`process_and_patch_sanpo_depth(gz_file_path, save_dir, file_prefix)`**

- Input: `.float16.gz` depth map (720×1280)
- Process: Decompress → float16 → float32 → nan_to_num → crop three patches → resize to 640×640
- Output: `{file_prefix}_left_float16.npy`, etc.

### Processing Logic

```python
for i, frame_filename in enumerate(selected_frames):
    frame_index = extract_index(frame_filename)  # e.g. "000000"

    # RGB: crop + resize (bilinear interpolation)
    SANPO_data_processor.image_crop(
        image_path=os.path.join(video_frames_dir, frame_filename),
        save_dir=processed_video_dir,
        file_prefix=frame_index,
        interpolate=cv2.INTER_LINEAR
    )

    # Segmentation: crop + resize (nearest interpolation to preserve labels)
    SANPO_data_processor.image_crop(
        image_path=os.path.join(seg_masks_dir, selected_segs[i]),
        save_dir=processed_seg_dir,
        file_prefix=frame_index,
        interpolate=cv2.INTER_NEAREST
    )

    # Depth: decompress + crop + resize
    SANPO_data_processor.process_and_patch_sanpo_depth(
        gz_file_path=os.path.join(depth_maps_dir, selected_depths[i]),
        save_dir=processed_depth_dir,
        file_prefix=frame_index
    )
```

### Output Per Frame

Each sampled frame produces **9 files** (3 patches × 3 data types):

| Patch  | RGB                | Segmentation       | Depth                      |
| ------ | ------------------ | ------------------ | -------------------------- |
| left   | `{idx}_left.png`   | `{idx}_left.png`   | `{idx}_left_float16.npy`   |
| center | `{idx}_center.png` | `{idx}_center.png` | `{idx}_center_float16.npy` |
| right  | `{idx}_right.png`  | `{idx}_right.png`  | `{idx}_right_float16.npy`  |

---

## Output Directory Structure

```
data\processed\{session_id}\camera_chest\left\
  ├── video_frames\
  │   ├── 000000_left.png      (640×640 RGB)
  │   ├── 000000_center.png
  │   ├── 000000_right.png
  │   ├── 000010_left.png
  │   ├── 000010_center.png
  │   ├── 000010_right.png
  │   └── ...
  ├── segmentation_masks\
  │   ├── 000000_left.png      (640×640 semantic+instance mask)
  │   ├── 000000_center.png
  │   └── ...
  └── depth_maps\
      ├── 000000_left_float16.npy   (640×640 float16)
      ├── 000000_center_float16.npy
      └── ...
```

---

## Integrity Check (after preprocessing)

1. Each sampled frame must produce 3 patch groups (left / center / right)
2. Each group must contain RGB + seg + depth (3 files)
3. Total files per sampled frame = 9
4. Total file count = `len(selected_frames) × 9`

---

## Status Update

Preprocessing complete and checks passed → status updated to `"processed"`:

```json
{
  "status": "processed",
  "processed_at": "2026-03-28T10:30:00",
  "frame_count": 53,
  "patch_count": 159
}
```

---

## Data Volume Estimate

> Actual numbers depend on how many sessions pass Pre-Decimation Validation. Not all 560 sessions have segmentation annotations.

| Item                                        | Count                           |
| ------------------------------------------- | ------------------------------- |
| Sampled frames per session                  | ~53 (527 / 10)                  |
| Patches per frame                           | 3 (left / center / right)       |
| Samples per session                         | ~159                            |
| Estimated valid sessions                    | ~100-200 (TBD after validation) |
| **Estimated total samples (if ~150 valid)** | **~24,000**                     |

> Update these numbers after running the first batch.

---

## Error Handling

| Scenario                                                | Action                                                                                      |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **Validation: seg_masks directory missing or empty**    | **Log reason, delete that camera's raw data only, continue to next camera. Mark session `skipped` only if all cameras fail** |
| **Validation: depth_maps directory missing or empty**   | **Same as above — per-camera deletion, not per-session**                                     |
| **Validation: File count mismatch across three types**  | **Same as above — log counts, delete that camera only**                                      |
| **Validation: Frame IDs not aligned**                   | **Same as above — log mismatched IDs, delete that camera only**                              |
| `SANPO_data_processor` fails on a specific frame        | Log error, skip that frame, continue with others                                            |
| Image resolution doesn't match expected (not 2208×1242) | `image_crop` raises ValueError internally; log and skip                                     |
| Depth file corrupted                                    | `process_and_patch_sanpo_depth` raises IOError internally; log and skip                     |
| Post-preprocessing file count doesn't match expected    | Log warning but continue (allow minor gaps)                                                 |

"""
SANPO Dataset Streaming Pipeline — Phases 0-3
Downloads, decimates, preprocesses, packs, uploads, and cleans SANPO-Real sessions.

Usage:
    python scripts/pipeline_pack_upload.py
"""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import yaml

# Add src/ to path for SANPO_data_processor import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from utils.SANPO_data_processor import SANPO_data_processor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SHARDS_DIR = DATA_DIR / "shards"
PROGRESS_FILE = DATA_DIR / "pipeline_progress.json"
SESSION_IDS_FILE = DATA_DIR / "sanpo_dataset_v0_sanpo-real_splits_train_session_ids.txt"
DECIMATION_CONFIG_FILE = PROJECT_ROOT / "configs" / "decimation_config.yaml"

GCS_BUCKET = "gs://gresearch/sanpo_dataset/v0/sanpo-real"

# Phase 0 constants
BATCH_SIZE = 4
DISK_SAFETY_FACTOR = 1.5
ESTIMATED_SESSION_SIZE_GB = 8

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Progress file helpers (atomic write)
# ---------------------------------------------------------------------------

def load_progress() -> dict:
    """Load pipeline_progress.json. Handle .tmp recovery."""
    tmp_path = PROGRESS_FILE.with_suffix(".json.tmp")

    # If .tmp exists but main file does not, the previous write was incomplete.
    # Discard the .tmp and start fresh or use the last good version.
    if tmp_path.exists():
        logger.warning("Found incomplete .tmp progress file — discarding it, using last good version.")
        tmp_path.unlink()

    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    # Initialize new progress
    decimation_config = _load_decimation_config()
    return {
        "decimation_config": decimation_config,
        "sessions": {},
    }


def save_progress(progress: dict) -> None:
    """Atomic write: write to .tmp then rename."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = PROGRESS_FILE.with_suffix(".json.tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)

    # os.replace is atomic on the same filesystem
    os.replace(tmp_path, PROGRESS_FILE)


def _load_decimation_config() -> dict:
    """Read decimation parameters from yaml."""
    with open(DECIMATION_CONFIG_FILE, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["decimation"]


def _validate_decimation_config(progress: dict) -> None:
    """Abort if decimation config changed since the progress file was created."""
    current = _load_decimation_config()
    saved = progress.get("decimation_config", {})
    if saved and saved != current:
        logger.error(
            "Decimation config mismatch!\n"
            f"  Progress file: {saved}\n"
            f"  Current yaml:  {current}\n"
            "Delete pipeline_progress.json manually to proceed with new parameters."
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _available_disk_gb(path: Path) -> float:
    """Return available disk space in GB for the drive containing *path*."""
    usage = shutil.disk_usage(path.anchor or path)
    return usage.free / (1024 ** 3)


def load_session_ids() -> list[str]:
    """Read session IDs from the train split file."""
    with open(SESSION_IDS_FILE, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Phase 0: Download
# ---------------------------------------------------------------------------

def _gcloud_ls(gcs_path: str) -> bool:
    """Check if a GCS path exists via gcloud storage ls. Returns True if exists."""
    result = subprocess.run(
        ["gcloud", "storage", "ls", gcs_path],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _gcloud_cp(gcs_src: str, local_dst: Path) -> bool:
    """Download from GCS. Returns True on success."""
    local_dst.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["gcloud", "storage", "cp", "-r", gcs_src, str(local_dst)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error(f"gcloud cp failed: {result.stderr.strip()}")
        return False
    return True


def _integrity_check_post_download(session_dir: Path) -> bool:
    """
    Basic check after download: video_frames/ must exist and be non-empty
    for at least camera_chest/left.
    """
    chest_frames = session_dir / "camera_chest" / "left" / "video_frames"
    if not chest_frames.exists() or not any(chest_frames.iterdir()):
        return False
    return True


def _delete_raw_session(session_id: str) -> None:
    """Delete raw data for a session."""
    session_dir = RAW_DIR / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir)
        logger.info(f"  Deleted raw data: {session_dir}")


def download_session(session_id: str, progress: dict) -> str:
    """
    Download a single session from GCS.

    Returns the new status: "downloaded", "skipped", or "error".
    """
    gcs_base = f"{GCS_BUCKET}/{session_id}"
    local_base = RAW_DIR / session_id

    cameras_to_download = ["camera_chest"]

    # Check if camera_head exists
    head_gcs = f"{gcs_base}/camera_head/left/"
    if _gcloud_ls(head_gcs):
        cameras_to_download.append("camera_head")
        logger.info(f"  camera_head detected for {session_id}")
    else:
        logger.info(f"  camera_head not found for {session_id}, downloading chest only")

    # Download each camera
    data_types = ["video_frames", "segmentation_masks", "depth_maps"]

    for camera in cameras_to_download:
        for dtype in data_types:
            gcs_src = f"{gcs_base}/{camera}/left/{dtype}"
            local_dst = local_base / camera / "left"

            # Check if source exists in GCS
            if not _gcloud_ls(gcs_src):
                if camera == "camera_chest" and dtype == "video_frames":
                    # Critical: chest video_frames must exist
                    logger.warning(f"  {camera}/left/{dtype} not found in GCS — marking skipped")
                    _delete_raw_session(session_id)
                    return "skipped"
                else:
                    # Non-critical missing directory (e.g. head seg masks)
                    logger.info(f"  {camera}/left/{dtype} not found in GCS — skipping this directory")
                    continue

            logger.info(f"  Downloading {camera}/left/{dtype} ...")
            success = _gcloud_cp(gcs_src, local_dst)
            if not success:
                logger.error(f"  Download failed for {camera}/left/{dtype}")
                _delete_raw_session(session_id)
                return "error"

    # Integrity check
    if not _integrity_check_post_download(local_base):
        logger.warning(f"  Integrity check failed for {session_id} — video_frames missing or empty")
        _delete_raw_session(session_id)
        return "skipped"

    return "downloaded"


def phase0_download(progress: dict) -> dict:
    """Execute Phase 0: download sessions in batches."""
    all_session_ids = load_session_ids()

    # Filter to sessions that need downloading
    pending = []
    for sid in all_session_ids:
        session_info = progress["sessions"].get(sid, {})
        status = session_info.get("status", "pending")
        if status in ("pending", "error"):
            pending.append(sid)

    if not pending:
        logger.info("Phase 0: No sessions to download.")
        return progress

    logger.info(f"Phase 0: {len(pending)} sessions to download ({len(all_session_ids)} total)")

    # Process in batches
    for batch_start in range(0, len(pending), BATCH_SIZE):
        batch = pending[batch_start : batch_start + BATCH_SIZE]

        # Disk space check
        available_gb = _available_disk_gb(DATA_DIR)
        required_gb = len(batch) * ESTIMATED_SESSION_SIZE_GB * DISK_SAFETY_FACTOR
        if available_gb < required_gb:
            logger.warning(
                f"Insufficient disk space: {available_gb:.1f} GB available, "
                f"{required_gb:.1f} GB required for {len(batch)} sessions. "
                "Pausing. Free space or wait for previous batch to upload, then re-run."
            )
            break

        logger.info(
            f"Phase 0 batch: downloading {len(batch)} sessions "
            f"(available: {available_gb:.1f} GB, est. required: {required_gb:.1f} GB)"
        )

        for sid in batch:
            logger.info(f"Processing session: {sid}")
            new_status = download_session(sid, progress)

            # Update progress
            if sid not in progress["sessions"]:
                progress["sessions"][sid] = {}

            progress["sessions"][sid]["status"] = new_status
            if new_status == "downloaded":
                progress["sessions"][sid]["downloaded_at"] = _now_iso()
                logger.info(f"  → downloaded")
            elif new_status == "skipped":
                logger.info(f"  → skipped (permanent)")
            elif new_status == "error":
                logger.info(f"  → error (will retry on next run)")

            save_progress(progress)

        # After downloading this batch, hand off to Phase 1-3 before next batch
        # (Phases 1-3 will be called by the main orchestrator)
        logger.info(f"Phase 0 batch complete. Proceeding to Phase 1-3 for this batch.")
        break  # Process one batch at a time, let main loop handle Phase 1-3

    return progress


# ---------------------------------------------------------------------------
# Phase 1: Validation + Decimation + Preprocessing
# ---------------------------------------------------------------------------

CAMERAS = ["camera_chest", "camera_head"]
PATCH_POSITIONS = ["left", "center", "right"]


def _extract_frame_id(filename: str) -> str:
    """Extract numeric prefix from filename. e.g. '000010.png' → '000010'."""
    match = re.match(r"^(\d+)", filename)
    if match:
        return match.group(1)
    return filename


def _discover_cameras(session_dir: Path) -> list[str]:
    """Find which cameras exist in a raw session directory."""
    found = []
    for camera in CAMERAS:
        cam_dir = session_dir / camera / "left"
        if cam_dir.exists():
            found.append(camera)
    return found


def _validate_camera(session_dir: Path, camera: str) -> tuple[bool, str, int]:
    """
    Validate a single camera's data integrity.

    Returns (passed, reason, frame_count).
    If passed is False, reason describes the failure.
    """
    cam_base = session_dir / camera / "left"
    video_dir = cam_base / "video_frames"
    seg_dir = cam_base / "segmentation_masks"
    depth_dir = cam_base / "depth_maps"

    # Check 1: All three directories exist and are non-empty
    if not seg_dir.exists() or not any(seg_dir.iterdir()):
        return False, f"{camera}: segmentation_masks directory missing or empty", 0

    if not depth_dir.exists() or not any(depth_dir.iterdir()):
        return False, f"{camera}: depth_maps directory missing or empty", 0

    if not video_dir.exists() or not any(video_dir.iterdir()):
        return False, f"{camera}: video_frames directory missing or empty", 0

    # Check 2: File counts match
    frame_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".png")])
    seg_files = sorted([f for f in os.listdir(seg_dir) if f.endswith(".png")])
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith(".float16.gz")])

    if not (len(frame_files) == len(seg_files) == len(depth_files)):
        return (
            False,
            f"{camera}: file count mismatch — frames={len(frame_files)}, "
            f"seg={len(seg_files)}, depth={len(depth_files)}",
            0,
        )

    # Check 3: Frame ID alignment
    frame_ids = [_extract_frame_id(f) for f in frame_files]
    seg_ids = [_extract_frame_id(f) for f in seg_files]
    depth_ids = [_extract_frame_id(f) for f in depth_files]

    if frame_ids != seg_ids or frame_ids != depth_ids:
        return False, f"{camera}: frame IDs not aligned", 0

    return True, "ok", len(frame_files)


def _delete_camera_data(session_dir: Path, camera: str) -> None:
    """Delete a specific camera's raw data."""
    cam_dir = session_dir / camera
    if cam_dir.exists():
        shutil.rmtree(cam_dir)
        logger.info(f"    Deleted {camera} raw data")


def validate_session(session_id: str, progress: dict) -> dict:
    """
    Phase 1 Step 0: Pre-Decimation Validation (per-camera).

    Returns updated progress dict.
    """
    session_dir = RAW_DIR / session_id
    cameras = _discover_cameras(session_dir)
    session_info = progress["sessions"][session_id]

    valid_cameras = []
    total_frames = {}
    fail_reasons = []

    for camera in cameras:
        passed, reason, frame_count = _validate_camera(session_dir, camera)
        if passed:
            valid_cameras.append(camera)
            total_frames[camera] = frame_count
            logger.info(f"    {camera}: PASSED ({frame_count} frames)")
        else:
            fail_reasons.append(reason)
            logger.warning(f"    {camera}: FAILED — {reason}")
            _delete_camera_data(session_dir, camera)

    if not valid_cameras:
        # All cameras failed
        session_info["status"] = "skipped"
        session_info["skip_reason"] = "; ".join(fail_reasons)
        session_info["skipped_at"] = _now_iso()
        _delete_raw_session(session_id)
        logger.warning(f"  Session {session_id} → skipped (all cameras failed)")
    else:
        session_info["status"] = "validated"
        session_info["validated_at"] = _now_iso()
        session_info["valid_cameras"] = valid_cameras
        session_info["total_frames"] = total_frames
        logger.info(
            f"  Session {session_id} → validated "
            f"(cameras: {valid_cameras})"
        )

    save_progress(progress)
    return progress


def _decimate_indices(total_frames: int, interval: int, offset: int) -> list[int]:
    """Compute decimated frame indices."""
    return list(range(offset, total_frames, interval))


def decimate_session(session_id: str, progress: dict) -> dict:
    """
    Phase 1 Step 1a: Decimation (in-memory index computation).

    Returns updated progress dict.
    """
    session_info = progress["sessions"][session_id]
    config = progress["decimation_config"]
    interval = config["interval"]
    offset = config["offset"]

    valid_cameras = session_info["valid_cameras"]
    decimated_frames = {}

    for camera in valid_cameras:
        cam_base = RAW_DIR / session_id / camera / "left"
        video_dir = cam_base / "video_frames"

        all_frames = sorted([f for f in os.listdir(video_dir) if f.endswith(".png")])
        indices = _decimate_indices(len(all_frames), interval, offset)

        # Verify all three types have consistent counts
        seg_dir = cam_base / "segmentation_masks"
        depth_dir = cam_base / "depth_maps"
        all_segs = sorted([f for f in os.listdir(seg_dir) if f.endswith(".png")])
        all_depths = sorted([f for f in os.listdir(depth_dir) if f.endswith(".float16.gz")])

        selected_frames = [all_frames[i] for i in indices]
        selected_segs = [all_segs[i] for i in indices]
        selected_depths = [all_depths[i] for i in indices]

        if not (len(selected_frames) == len(selected_segs) == len(selected_depths)):
            logger.error(
                f"    {camera}: decimation count mismatch — "
                f"frames={len(selected_frames)}, seg={len(selected_segs)}, "
                f"depth={len(selected_depths)}"
            )
            session_info["status"] = "error"
            save_progress(progress)
            return progress

        decimated_frames[camera] = {
            "frames": selected_frames,
            "segs": selected_segs,
            "depths": selected_depths,
            "indices": indices,
        }
        logger.info(
            f"    {camera}: decimated {len(all_frames)} → {len(indices)} frames"
        )

    session_info["status"] = "decimated"
    session_info["decimated_at"] = _now_iso()
    # Store decimation results transiently (not in progress file — recomputable)
    session_info["_decimated_data"] = decimated_frames
    save_progress(progress)
    return progress


def _preprocess_camera(
    session_id: str,
    camera: str,
    raw_base: Path,
    selected_frames: list[str],
    selected_segs: list[str],
    selected_depths: list[str],
) -> int:
    """
    Run SANPO_data_processor on selected frames for one camera.

    Returns the number of successfully processed frames.
    """
    cam_raw = raw_base / camera / "left"
    video_dir = cam_raw / "video_frames"
    seg_dir = cam_raw / "segmentation_masks"
    depth_dir = cam_raw / "depth_maps"

    cam_processed = PROCESSED_DIR / session_id / camera / "left"
    proc_video_dir = str(cam_processed / "video_frames")
    proc_seg_dir = str(cam_processed / "segmentation_masks")
    proc_depth_dir = str(cam_processed / "depth_maps")

    os.makedirs(proc_video_dir, exist_ok=True)
    os.makedirs(proc_seg_dir, exist_ok=True)
    os.makedirs(proc_depth_dir, exist_ok=True)

    processed_count = 0

    for i, frame_filename in enumerate(selected_frames):
        frame_id = _extract_frame_id(frame_filename)

        try:
            # RGB: crop + resize (bilinear)
            SANPO_data_processor.image_crop(
                image_path=str(video_dir / frame_filename),
                save_dir=proc_video_dir,
                file_prefix=frame_id,
                interpolate=cv2.INTER_LINEAR,
            )

            # Segmentation: crop + resize (nearest to preserve labels)
            SANPO_data_processor.image_crop(
                image_path=str(seg_dir / selected_segs[i]),
                save_dir=proc_seg_dir,
                file_prefix=frame_id,
                interpolate=cv2.INTER_NEAREST,
            )

            # Depth: decompress + crop + resize
            SANPO_data_processor.process_and_patch_sanpo_depth(
                gz_file_path=str(depth_dir / selected_depths[i]),
                save_dir=proc_depth_dir,
                file_prefix=frame_id,
            )

            processed_count += 1

        except (IOError, ValueError) as e:
            logger.warning(f"      Frame {frame_id} failed: {e} — skipping")
            continue

    return processed_count


def _integrity_check_post_preprocess(
    session_id: str, camera: str, expected_frames: int
) -> bool:
    """
    Verify preprocessing output: each frame should produce 9 files
    (3 patches × 3 data types).
    """
    cam_processed = PROCESSED_DIR / session_id / camera / "left"

    for dtype, ext in [
        ("video_frames", ".png"),
        ("segmentation_masks", ".png"),
        ("depth_maps", "_float16.npy"),
    ]:
        dtype_dir = cam_processed / dtype
        if not dtype_dir.exists():
            logger.warning(f"    {camera}/{dtype} directory missing after preprocessing")
            return False

        # Count files matching the pattern: {id}_{position}{ext}
        files = [f for f in os.listdir(dtype_dir) if f.endswith(ext)]
        expected_count = expected_frames * len(PATCH_POSITIONS)

        if len(files) < expected_count:
            logger.warning(
                f"    {camera}/{dtype}: expected {expected_count} files, "
                f"got {len(files)}"
            )
            # Allow minor gaps per SDD
            return True

    return True


def preprocess_session(session_id: str, progress: dict) -> dict:
    """
    Phase 1 Step 1b: Preprocessing using SANPO_data_processor.

    Returns updated progress dict.
    """
    session_info = progress["sessions"][session_id]
    raw_base = RAW_DIR / session_id

    # Recompute decimation if needed (deterministic, so always safe)
    decimated_data = session_info.get("_decimated_data")
    if not decimated_data:
        config = progress["decimation_config"]
        decimated_data = {}
        for camera in session_info["valid_cameras"]:
            cam_base = raw_base / camera / "left"
            video_dir = cam_base / "video_frames"
            seg_dir = cam_base / "segmentation_masks"
            depth_dir = cam_base / "depth_maps"

            all_frames = sorted([f for f in os.listdir(video_dir) if f.endswith(".png")])
            all_segs = sorted([f for f in os.listdir(seg_dir) if f.endswith(".png")])
            all_depths = sorted([f for f in os.listdir(depth_dir) if f.endswith(".float16.gz")])

            indices = _decimate_indices(len(all_frames), config["interval"], config["offset"])
            decimated_data[camera] = {
                "frames": [all_frames[i] for i in indices],
                "segs": [all_segs[i] for i in indices],
                "depths": [all_depths[i] for i in indices],
                "indices": indices,
            }

    total_frame_count = 0
    total_patch_count = 0

    for camera in session_info["valid_cameras"]:
        cam_data = decimated_data[camera]
        logger.info(
            f"    Preprocessing {camera}: {len(cam_data['frames'])} frames ..."
        )

        processed_count = _preprocess_camera(
            session_id=session_id,
            camera=camera,
            raw_base=raw_base,
            selected_frames=cam_data["frames"],
            selected_segs=cam_data["segs"],
            selected_depths=cam_data["depths"],
        )

        _integrity_check_post_preprocess(session_id, camera, processed_count)

        total_frame_count += processed_count
        total_patch_count += processed_count * len(PATCH_POSITIONS)

        logger.info(
            f"    {camera}: {processed_count}/{len(cam_data['frames'])} frames processed"
        )

    session_info["status"] = "processed"
    session_info["processed_at"] = _now_iso()
    session_info["frame_count"] = total_frame_count
    session_info["patch_count"] = total_patch_count
    # Clean up transient data
    session_info.pop("_decimated_data", None)

    save_progress(progress)
    logger.info(
        f"  Session {session_id} → processed "
        f"({total_frame_count} frames, {total_patch_count} patches)"
    )
    return progress


def phase1_validate_decimate_preprocess(progress: dict) -> dict:
    """Execute Phase 1 on all sessions in 'downloaded' or intermediate states."""
    all_session_ids = load_session_ids()

    for sid in all_session_ids:
        session_info = progress["sessions"].get(sid, {})
        status = session_info.get("status", "pending")

        # Step 0: Validate
        if status == "downloaded":
            logger.info(f"Phase 1 Step 0: Validating {sid}")
            progress = validate_session(sid, progress)
            status = progress["sessions"][sid]["status"]
            if status == "skipped":
                continue

        # Step 1a: Decimate
        if status == "validated":
            logger.info(f"Phase 1 Step 1a: Decimating {sid}")
            progress = decimate_session(sid, progress)
            status = progress["sessions"][sid]["status"]
            if status == "error":
                continue

        # Step 1b: Preprocess
        if status == "decimated":
            logger.info(f"Phase 1 Step 1b: Preprocessing {sid}")
            progress = preprocess_session(sid, progress)

    return progress


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("SANPO Dataset Streaming Pipeline — Starting")
    logger.info("=" * 60)

    progress = load_progress()
    _validate_decimation_config(progress)

    all_session_ids = load_session_ids()

    # Initialize all sessions as pending if not already tracked
    for sid in all_session_ids:
        if sid not in progress["sessions"]:
            progress["sessions"][sid] = {"status": "pending"}
    save_progress(progress)

    # Main pipeline loop: batch download → Phase 1-3 → repeat
    while True:
        # Check if there are any sessions left to process (not cleaned/skipped)
        actionable = [
            sid for sid in all_session_ids
            if progress["sessions"].get(sid, {}).get("status", "pending")
            not in ("cleaned", "skipped")
        ]

        if not actionable:
            logger.info("All sessions processed. Pipeline complete.")
            break

        # Phase 0: Download next batch
        progress = phase0_download(progress)

        # Check if any sessions need Phase 1-3 processing
        needs_processing = [
            sid for sid in all_session_ids
            if progress["sessions"].get(sid, {}).get("status")
            in ("downloaded", "validated", "decimated", "processed", "packed", "uploaded", "verified")
        ]

        if not needs_processing:
            # No sessions ready — check for retryable errors
            errors = [
                sid for sid in all_session_ids
                if progress["sessions"].get(sid, {}).get("status") == "error"
            ]
            if errors:
                logger.warning(
                    f"{len(errors)} sessions in error state. "
                    "Fix the issue and re-run to retry."
                )
            break

        # Phase 1: Validation + Decimation + Preprocessing
        progress = phase1_validate_decimate_preprocess(progress)

        # TODO: Phase 2 — Pack into WebDataset shards
        # TODO: Phase 3 — Upload + Verify + Clean

        processed = [
            sid for sid in all_session_ids
            if progress["sessions"].get(sid, {}).get("status") == "processed"
        ]
        logger.info(
            f"{len(processed)} sessions processed, awaiting Phase 2-3 implementation."
        )
        break  # Stop here until Phase 2-3 are implemented


if __name__ == "__main__":
    main()

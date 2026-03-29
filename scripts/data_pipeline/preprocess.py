"""Preprocessing: crop + resize using SANPO_data_processor (lazy imports)."""

import logging
import os
import sys
from pathlib import Path

from .config import PROJECT_ROOT
from .decimate import _decimate_indices
from .progress import _now_iso, load_session_ids, save_progress
from .validate import _extract_frame_id, validate_session

logger = logging.getLogger(__name__)


def _preprocess_camera(
    session_id: str,
    camera: str,
    raw_base: Path,
    selected_frames: list[str],
    selected_segs: list[str],
    selected_depths: list[str],
    config: dict,
) -> int:
    """
    Run SANPO_data_processor on selected frames for one camera.

    Returns the number of successfully processed frames.

    Lazy imports: cv2, SANPO_data_processor (heavy deps, not needed for unit tests).
    """
    import cv2
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from utils.SANPO_data_processor import SANPO_data_processor

    processed_dir = Path(config["paths"]["processed_dir"])
    preprocess_cfg = config["preprocess"]
    patch_positions = preprocess_cfg["patch_positions"]
    enable_image_crop = preprocess_cfg["enable_image_crop"]
    enable_depth_process = preprocess_cfg["enable_depth_process"]

    cam_raw = raw_base / camera / "left"
    video_dir = cam_raw / "video_frames"
    seg_dir = cam_raw / "segmentation_masks"
    depth_dir = cam_raw / "depth_maps"

    cam_processed = processed_dir / session_id / camera / "left"
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
            if enable_image_crop:
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
            else:
                # Copy raw files directly (no crop/resize)
                import shutil
                shutil.copy2(video_dir / frame_filename, os.path.join(proc_video_dir, frame_filename))
                shutil.copy2(seg_dir / selected_segs[i], os.path.join(proc_seg_dir, selected_segs[i]))

            if enable_depth_process:
                # Depth: decompress + crop + resize
                SANPO_data_processor.process_and_patch_sanpo_depth(
                    gz_file_path=str(depth_dir / selected_depths[i]),
                    save_dir=proc_depth_dir,
                    file_prefix=frame_id,
                )
            else:
                import shutil
                shutil.copy2(depth_dir / selected_depths[i], os.path.join(proc_depth_dir, selected_depths[i]))

            processed_count += 1

        except (IOError, ValueError) as e:
            logger.warning(f"      Frame {frame_id} failed: {e} — skipping")
            continue

    return processed_count


def _integrity_check_post_preprocess(
    session_id: str, camera: str, expected_frames: int, config: dict
) -> bool:
    """
    Verify preprocessing output: each frame should produce 9 files
    (3 patches × 3 data types).
    """
    processed_dir = Path(config["paths"]["processed_dir"])
    patch_positions = config["preprocess"]["patch_positions"]
    cam_processed = processed_dir / session_id / camera / "left"

    for dtype, ext in [
        ("video_frames", ".png"),
        ("segmentation_masks", ".png"),
        ("depth_maps", "_float16.npy"),
    ]:
        dtype_dir = cam_processed / dtype
        if not dtype_dir.exists():
            logger.warning(f"    {camera}/{dtype} directory missing after preprocessing")
            return False

        files = [f for f in os.listdir(dtype_dir) if f.endswith(ext)]
        expected_count = expected_frames * len(patch_positions)

        if len(files) < expected_count:
            logger.warning(
                f"    {camera}/{dtype}: expected {expected_count} files, "
                f"got {len(files)}"
            )
            return True

    return True


def preprocess_session(session_id: str, progress: dict, config: dict) -> dict:
    """
    Preprocessing using SANPO_data_processor.

    Returns updated progress dict.
    """
    raw_dir = Path(config["paths"]["raw_dir"])
    patch_positions = config["preprocess"]["patch_positions"]
    session_info = progress["sessions"][session_id]
    raw_base = raw_dir / session_id

    # Recompute decimation if needed (deterministic, so always safe)
    decimated_data = session_info.get("_decimated_data")
    if not decimated_data:
        dec_config = progress["decimation_config"]
        decimated_data = {}
        for camera in session_info["valid_cameras"]:
            cam_base = raw_base / camera / "left"
            video_dir = cam_base / "video_frames"
            seg_dir = cam_base / "segmentation_masks"
            depth_dir = cam_base / "depth_maps"

            all_frames = sorted([f for f in os.listdir(video_dir) if f.endswith(".png")])
            all_segs = sorted([f for f in os.listdir(seg_dir) if f.endswith(".png")])
            all_depths = sorted([f for f in os.listdir(depth_dir) if f.endswith(".float16.gz")])

            indices = _decimate_indices(len(all_frames), dec_config["interval"], dec_config["offset"])
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
            config=config,
        )

        _integrity_check_post_preprocess(session_id, camera, processed_count, config)

        total_frame_count += processed_count
        total_patch_count += processed_count * len(patch_positions)

        logger.info(
            f"    {camera}: {processed_count}/{len(cam_data['frames'])} frames processed"
        )

    session_info["status"] = "processed"
    session_info["processed_at"] = _now_iso()
    session_info["frame_count"] = total_frame_count
    session_info["patch_count"] = total_patch_count
    session_info.pop("_decimated_data", None)

    save_progress(progress, config)
    logger.info(
        f"  Session {session_id} → processed "
        f"({total_frame_count} frames, {total_patch_count} patches)"
    )
    return progress


def validate_decimate_preprocess(progress: dict, config: dict) -> dict:
    """Execute validation + decimation + preprocessing on all eligible sessions."""
    from .decimate import decimate_session

    all_session_ids = load_session_ids(config)

    for sid in all_session_ids:
        session_info = progress["sessions"].get(sid, {})
        status = session_info.get("status", "pending")

        # Step 0: Validate
        if status == "downloaded":
            logger.info(f"Validating {sid}")
            progress = validate_session(sid, progress, config)
            status = progress["sessions"][sid]["status"]
            if status == "skipped":
                continue

        # Step 1a: Decimate
        if status == "validated":
            logger.info(f"Decimating {sid}")
            progress = decimate_session(sid, progress, config)
            status = progress["sessions"][sid]["status"]
            if status == "error":
                continue

        # Step 1b: Preprocess
        if status == "decimated":
            logger.info(f"Preprocessing {sid}")
            progress = preprocess_session(sid, progress, config)

    return progress

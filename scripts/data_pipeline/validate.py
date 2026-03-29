"""Per-camera validation logic."""

import logging
import os
import re
import shutil
from pathlib import Path

from .progress import _now_iso, save_progress

logger = logging.getLogger(__name__)


def _extract_frame_id(filename: str) -> str:
    """Extract numeric prefix from filename. e.g. '000010.png' → '000010'."""
    match = re.match(r"^(\d+)", filename)
    if match:
        return match.group(1)
    return filename


def _discover_cameras(session_dir: Path, config: dict) -> list[str]:
    """Find which cameras exist in a raw session directory."""
    cameras = config["preprocess"]["cameras"]
    found = []
    for camera in cameras:
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
    if not seg_dir.exists():
        return False, f"{camera}: segmentation_masks directory missing", 0
    if not any(seg_dir.iterdir()):
        return False, f"{camera}: segmentation_masks directory empty", 0

    if not depth_dir.exists():
        return False, f"{camera}: depth_maps directory missing", 0
    if not any(depth_dir.iterdir()):
        return False, f"{camera}: depth_maps directory empty", 0

    if not video_dir.exists():
        return False, f"{camera}: video_frames directory missing", 0
    if not any(video_dir.iterdir()):
        return False, f"{camera}: video_frames directory empty", 0

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


def validate_session(session_id: str, progress: dict, config: dict) -> dict:
    """
    Pre-Decimation Validation (per-camera).

    Returns updated progress dict.
    """
    raw_dir = Path(config["paths"]["raw_dir"])
    session_dir = raw_dir / session_id
    cameras = _discover_cameras(session_dir, config)
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
        session_info["status"] = "skipped"
        session_info["skip_reason"] = "; ".join(fail_reasons)
        session_info["skipped_at"] = _now_iso()
        # Delete entire raw session
        if session_dir.exists():
            shutil.rmtree(session_dir)
            logger.info(f"  Deleted raw data: {session_dir}")
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

    save_progress(progress, config)
    return progress

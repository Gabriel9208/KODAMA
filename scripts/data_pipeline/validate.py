"""Per-camera validation logic."""

import logging
import os
import re
import shutil
from pathlib import Path
from typing import Callable

from .progress import _now_iso, save_progress

logger = logging.getLogger(__name__)


def _extract_frame_id(filename: str) -> str:
    """Extract numeric prefix from filename. e.g. '000010.png' → '000010'."""
    match = re.match(r"^(\d+)", filename)
    if match:
        return match.group(1)
    return filename


def _list_local_files(dir_path: str) -> list[str]:
    """List filenames in a local directory. Returns [] if missing or empty."""
    p = Path(dir_path)
    if not p.exists() or not any(p.iterdir()):
        return []
    return sorted(os.listdir(p))


def _discover_cameras(session_dir: Path, config: dict) -> list[str]:
    """Find which cameras exist in a raw session directory."""
    cameras = config["preprocess"]["cameras"]
    found = []
    for camera in cameras:
        cam_dir = session_dir / camera / "left"
        if cam_dir.exists():
            found.append(camera)
    return found


def _validate_camera(
    camera: str,
    list_files_fn: Callable[[str], list[str]],
    video_path: str,
    seg_path: str,
    depth_path: str,
) -> tuple[bool, str, int]:
    """
    Validate a single camera's data integrity.

    Args:
        camera: Camera name (e.g. "camera_chest").
        list_files_fn: Callable that takes a path string and returns sorted
            filenames, or [] if the path doesn't exist / is empty.
        video_path: Path (local or GCS) to video_frames directory.
        seg_path: Path (local or GCS) to segmentation_masks directory.
        depth_path: Path (local or GCS) to depth_maps directory.

    Returns (passed, reason, frame_count).
    If passed is False, reason describes the failure.
    """
    # Check 1: All three directories exist and are non-empty
    seg_files = list_files_fn(seg_path)
    if not seg_files:
        return False, f"{camera}: segmentation_masks directory missing or empty", 0

    depth_files = list_files_fn(depth_path)
    if not depth_files:
        return False, f"{camera}: depth_maps directory missing or empty", 0

    frame_files = list_files_fn(video_path)
    if not frame_files:
        return False, f"{camera}: video_frames directory missing or empty", 0

    # Filter by expected extensions
    frame_files = sorted(f for f in frame_files if f.endswith(".png"))
    seg_files = sorted(f for f in seg_files if f.endswith(".png"))
    depth_files = sorted(f for f in depth_files if f.endswith(".float16.gz"))

    # Check 2: File counts match
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
        cam_base = str(session_dir / camera / "left")
        passed, reason, frame_count = _validate_camera(
            camera,
            _list_local_files,
            f"{cam_base}/video_frames",
            f"{cam_base}/segmentation_masks",
            f"{cam_base}/depth_maps",
        )
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

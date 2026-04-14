"""Decimation (frame sampling) logic."""

import logging
import os
from pathlib import Path

from .progress import _now_iso, save_progress, sessions

logger = logging.getLogger(__name__)


def _decimate_indices(total_frames: int, interval: int, offset: int) -> list[int]:
    """Compute decimated frame indices."""
    return list(range(offset, total_frames, interval))


def decimate_session(session_id: str, progress: dict, config: dict) -> dict:
    """
    Decimation (in-memory index computation).

    Returns updated progress dict.
    """
    raw_dir = Path(config["paths"]["raw_dir"])
    session_info = sessions(progress, config)[session_id]
    dec_config = progress["decimation_config"]
    interval = dec_config["interval"]
    offset = dec_config["offset"]

    valid_cameras = session_info["valid_cameras"]
    decimated_frames = {}

    for camera in valid_cameras:
        cam_base = raw_dir / session_id / camera / "left"
        video_dir = cam_base / "video_frames"
        seg_dir = cam_base / "segmentation_masks"
        depth_dir = cam_base / "depth_maps"

        all_frames = sorted([f for f in os.listdir(video_dir) if f.endswith(".png")])
        all_segs = sorted([f for f in os.listdir(seg_dir) if f.endswith(".png")])
        all_depths = sorted([f for f in os.listdir(depth_dir) if f.endswith(".float16.gz")])

        indices = _decimate_indices(len(all_frames), interval, offset)

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
            save_progress(progress, config)
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
    session_info["_decimated_data"] = decimated_frames
    save_progress(progress, config)
    return progress

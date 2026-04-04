"""Pack processed data into WebDataset shards."""

import json
import logging
import os
import re
import tarfile
from pathlib import Path

from .progress import _now_iso, save_progress

logger = logging.getLogger(__name__)


def _get_batch_index(progress: dict) -> int:
    """Determine the current batch index from progress."""
    return progress.get("next_batch_index", 0)


def _increment_batch_index(progress: dict) -> None:
    """Increment batch index after a successful batch."""
    progress["next_batch_index"] = progress.get("next_batch_index", 0) + 1


def _parse_processed_filename(filename: str) -> tuple[str, str] | None:
    """
    Parse a processed filename like '000010_left.png' → ('000010', 'left').
    Returns None if parsing fails.
    """
    match = re.match(r"^(\d+)_(left|center|right)", filename)
    if match:
        return match.group(1), match.group(2)
    return None


def pack_shards(batch_sessions: list[str], progress: dict, config: dict) -> dict:
    """Pack all processed sessions in the batch into WebDataset shards."""
    import webdataset as wds

    shards_dir = Path(config["paths"]["shards_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])
    shard_max_size = int(float(config["packing"]["shard_max_size"]))

    # Safety: data/shards/ must be empty
    shards_dir.mkdir(parents=True, exist_ok=True)
    existing_shards = list(shards_dir.glob("shard-*.tar"))
    if existing_shards:
        logger.error(
            f"data/shards/ is not empty ({len(existing_shards)} files). "
            "Previous batch cleanup incomplete. Aborting."
        )
        return progress

    batch_index = _get_batch_index(progress)
    shard_pattern = os.path.relpath(shards_dir / f"shard-{batch_index:03d}-%06d.tar")
    decimation_config = progress["decimation_config"]

    total_samples = 0
    expected_samples = sum(
        progress["sessions"][sid].get("patch_count", 0) for sid in batch_sessions
    )

    logger.info(
        f"Packing {len(batch_sessions)} sessions into shards "
        f"(batch {batch_index}, expected ~{expected_samples} samples)"
    )

    with wds.ShardWriter(shard_pattern, maxsize=shard_max_size) as sink:
        for sid in batch_sessions:
            session_info = progress["sessions"][sid]
            valid_cameras = session_info.get("valid_cameras", [])

            for camera in valid_cameras:
                camera_short = camera.replace("camera_", "")
                proc_base = processed_dir / sid / camera / "left"
                video_dir = proc_base / "video_frames"
                seg_dir = proc_base / "segmentation_masks"
                depth_dir = proc_base / "depth_maps"

                if not video_dir.exists():
                    logger.warning(f"  {sid}/{camera}: video_frames dir missing, skipping")
                    continue

                rgb_files = sorted(
                    f for f in os.listdir(video_dir) if f.endswith(".png")
                )

                for rgb_filename in rgb_files:
                    parsed = _parse_processed_filename(rgb_filename)
                    if not parsed:
                        continue
                    frame_index, patch = parsed

                    key = f"{sid}_{camera_short}_{frame_index}_{patch}"

                    seg_file = seg_dir / f"{frame_index}_{patch}.png"
                    depth_file = depth_dir / f"{frame_index}_{patch}_float16.npy"

                    if not seg_file.exists() or not depth_file.exists():
                        logger.warning(f"  Missing files for {key}, skipping")
                        continue

                    metadata = {
                        "session_id": str(sid),              
                        "camera": str(camera),                
                        "side": "left",
                        "frame_index": str(int(frame_index)),
                        "patch": str(patch),                  
                        "original_filename": f"{frame_index}.png",
                        "decimation_interval": str(decimation_config["interval"]),
                        "decimation_offset": str(decimation_config["offset"]),
                    }

                    sample = {
                        "__key__": key,
                        "png": open(video_dir / rgb_filename, "rb").read(),
                        "seg.png": open(seg_file, "rb").read(),
                        "depth.npy": open(depth_file, "rb").read(),
                        "json": json.dumps(metadata),
                    }
                    sink.write(sample)
                    total_samples += 1

    # Record shard files
    shard_files = sorted(f.name for f in shards_dir.glob("shard-*.tar"))
    logger.info(
        f"Packed {total_samples} samples into {len(shard_files)} shards"
    )

    if total_samples == 0:
        logger.error("No samples packed. Aborting.")
        return progress

    # Integrity check: verify shard contents
    verified_count = 0
    for shard_file in shard_files:
        shard_path = shards_dir / shard_file
        with tarfile.open(shard_path, "r") as tar:
            members = tar.getnames()
            keys = set()
            for m in members:
                base = m.split(".")[0]
                keys.add(base)
            verified_count += len(keys)

    if verified_count != total_samples:
        logger.warning(
            f"Pack integrity: expected {total_samples} samples, "
            f"found {verified_count} in shards"
        )

    # Update all batch sessions
    for sid in batch_sessions:
        progress["sessions"][sid]["status"] = "packed"
        progress["sessions"][sid]["packed_at"] = _now_iso()
        progress["sessions"][sid]["shard_files"] = shard_files

    save_progress(progress, config)
    return progress

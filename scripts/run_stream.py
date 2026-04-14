"""
SANPO Dataset Streaming Pipeline — Mount & Stream
Mounts Google Drive via rclone and streams WebDataset shards for training.

Execution environment: Ubuntu (2×P100 16GB)

Usage:
    # 1. Mount and stream (default config)
    python scripts/run_stream.py

    # 2. With custom config
    python scripts/run_stream.py --config configs/pipeline_config_test.yaml

    # 3. Unmount after training
    fusermount -u ~/gdrive/shards/
"""

import argparse
import glob
import io
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import webdataset as wds
from PIL import Image
from torch.utils.data import DataLoader

# Ensure scripts/ is on sys.path so data_pipeline package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_pipeline.config import load_pipeline_config

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
# Mount Google Drive
# ---------------------------------------------------------------------------

def is_mounted(mount_point: Path) -> bool:
    """Check if the mount point has shard files (i.e. is already mounted)."""
    if not mount_point.exists():
        return False
    shards = list(mount_point.glob("shard-*.tar"))
    return len(shards) > 0


def mount_gdrive(mount_point: Path, gdrive_remote: str) -> None:
    """Mount Google Drive shards directory via rclone."""
    if is_mounted(mount_point):
        shards = sorted(mount_point.glob("shard-*.tar"))
        logger.info(f"Already mounted at {mount_point} ({len(shards)} shards)")
        return

    mount_point.mkdir(parents=True, exist_ok=True)

    logger.info(f"Mounting {gdrive_remote} → {mount_point} ...")
    result = subprocess.run(
        [
            "rclone", "mount",
            gdrive_remote,
            str(mount_point),
            "--vfs-cache-mode", "full",
            "--vfs-cache-max-size", "50G",
            "--buffer-size", "256M",
            "--daemon",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"Mount failed: {result.stderr.strip()}")
        sys.exit(1)

    # Wait briefly and verify
    import time
    for attempt in range(10):
        time.sleep(2)
        if is_mounted(mount_point):
            shards = sorted(mount_point.glob("shard-*.tar"))
            logger.info(f"Mount successful ({len(shards)} shards)")
            return

    logger.error("Mount appeared to succeed but no shards found. Check rclone config.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# WebDataset streaming
# ---------------------------------------------------------------------------

def log_and_continue(exn):
    """Error handler: log corrupt samples and continue."""
    logger.warning(f"Skipping corrupt sample: {exn}")
    return True


def seg_decoder(key, data):
    """Decode _seg.png as raw uint8 numpy array, bypassing float normalization."""
    if key.endswith("_seg.png"):
        return np.array(Image.open(io.BytesIO(data)))  # (H, W, 3) uint8
    return None  # fallback to default decoder


def decode_seg_mask(seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Decode segmentation mask from RGB to semantic + instance channels.

    Args:
        seg: (H, W, 3) uint8 RGB array

    Returns:
        semantic: (H, W) uint8 — class ID (0-30)
        instance: (H, W) int32 — instance ID (G*256 + B)
    """
    semantic = seg[:, :, 0]
    instance = seg[:, :, 1].astype(np.int32) * 256 + seg[:, :, 2].astype(np.int32)
    return semantic, instance


def create_dataset(
    mount_point: Path,
    batch_size: int,
    shuffle_buffer: int,
) -> wds.WebDataset:
    """Create a WebDataset pipeline from mounted shards."""
    shards = sorted(glob.glob(str(mount_point / "shard-*.tar")))
    if not shards:
        logger.error(f"No shards found at {mount_point}")
        sys.exit(1)

    logger.info(f"Found {len(shards)} shards")

    dataset = (
        wds.WebDataset(shards, shardshuffle=True, handler=log_and_continue)
        .shuffle(shuffle_buffer)
        .decode(seg_decoder, "rgb", handler=log_and_continue)
        .to_tuple("png", "_seg.png", "_depth.npy", "json", handler=log_and_continue)
        .batched(batch_size)
    )

    return dataset


def create_dataloader(
    dataset: wds.WebDataset,
    num_workers: int,
) -> DataLoader:
    """Wrap WebDataset in a DataLoader."""
    return DataLoader(dataset, batch_size=None, num_workers=num_workers)


# ---------------------------------------------------------------------------
# Training loop (skeleton)
# ---------------------------------------------------------------------------

def train(args):
    """Main training entry point."""
    config_path = Path(args.config) if args.config else None
    config = load_pipeline_config(config_path)
    streaming_cfg = config["streaming"]
    gdrive_remote = config["remote"]["gdrive_remote"]

    # Resolve mount point (CLI override > config)
    mount_point = Path(args.mount_point) if args.mount_point else Path(streaming_cfg["mount_point"]).expanduser()
    batch_size = args.batch_size or streaming_cfg["batch_size"]
    shuffle_buffer = args.shuffle_buffer or streaming_cfg["shuffle_buffer"]
    num_workers = args.num_workers or streaming_cfg["num_workers"]

    # Mount
    mount_gdrive(mount_point, gdrive_remote)

    # Create dataset
    dataset = create_dataset(
        mount_point=mount_point,
        batch_size=batch_size,
        shuffle_buffer=shuffle_buffer,
    )
    loader = create_dataloader(dataset, num_workers=num_workers)

    # Training loop skeleton
    logger.info(
        f"Starting training — batch_size={batch_size}, "
        f"shuffle_buffer={shuffle_buffer}, num_workers={num_workers}"
    )

    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        sample_count = 0

        for batch_idx, batch in enumerate(loader):
            rgb, seg, depth, metadata = batch
            # rgb:   list of (H, W, 3) float32 [0, 1]  — decoded by "rgb"
            # seg:   list of (H, W, 3) uint8            — decoded by seg_decoder
            # depth: list of (H, W) float16             — loaded from .npy
            # metadata: list of dict

            current_batch_size = len(rgb)
            sample_count += current_batch_size

            # --- Decode segmentation masks ---
            # for i in range(current_batch_size):
            #     semantic, instance = decode_seg_mask(seg[i])

            # --- TODO: Your training logic here ---
            # 1. Convert to tensors
            # 2. Forward pass
            # 3. Compute loss
            # 4. Backward + optimize

            if (batch_idx + 1) % 50 == 0:
                logger.info(
                    f"  Batch {batch_idx + 1}: {sample_count} samples processed"
                )

        logger.info(f"Epoch {epoch + 1} complete — {sample_count} samples")

    logger.info("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="SANPO dataset streaming pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to pipeline config YAML (default: configs/pipeline_config.yaml)",
    )
    parser.add_argument(
        "--mount-point",
        type=str,
        default=None,
        help="rclone mount path (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Per-GPU batch size (overrides config)",
    )
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=None,
        help="Shuffle buffer size (overrides config)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers (overrides config)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

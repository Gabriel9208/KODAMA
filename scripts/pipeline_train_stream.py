"""
SANPO Dataset Streaming Training Pipeline — Phase 4
Mounts Google Drive via rclone and streams WebDataset shards for training.

Execution environment: Ubuntu (2×P100 16GB)

Usage:
    # 1. Mount first (or let this script do it)
    python scripts/pipeline_train_stream.py

    # 2. Unmount after training
    fusermount -u ~/gdrive/shards/
"""

import argparse
import glob
import io
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import webdataset as wds
from PIL import Image
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MOUNT_POINT = Path.home() / "gdrive" / "shards"
GDRIVE_REMOTE = "gdrive:SANPO-Dataset/shards/"

DEFAULT_BATCH_SIZE = 8
DEFAULT_SHUFFLE_BUFFER = 2000
DEFAULT_NUM_WORKERS = 4

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
# Step 4a: Mount Google Drive
# ---------------------------------------------------------------------------

def is_mounted(mount_point: Path) -> bool:
    """Check if the mount point has shard files (i.e. is already mounted)."""
    if not mount_point.exists():
        return False
    shards = list(mount_point.glob("shard-*.tar"))
    return len(shards) > 0


def mount_gdrive(mount_point: Path) -> None:
    """Mount Google Drive shards directory via rclone."""
    if is_mounted(mount_point):
        shards = sorted(mount_point.glob("shard-*.tar"))
        logger.info(f"Already mounted at {mount_point} ({len(shards)} shards)")
        return

    mount_point.mkdir(parents=True, exist_ok=True)

    logger.info(f"Mounting {GDRIVE_REMOTE} → {mount_point} ...")
    result = subprocess.run(
        [
            "rclone", "mount",
            GDRIVE_REMOTE,
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
# Step 4b: WebDataset streaming
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
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle_buffer: int = DEFAULT_SHUFFLE_BUFFER,
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
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> DataLoader:
    """Wrap WebDataset in a DataLoader."""
    return DataLoader(dataset, batch_size=None, num_workers=num_workers)


# ---------------------------------------------------------------------------
# Training loop (skeleton)
# ---------------------------------------------------------------------------

def train(args):
    """Main training entry point."""
    mount_point = Path(args.mount_point)

    # Step 4a: Mount
    mount_gdrive(mount_point)

    # Step 4b: Create dataset
    dataset = create_dataset(
        mount_point=mount_point,
        batch_size=args.batch_size,
        shuffle_buffer=args.shuffle_buffer,
    )
    loader = create_dataloader(dataset, num_workers=args.num_workers)

    # Training loop skeleton
    logger.info(
        f"Starting training — batch_size={args.batch_size}, "
        f"shuffle_buffer={args.shuffle_buffer}, num_workers={args.num_workers}"
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

            batch_size = len(rgb)
            sample_count += batch_size

            # --- Decode segmentation masks ---
            # for i in range(batch_size):
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
        description="SANPO streaming training pipeline (Phase 4)"
    )
    parser.add_argument(
        "--mount-point",
        type=str,
        default=str(DEFAULT_MOUNT_POINT),
        help=f"rclone mount path (default: {DEFAULT_MOUNT_POINT})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Per-GPU batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=DEFAULT_SHUFFLE_BUFFER,
        help=f"Shuffle buffer size (default: {DEFAULT_SHUFFLE_BUFFER})",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=f"DataLoader workers (default: {DEFAULT_NUM_WORKERS})",
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

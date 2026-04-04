"""Upload shards to Google Drive, verify, and clean up."""

import logging
import shutil
import subprocess
from pathlib import Path

from .pack import _increment_batch_index
from .progress import _now_iso, save_progress

logger = logging.getLogger(__name__)


def upload_shards(batch_sessions: list[str], progress: dict, config: dict) -> dict:
    """Upload shards to Google Drive via rclone."""
    shards_dir = Path(config["paths"]["shards_dir"])
    gdrive_remote = config["remote"]["gdrive_remote"]
    transfers = str(config["upload"]["rclone_transfers"])
    chunk_size = config["upload"]["rclone_chunk_size"]

    logger.info("Uploading shards to Google Drive ...")

    rclone_bin = shutil.which("rclone")
    if rclone_bin is None:
        raise FileNotFoundError(
            "rclone not found on PATH. Install it or add its directory to PATH."
        )

    result = subprocess.run(
        [
            rclone_bin, "copy",
            str(shards_dir),
            gdrive_remote,
            "--transfers", transfers,
            "--drive-chunk-size", chunk_size,
            "--progress",
        ],
        capture_output=False,
    )

    if result.returncode != 0:
        logger.error("rclone copy failed. Will retry on next run.")
        return progress

    for sid in batch_sessions:
        progress["sessions"][sid]["status"] = "uploaded"
        progress["sessions"][sid]["uploaded_at"] = _now_iso()

    save_progress(progress, config)
    logger.info("Upload complete.")
    return progress


def verify_shards(batch_sessions: list[str], progress: dict, config: dict) -> dict:
    """Verify uploaded shards via rclone check."""
    shards_dir = Path(config["paths"]["shards_dir"])
    gdrive_remote = config["remote"]["gdrive_remote"]

    logger.info("Verifying shards on Google Drive ...")

    result = subprocess.run(
        [
            "rclone", "check",
            str(shards_dir),
            gdrive_remote,
            "--one-way",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(
            f"rclone check failed.\n{result.stderr.strip()}\n"
            "Will re-upload and re-verify on next run."
        )
        for sid in batch_sessions:
            progress["sessions"][sid]["status"] = "packed"
        save_progress(progress, config)
        return progress

    for sid in batch_sessions:
        progress["sessions"][sid]["status"] = "verified"
        progress["sessions"][sid]["verified_at"] = _now_iso()

    save_progress(progress, config)
    logger.info("Verification passed.")
    return progress


def cleanup_batch(batch_sessions: list[str], progress: dict, config: dict) -> dict:
    """Delete local raw, processed, and shard files after verification."""
    raw_dir = Path(config["paths"]["raw_dir"])
    processed_dir = Path(config["paths"]["processed_dir"])
    shards_dir = Path(config["paths"]["shards_dir"])

    # Safety: all batch sessions must be verified
    all_verified = all(
        progress["sessions"][sid]["status"] == "verified"
        for sid in batch_sessions
    )
    if not all_verified:
        logger.error("Not all sessions verified. Aborting cleanup.")
        return progress

    logger.info(f"Cleaning up {len(batch_sessions)} sessions ...")

    for sid in batch_sessions:
        raw = raw_dir / sid
        proc = processed_dir / sid

        if raw.exists():
            shutil.rmtree(raw)
            logger.info(f"  Deleted raw: {sid}")

        if proc.exists():
            shutil.rmtree(proc)
            logger.info(f"  Deleted processed: {sid}")

    # Delete all shards
    shard_files = list(shards_dir.glob("shard-*.tar"))
    for f in shard_files:
        f.unlink()
    logger.info(f"  Deleted {len(shard_files)} shard files")

    for sid in batch_sessions:
        progress["sessions"][sid]["status"] = "cleaned"
        progress["sessions"][sid]["cleaned_at"] = _now_iso()

    _increment_batch_index(progress)
    save_progress(progress, config)
    logger.info("Cleanup complete. Ready for next batch.")
    return progress

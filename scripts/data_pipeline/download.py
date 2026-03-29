"""Download sessions from GCS."""

import logging
import shutil
import subprocess
from pathlib import Path

from .progress import (
    _available_disk_gb,
    _now_iso,
    load_session_ids,
    save_progress,
)

logger = logging.getLogger(__name__)


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


def _delete_raw_session(session_id: str, config: dict) -> None:
    """Delete raw data for a session."""
    raw_dir = Path(config["paths"]["raw_dir"])
    session_dir = raw_dir / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir)
        logger.info(f"  Deleted raw data: {session_dir}")


def download_session(session_id: str, progress: dict, config: dict) -> str:
    """
    Download a single session from GCS.

    Returns the new status: "downloaded", "skipped", or "error".
    """
    gcs_bucket = config["remote"]["gcs_bucket"]
    raw_dir = Path(config["paths"]["raw_dir"])
    gcs_base = f"{gcs_bucket}/{session_id}"
    local_base = raw_dir / session_id

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

            if not _gcloud_ls(gcs_src):
                if camera == "camera_chest" and dtype == "video_frames":
                    logger.warning(f"  {camera}/left/{dtype} not found in GCS — marking skipped")
                    _delete_raw_session(session_id, config)
                    return "skipped"
                else:
                    logger.info(f"  {camera}/left/{dtype} not found in GCS — skipping this directory")
                    continue

            logger.info(f"  Downloading {camera}/left/{dtype} ...")
            success = _gcloud_cp(gcs_src, local_dst)
            if not success:
                logger.error(f"  Download failed for {camera}/left/{dtype}")
                _delete_raw_session(session_id, config)
                return "error"

    # Integrity check
    if not _integrity_check_post_download(local_base):
        logger.warning(f"  Integrity check failed for {session_id} — video_frames missing or empty")
        _delete_raw_session(session_id, config)
        return "skipped"

    return "downloaded"


def download_batch(progress: dict, config: dict) -> dict:
    """Execute download for one batch of sessions."""
    all_session_ids = load_session_ids(config)
    batch_size = config["download"]["batch_size"]
    disk_safety_factor = config["download"]["disk_safety_factor"]
    estimated_session_size_gb = config["download"]["estimated_session_size_gb"]
    data_dir = Path(config["paths"]["data_dir"])

    # Filter to sessions that need downloading
    pending = []
    for sid in all_session_ids:
        session_info = progress["sessions"].get(sid, {})
        status = session_info.get("status", "pending")
        if status in ("pending", "error"):
            pending.append(sid)

    if not pending:
        logger.info("Download: No sessions to download.")
        return progress

    logger.info(f"Download: {len(pending)} sessions to download ({len(all_session_ids)} total)")

    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start : batch_start + batch_size]

        # Disk space check
        available_gb = _available_disk_gb(data_dir)
        required_gb = len(batch) * estimated_session_size_gb * disk_safety_factor
        if available_gb < required_gb:
            logger.warning(
                f"Insufficient disk space: {available_gb:.1f} GB available, "
                f"{required_gb:.1f} GB required for {len(batch)} sessions. "
                "Pausing. Free space or wait for previous batch to upload, then re-run."
            )
            break

        logger.info(
            f"Download batch: downloading {len(batch)} sessions "
            f"(available: {available_gb:.1f} GB, est. required: {required_gb:.1f} GB)"
        )

        for sid in batch:
            logger.info(f"Processing session: {sid}")
            new_status = download_session(sid, progress, config)

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

            save_progress(progress, config)

        logger.info(f"Download batch complete. Proceeding to next phases.")
        break  # Process one batch at a time

    return progress

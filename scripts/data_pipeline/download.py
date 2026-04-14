"""Download sessions from GCS."""

import logging
import shutil
import subprocess
from pathlib import Path

from .progress import (
    _available_disk_gb,
    _now_iso,
    clear_stale_pending_sessions,
    load_session_ids,
    save_progress,
    sessions,
)

logger = logging.getLogger(__name__)


def _gcloud_ls(gcs_path: str) -> bool:
    """Check if a GCS path exists via gcloud storage ls. Returns True if exists."""
    result = subprocess.run(
        ["gcloud", "storage", "ls", gcs_path],
        capture_output=True,
        text=True
    )
    return result.returncode == 0


def _gcloud_cp(gcs_src: str, local_dst: Path) -> bool:
    """Download from GCS. Returns True on success."""
    local_dst.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["gcloud", "storage", "cp", "-r", gcs_src, str(local_dst)],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        logger.error(f"gcloud cp failed: {result.stderr.strip()}")
        return False
    return True


def _integrity_check_post_download(session_dir: Path, valid_cameras: list[str]) -> bool:
    """
    Basic check after download: video_frames/ must exist and be non-empty
    for at least one of the downloaded cameras.
    """
    for camera in valid_cameras:
        cam_frames = session_dir / camera / "left" / "video_frames"
        if cam_frames.exists() and any(cam_frames.iterdir()):
            return True
    return False


def _delete_raw_session(session_id: str, config: dict) -> None:
    """Delete raw data for a session."""
    raw_dir = Path(config["paths"]["raw_dir"])
    session_dir = raw_dir / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir)
        logger.info(f"  Deleted raw data: {session_dir}")


def download_session(session_id: str, progress: dict, config: dict) -> dict:
    """
    Download a single session from GCS.

    Returns a dict of session fields to merge into the progress entry.
    Always includes "status". May include: downloaded_at, skipped_at,
    skip_reason, camera_skip_reasons, cameras_missing_depth.
    """
    gcs_bucket = config["remote"]["gcs_bucket"]
    raw_dir = Path(config["paths"]["raw_dir"])
    gcs_base = f"{gcs_bucket}/{session_id}"
    local_base = raw_dir / session_id

    cameras_to_check = ["camera_chest"]

    if _gcloud_ls(f"{gcs_base}/camera_head/left/"):
        cameras_to_check.append("camera_head")
        logger.info(f"  camera_head detected for {session_id}")
    else:
        logger.info(f"  camera_head not found for {session_id}, downloading chest only")

    # Phase A: GCS existence checks with differentiated handling
    camera_skip_reasons: dict[str, str] = {}
    cameras_missing_depth: list[str] = []
    valid_cameras: list[str] = []

    for camera in cameras_to_check:
        skip_this_camera = False
        for dtype in ("video_frames", "segmentation_masks"):
            if not _gcloud_ls(f"{gcs_base}/{camera}/left/{dtype}"):
                reason = f"{dtype} missing on GCS"
                logger.warning(f"  {camera}/left/{dtype} not found in GCS — skipping camera")
                camera_skip_reasons[camera] = reason
                skip_this_camera = True
                break
        if skip_this_camera:
            continue

        # depth_maps is optional — missing is non-fatal for the camera
        if not _gcloud_ls(f"{gcs_base}/{camera}/left/depth_maps"):
            logger.warning(
                f"  {camera}/left/depth_maps not found in GCS — "
                "will download video_frames + segmentation_masks only"
            )
            cameras_missing_depth.append(camera)

        valid_cameras.append(camera)

    # All cameras failed existence check → session-level skip
    if not valid_cameras:
        _delete_raw_session(session_id, config)
        result: dict = {
            "status": "skipped",
            "skip_reason": "no valid cameras: all missing required data types on GCS",
            "skipped_at": _now_iso(),
        }
        if camera_skip_reasons:
            result["camera_skip_reasons"] = camera_skip_reasons
        return result

    # Phase B: Download valid cameras (only confirmed-present data types)
    for camera in valid_cameras:
        dtypes_to_download = ["video_frames", "segmentation_masks"]
        if camera not in cameras_missing_depth:
            dtypes_to_download.append("depth_maps")

        for dtype in dtypes_to_download:
            gcs_src = f"{gcs_base}/{camera}/left/{dtype}"
            local_dst = local_base / camera / "left"
            logger.info(f"  Downloading {camera}/left/{dtype} ...")
            if not _gcloud_cp(gcs_src, local_dst):
                logger.error(f"  Download failed for {camera}/left/{dtype}")
                _delete_raw_session(session_id, config)
                return {"status": "error"}

    # Integrity check
    if not _integrity_check_post_download(local_base, valid_cameras):
        logger.warning(f"  Integrity check failed for {session_id} — video_frames missing or empty")
        _delete_raw_session(session_id, config)
        return {
            "status": "skipped",
            "skip_reason": "integrity check failed: video_frames missing or empty",
            "skipped_at": _now_iso(),
        }

    result = {"status": "downloaded", "downloaded_at": _now_iso()}
    if camera_skip_reasons:
        result["camera_skip_reasons"] = camera_skip_reasons
        result["skip_reason"] = "; ".join(
            f"{cam}: {reason}" for cam, reason in camera_skip_reasons.items()
        )
    if cameras_missing_depth:
        result["cameras_missing_depth"] = cameras_missing_depth
    return result


_IN_FLIGHT_STATUSES = (
    "downloaded", "validated", "decimated", "processed",
    "packed", "uploaded", "verified",
)


def download_batch(progress: dict, config: dict) -> dict:
    """Execute download for one batch of sessions.

    Respects batch_size strictly: counts sessions already in-flight
    (downloaded through verified) and only downloads enough to fill
    the remaining slots. If the batch is already full, downloads nothing.
    """
    all_session_ids = load_session_ids(config)
    batch_size = config["download"]["batch_size"]
    disk_safety_factor = config["download"]["disk_safety_factor"]
    estimated_session_size_gb = config["download"]["estimated_session_size_gb"]
    data_dir = Path(config["paths"]["data_dir"])

    sess = sessions(progress, config)

    # Clear any stale pending sessions before filtering
    cleaned = clear_stale_pending_sessions(progress, config)
    if cleaned:
        save_progress(progress, config)

    # Count sessions already in-flight (downloaded but not yet cleaned)
    in_flight = [
        sid for sid in all_session_ids
        if sess.get(sid, {}).get("status") in _IN_FLIGHT_STATUSES
    ]
    slots_available = batch_size - len(in_flight)

    if slots_available <= 0:
        logger.info(
            f"Download: {len(in_flight)} sessions already in-flight "
            f"(batch_size={batch_size}). Skipping download."
        )
        return progress

    # Filter to sessions that need downloading
    pending = []
    for sid in all_session_ids:
        session_info = sess.get(sid, {})
        status = session_info.get("status", "pending")
        if status in ("pending", "error"):
            pending.append(sid)

    if not pending:
        logger.info("Download: No sessions to download.")
        return progress

    # Only download enough to fill remaining slots
    batch = pending[:slots_available]

    logger.info(
        f"Download: {len(in_flight)} in-flight, "
        f"{slots_available} slots available, "
        f"downloading {len(batch)} sessions"
    )

    # Disk space check
    available_gb = _available_disk_gb(data_dir)
    required_gb = len(batch) * estimated_session_size_gb * disk_safety_factor
    if available_gb < required_gb:
        logger.warning(
            f"Insufficient disk space: {available_gb:.1f} GB available, "
            f"{required_gb:.1f} GB required for {len(batch)} sessions. "
            "Pausing. Free space or wait for previous batch to upload, then re-run."
        )
        return progress

    for sid in batch:
        logger.info(f"Processing session: {sid}")
        session_update = download_session(sid, progress, config)

        if sid not in sess:
            sess[sid] = {}
        sess[sid].update(session_update)
        new_status = session_update["status"]

        if new_status == "downloaded":
            logger.info("  → downloaded")
        elif new_status == "skipped":
            logger.info("  → skipped (permanent)")
        elif new_status == "error":
            logger.info("  → error (will retry on next run)")

        save_progress(progress, config)

    logger.info("Download batch complete. Proceeding to next phases.")
    return progress

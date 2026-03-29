"""
SANPO Dataset Streaming Pipeline — Entry Point
Downloads, validates, decimates, preprocesses, packs, uploads, and cleans SANPO-Real sessions.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --config configs/pipeline_config_test.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure scripts/ is on sys.path so data_pipeline package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_pipeline.config import load_pipeline_config
from data_pipeline.display import format_batch_range, format_resume_summary
from data_pipeline.download import download_batch
from data_pipeline.pack import _get_batch_index, pack_shards
from data_pipeline.preprocess import validate_decimate_preprocess
from data_pipeline.progress import (
    _validate_decimation_config,
    load_progress,
    load_session_ids,
    save_progress,
)
from data_pipeline.upload import cleanup_batch, upload_shards, verify_shards

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
# Main orchestrator
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    config_path = Path(args.config) if args.config else None
    config = load_pipeline_config(config_path)
    batch_size = config["download"]["batch_size"]

    logger.info("=" * 60)
    logger.info("SANPO Dataset Streaming Pipeline — Starting")
    logger.info("=" * 60)

    progress = load_progress(config)
    _validate_decimation_config(progress, config)

    all_session_ids = load_session_ids(config)
    total_sessions = len(all_session_ids)

    # Initialize all sessions as pending if not already tracked
    for sid in all_session_ids:
        if sid not in progress["sessions"]:
            progress["sessions"][sid] = {"status": "pending"}
    save_progress(progress, config)

    # Resume summary
    logger.info(format_resume_summary(progress, total_sessions))

    # Main pipeline loop: batch download → validate/decimate/preprocess → pack/upload/clean → repeat
    while True:
        actionable = [
            sid for sid in all_session_ids
            if progress["sessions"].get(sid, {}).get("status", "pending")
            not in ("cleaned", "skipped")
        ]

        if not actionable:
            logger.info("All sessions processed. Pipeline complete.")
            break

        # Batch header
        batch_index = _get_batch_index(progress)
        batch_range = format_batch_range(batch_index, batch_size, total_sessions)
        logger.info(f"[Batch {batch_index + 1}] Processing sessions {batch_range}")

        # Download
        progress = download_batch(progress, config)

        # Check if any sessions need further processing
        needs_processing = [
            sid for sid in all_session_ids
            if progress["sessions"].get(sid, {}).get("status")
            in ("downloaded", "validated", "decimated", "processed", "packed", "uploaded", "verified")
        ]

        if not needs_processing:
            errors = [
                sid for sid in all_session_ids
                if progress["sessions"].get(sid, {}).get("status") == "error"
            ]
            if errors:
                logger.warning(
                    f"{len(errors)} sessions in error state. "
                    "Fix the issue and re-run to retry."
                )
            break

        # Validate + Decimate + Preprocess
        progress = validate_decimate_preprocess(progress, config)

        # Collect batch sessions for pack/upload/clean
        batch_sessions = [
            sid for sid in all_session_ids
            if progress["sessions"].get(sid, {}).get("status")
            in ("processed", "packed", "uploaded", "verified")
        ]

        if not batch_sessions:
            logger.info("No sessions ready for packing.")
            continue

        statuses = {
            progress["sessions"][sid]["status"] for sid in batch_sessions
        }

        # Pack
        if "processed" in statuses:
            to_pack = [
                sid for sid in batch_sessions
                if progress["sessions"][sid]["status"] == "processed"
            ]
            progress = pack_shards(to_pack, progress, config)
            if any(
                progress["sessions"][sid]["status"] != "packed"
                for sid in to_pack
            ):
                logger.warning("Packing incomplete. Stopping this batch.")
                break

        # Refresh batch list
        batch_sessions = [
            sid for sid in all_session_ids
            if progress["sessions"].get(sid, {}).get("status")
            in ("packed", "uploaded", "verified")
        ]

        # Upload
        packed = [
            sid for sid in batch_sessions
            if progress["sessions"][sid]["status"] == "packed"
        ]
        if packed:
            progress = upload_shards(batch_sessions, progress, config)
            if any(
                progress["sessions"][sid]["status"] != "uploaded"
                for sid in packed
            ):
                logger.warning("Upload incomplete. Will retry on next run.")
                break

        # Verify
        uploaded = [
            sid for sid in batch_sessions
            if progress["sessions"][sid]["status"] == "uploaded"
        ]
        if uploaded:
            progress = verify_shards(batch_sessions, progress, config)
            if any(
                progress["sessions"][sid]["status"] != "verified"
                for sid in uploaded
            ):
                logger.warning("Verification failed. Will retry on next run.")
                break

        # Cleanup
        verified = [
            sid for sid in batch_sessions
            if progress["sessions"][sid]["status"] == "verified"
        ]
        if verified:
            progress = cleanup_batch(batch_sessions, progress, config)


def parse_args():
    parser = argparse.ArgumentParser(
        description="SANPO streaming data pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to pipeline config YAML (default: configs/pipeline_config.yaml)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()

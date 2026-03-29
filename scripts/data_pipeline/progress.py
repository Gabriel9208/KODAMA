"""Progress file helpers (atomic write) and shared utilities."""

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _available_disk_gb(path: Path) -> float:
    """Return available disk space in GB for the drive containing *path*."""
    usage = shutil.disk_usage(path.anchor or path)
    return usage.free / (1024 ** 3)


def _load_decimation_config(config: dict) -> dict:
    """Read decimation parameters from yaml."""
    dec_path = config["paths"]["decimation_config_file"]
    with open(dec_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["decimation"]


def _validate_decimation_config(progress: dict, config: dict) -> None:
    """Abort if decimation config changed since the progress file was created."""
    import sys

    current = _load_decimation_config(config)
    saved = progress.get("decimation_config", {})
    if saved and saved != current:
        logger.error(
            "Decimation config mismatch!\n"
            f"  Progress file: {saved}\n"
            f"  Current yaml:  {current}\n"
            "Delete pipeline_progress.json manually to proceed with new parameters."
        )
        sys.exit(1)


def load_progress(config: dict) -> dict:
    """Load pipeline_progress.json. Handle .tmp recovery."""
    progress_file = Path(config["paths"]["progress_file"])
    tmp_path = progress_file.with_suffix(".json.tmp")

    if tmp_path.exists():
        logger.warning("Found incomplete .tmp progress file — discarding it, using last good version.")
        tmp_path.unlink()

    if progress_file.exists():
        with open(progress_file, "r", encoding="utf-8") as f:
            return json.load(f)

    # Initialize new progress
    decimation_config = _load_decimation_config(config)
    return {
        "decimation_config": decimation_config,
        "sessions": {},
    }


def save_progress(progress: dict, config: dict) -> None:
    """Atomic write: write to .tmp then rename."""
    progress_file = Path(config["paths"]["progress_file"])
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = progress_file.with_suffix(".json.tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)

    os.replace(tmp_path, progress_file)


def load_session_ids(config: dict) -> list[str]:
    """Read session IDs from the split file."""
    session_ids_file = config["paths"]["session_ids_file"]
    with open(session_ids_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

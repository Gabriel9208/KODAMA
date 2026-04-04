"""Pipeline configuration loader."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_config_cache: dict | None = None


def load_pipeline_config(config_path: Path | None = None) -> dict:
    """
    Load pipeline_config.yaml and resolve all relative paths to absolute.

    Caches the result — subsequent calls return the same dict.
    """
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "pipeline_config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolve relative paths to absolute
    for key, value in cfg.get("paths", {}).items():
        cfg["paths"][key] = str(PROJECT_ROOT / value)

    _config_cache = cfg
    return cfg


def reset_config_cache() -> None:
    """Clear the config cache (for testing)."""
    global _config_cache
    _config_cache = None

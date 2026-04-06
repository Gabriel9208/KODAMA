"""CLI progress display helpers (pure functions)."""


def format_batch_range(batch_index: int, batch_size: int, total_sessions: int) -> str:
    """Format batch session range string. e.g. '1~4 / 560'."""
    start = batch_index * batch_size + 1
    end = min((batch_index + 1) * batch_size, total_sessions)
    return f"{start}~{end} / {total_sessions}"


def format_resume_summary(progress: dict, total_sessions: int, split: str = "train") -> str:
    """Format resume summary string. e.g. '[train] Resuming pipeline: 24 / 560 sessions completed, 3 skipped'."""
    split_sessions = progress.get("sessions", {}).get(split, {})
    cleaned = sum(1 for s in split_sessions.values() if s.get("status") == "cleaned")
    skipped = sum(1 for s in split_sessions.values() if s.get("status") == "skipped")
    return f"[{split}] Resuming pipeline: {cleaned} / {total_sessions} sessions completed, {skipped} skipped"

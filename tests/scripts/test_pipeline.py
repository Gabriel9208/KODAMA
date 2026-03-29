"""
Unit tests for pipeline_pack_upload.py
Scope: Decimation logic, validation logic, error messages, CLI progress display.
"""

import os
import sys
from pathlib import Path

import pytest

# Add scripts/ and src/ to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pipeline_pack_upload import (
    _decimate_indices,
    _extract_frame_id,
    _validate_camera,
    format_batch_range,
    format_resume_summary,
)


# =========================================================================
# 1. Decimation Sampling Logic
# =========================================================================


class TestDecimationSampling:
    """Test _decimate_indices: pure logic, no filesystem."""

    def test_1_1_basic_sampling_default_params(self):
        """interval=10, offset=0, 527 frames → indices [0,10,20,...,520], count=53."""
        indices = _decimate_indices(527, interval=10, offset=0)
        assert indices == list(range(0, 527, 10))
        assert len(indices) == 53

    def test_1_2_sampling_nonzero_offset(self):
        """interval=10, offset=3, 527 frames → indices [3,13,23,...,523], count=53."""
        indices = _decimate_indices(527, interval=10, offset=3)
        assert indices == list(range(3, 527, 10))
        assert len(indices) == 53

    def test_1_3_interval_larger_than_total(self):
        """interval=10, offset=0, 5 frames → [0], count=1."""
        indices = _decimate_indices(5, interval=10, offset=0)
        assert indices == [0]
        assert len(indices) == 1

    def test_1_4_offset_beyond_total(self):
        """offset=7, 5 frames → [], count=0. Must not crash."""
        indices = _decimate_indices(5, interval=10, offset=7)
        assert indices == []
        assert len(indices) == 0

    def test_1_5_determinism(self):
        """Same input + config = same output every time."""
        run1 = _decimate_indices(527, interval=10, offset=0)
        run2 = _decimate_indices(527, interval=10, offset=0)
        assert run1 == run2

    def test_1_6_interval_1_returns_all(self):
        """interval=1, offset=0, 10 frames → all 10."""
        indices = _decimate_indices(10, interval=1, offset=0)
        assert indices == list(range(10))
        assert len(indices) == 10


# =========================================================================
# 2. Three Data Types Stay Synchronized
# =========================================================================


class TestSynchronization:
    """Decimation indices applied to different file lists stay in sync."""

    def test_2_1_same_indices_all_types(self):
        """50 files per type, interval=10 → same 5 indices for all."""
        frames = [f"{i:06d}.png" for i in range(50)]
        segs = [f"{i:06d}.png" for i in range(50)]
        depths = [f"{i:06d}.float16.gz" for i in range(50)]

        indices = _decimate_indices(len(frames), interval=10, offset=0)
        assert indices == [0, 10, 20, 30, 40]

        selected_frames = [frames[i] for i in indices]
        selected_segs = [segs[i] for i in indices]
        selected_depths = [depths[i] for i in indices]

        # All three share the same numeric prefixes
        frame_ids = [_extract_frame_id(f) for f in selected_frames]
        seg_ids = [_extract_frame_id(f) for f in selected_segs]
        depth_ids = [_extract_frame_id(f) for f in selected_depths]
        assert frame_ids == seg_ids == depth_ids

    def test_2_2_different_prefixes_align_by_position(self):
        """Files with different naming but same sorted order align by index."""
        frames = ["frame_000.png", "frame_001.png", "frame_002.png"]
        segs = ["seg_000.png", "seg_001.png", "seg_002.png"]
        depths = ["depth_000.float16.gz", "depth_001.float16.gz", "depth_002.float16.gz"]

        indices = _decimate_indices(len(frames), interval=2, offset=0)
        assert indices == [0, 2]

        selected_frames = [frames[i] for i in indices]
        selected_segs = [segs[i] for i in indices]
        selected_depths = [depths[i] for i in indices]

        assert selected_frames == ["frame_000.png", "frame_002.png"]
        assert selected_segs == ["seg_000.png", "seg_002.png"]
        assert selected_depths == ["depth_000.float16.gz", "depth_002.float16.gz"]


# =========================================================================
# Helpers for validation tests
# =========================================================================


def _create_camera_dir(
    base: Path,
    camera: str,
    frame_ids: list[str] | None = None,
    seg_ids: list[str] | None = None,
    depth_ids: list[str] | None = None,
    skip_seg_dir: bool = False,
    skip_depth_dir: bool = False,
    empty_seg_dir: bool = False,
    empty_depth_dir: bool = False,
):
    """Create a mock camera directory structure for validation tests."""
    cam_base = base / camera / "left"

    # video_frames
    if frame_ids is not None:
        vf_dir = cam_base / "video_frames"
        vf_dir.mkdir(parents=True, exist_ok=True)
        for fid in frame_ids:
            (vf_dir / f"{fid}.png").touch()

    # segmentation_masks
    if not skip_seg_dir:
        seg_dir = cam_base / "segmentation_masks"
        seg_dir.mkdir(parents=True, exist_ok=True)
        if not empty_seg_dir and seg_ids is not None:
            for sid in seg_ids:
                (seg_dir / f"{sid}.png").touch()

    # depth_maps
    if not skip_depth_dir:
        depth_dir = cam_base / "depth_maps"
        depth_dir.mkdir(parents=True, exist_ok=True)
        if not empty_depth_dir and depth_ids is not None:
            for did in depth_ids:
                (depth_dir / f"{did}.float16.gz").touch()


# =========================================================================
# 3. Pre-Decimation Validation Logic
# =========================================================================


class TestValidation:
    """Test _validate_camera with mock directory structures."""

    def _ids(self, n: int) -> list[str]:
        return [f"{i:06d}" for i in range(n)]

    def test_3_1_all_present_counts_match(self, tmp_path):
        """50 files in each dir → PASS."""
        ids = self._ids(50)
        _create_camera_dir(tmp_path, "camera_chest", frame_ids=ids, seg_ids=ids, depth_ids=ids)
        passed, reason, count = _validate_camera(tmp_path, "camera_chest")
        assert passed is True
        assert count == 50

    def test_3_2_seg_dir_missing(self, tmp_path):
        """segmentation_masks dir does not exist → FAIL with 'missing'."""
        ids = self._ids(50)
        _create_camera_dir(tmp_path, "camera_chest", frame_ids=ids, seg_ids=ids, depth_ids=ids, skip_seg_dir=True)
        passed, reason, count = _validate_camera(tmp_path, "camera_chest")
        assert passed is False
        assert "segmentation_masks" in reason
        assert "missing" in reason

    def test_3_3_seg_dir_empty(self, tmp_path):
        """segmentation_masks dir exists but empty → FAIL with 'empty'."""
        ids = self._ids(50)
        _create_camera_dir(tmp_path, "camera_chest", frame_ids=ids, seg_ids=ids, depth_ids=ids, empty_seg_dir=True)
        passed, reason, count = _validate_camera(tmp_path, "camera_chest")
        assert passed is False
        assert "segmentation_masks" in reason
        assert "empty" in reason

    def test_3_4_depth_dir_missing(self, tmp_path):
        """depth_maps dir does not exist → FAIL with 'missing'."""
        ids = self._ids(50)
        _create_camera_dir(tmp_path, "camera_chest", frame_ids=ids, seg_ids=ids, depth_ids=ids, skip_depth_dir=True)
        passed, reason, count = _validate_camera(tmp_path, "camera_chest")
        assert passed is False
        assert "depth_maps" in reason
        assert "missing" in reason

    def test_3_5_seg_count_mismatch(self, tmp_path):
        """seg has 30 files, others have 50 → FAIL with counts in reason."""
        ids_50 = self._ids(50)
        ids_30 = self._ids(30)
        _create_camera_dir(tmp_path, "camera_chest", frame_ids=ids_50, seg_ids=ids_30, depth_ids=ids_50)
        passed, reason, count = _validate_camera(tmp_path, "camera_chest")
        assert passed is False
        assert "file count mismatch" in reason
        assert "50" in reason
        assert "30" in reason

    def test_3_6_depth_count_mismatch(self, tmp_path):
        """depth has 45 files, others have 50 → FAIL with counts in reason."""
        ids_50 = self._ids(50)
        ids_45 = self._ids(45)
        _create_camera_dir(tmp_path, "camera_chest", frame_ids=ids_50, seg_ids=ids_50, depth_ids=ids_45)
        passed, reason, count = _validate_camera(tmp_path, "camera_chest")
        assert passed is False
        assert "file count mismatch" in reason
        assert "50" in reason
        assert "45" in reason

    def test_3_7_frame_id_misalignment(self, tmp_path):
        """seg has 000003 instead of 000002 → FAIL with 'not aligned'."""
        frame_ids = ["000000", "000001", "000002"]
        seg_ids = ["000000", "000001", "000003"]
        depth_ids = ["000000", "000001", "000002"]
        _create_camera_dir(tmp_path, "camera_chest", frame_ids=frame_ids, seg_ids=seg_ids, depth_ids=depth_ids)
        passed, reason, count = _validate_camera(tmp_path, "camera_chest")
        assert passed is False
        assert "not aligned" in reason

    def test_3_8_non_contiguous_ids_all_match(self, tmp_path):
        """Non-contiguous IDs (000000, 000005, 000010) all match → PASS."""
        ids = ["000000", "000005", "000010"]
        _create_camera_dir(tmp_path, "camera_chest", frame_ids=ids, seg_ids=ids, depth_ids=ids)
        passed, reason, count = _validate_camera(tmp_path, "camera_chest")
        assert passed is True
        assert count == 3


# =========================================================================
# 4. Error Messages
# =========================================================================


class TestErrorMessages:
    """Validation failures include actionable detail."""

    def _ids(self, n: int) -> list[str]:
        return [f"{i:06d}" for i in range(n)]

    def test_4_1_failure_messages_contain_camera_name(self, tmp_path):
        """All failure reasons include the camera name."""
        ids = self._ids(50)
        # seg missing
        _create_camera_dir(tmp_path / "a", "camera_chest", frame_ids=ids, seg_ids=ids, depth_ids=ids, skip_seg_dir=True)
        _, reason, _ = _validate_camera(tmp_path / "a", "camera_chest")
        assert "camera_chest" in reason

        # depth missing
        _create_camera_dir(tmp_path / "b", "camera_head", frame_ids=ids, seg_ids=ids, depth_ids=ids, skip_depth_dir=True)
        _, reason, _ = _validate_camera(tmp_path / "b", "camera_head")
        assert "camera_head" in reason

    def test_4_2_decimation_empty_frames(self):
        """Empty frame list → empty indices, no crash."""
        indices = _decimate_indices(0, interval=10, offset=0)
        assert indices == []


# =========================================================================
# 5. CLI Progress Display (Phase 5)
# =========================================================================


class TestCLIProgress:
    """Test format_batch_range and format_resume_summary."""

    def test_5_1_batch_range_first_batch(self):
        """batch_index=0, batch_size=4, total=560 → '1~4 / 560'."""
        result = format_batch_range(0, 4, 560)
        assert result == "1~4 / 560"

    def test_5_1_batch_range_third_batch(self):
        """batch_index=2, batch_size=4, total=560 → '9~12 / 560'."""
        result = format_batch_range(2, 4, 560)
        assert result == "9~12 / 560"

    def test_5_2_last_batch_remainder(self):
        """Last batch clamps to total, not beyond."""
        # 563 sessions / batch_size=8 → batch 70 starts at 561
        result = format_batch_range(70, 8, 563)
        assert "561~563 / 563" == result

    def test_5_3_resume_summary(self):
        """24 cleaned + 3 skipped + rest pending."""
        progress = {
            "sessions": {}
        }
        for i in range(24):
            progress["sessions"][f"s{i}"] = {"status": "cleaned"}
        for i in range(24, 27):
            progress["sessions"][f"s{i}"] = {"status": "skipped"}
        for i in range(27, 560):
            progress["sessions"][f"s{i}"] = {"status": "pending"}

        result = format_resume_summary(progress, 560)
        assert "24 / 560 sessions completed" in result
        assert "3 skipped" in result

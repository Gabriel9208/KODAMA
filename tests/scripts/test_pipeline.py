"""
Unit tests for data_pipeline package.
Scope: Decimation logic, validation logic, error messages, CLI progress display.
"""

import sys
from pathlib import Path
from unittest.mock import patch

# Add scripts/ and src/ to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_pipeline.decimate import _decimate_indices  # noqa: E402
from data_pipeline.display import format_batch_range, format_resume_summary  # noqa: E402
from data_pipeline.download import download_session  # noqa: E402
from data_pipeline.progress import clear_stale_pending_sessions  # noqa: E402
from data_pipeline.validate import (  # noqa: E402
    _extract_frame_id,
    _list_local_files,
    _validate_camera,
    validate_session,
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


def _call_validate(base, camera):
    """Helper: call _validate_camera with local listing for a mock directory."""
    cam_base = str(base / camera / "left")
    return _validate_camera(
        camera,
        _list_local_files,
        f"{cam_base}/video_frames",
        f"{cam_base}/segmentation_masks",
        f"{cam_base}/depth_maps",
    )


class TestValidation:
    """Test _validate_camera with mock directory structures."""

    def _ids(self, n: int) -> list[str]:
        return [f"{i:06d}" for i in range(n)]

    def test_3_1_all_present_counts_match(self, tmp_path):
        """50 files in each dir → PASS."""
        ids = self._ids(50)
        _create_camera_dir(tmp_path, "camera_chest", frame_ids=ids, seg_ids=ids, depth_ids=ids)
        passed, reason, count = _call_validate(tmp_path, "camera_chest")
        assert passed is True
        assert count == 50

    def test_3_2_seg_dir_missing(self, tmp_path):
        """segmentation_masks dir does not exist → FAIL with 'missing'."""
        ids = self._ids(50)
        _create_camera_dir(tmp_path, "camera_chest", frame_ids=ids, seg_ids=ids, depth_ids=ids, skip_seg_dir=True)
        passed, reason, count = _call_validate(tmp_path, "camera_chest")
        assert passed is False
        assert "segmentation_masks" in reason
        assert "missing" in reason

    def test_3_3_seg_dir_empty(self, tmp_path):
        """segmentation_masks dir exists but empty → FAIL with 'empty'."""
        ids = self._ids(50)
        _create_camera_dir(tmp_path, "camera_chest", frame_ids=ids, seg_ids=ids, depth_ids=ids, empty_seg_dir=True)
        passed, reason, count = _call_validate(tmp_path, "camera_chest")
        assert passed is False
        assert "segmentation_masks" in reason
        assert "empty" in reason

    def test_3_4_depth_dir_missing(self, tmp_path):
        """depth_maps dir does not exist → FAIL with 'missing'."""
        ids = self._ids(50)
        _create_camera_dir(tmp_path, "camera_chest", frame_ids=ids, seg_ids=ids, depth_ids=ids, skip_depth_dir=True)
        passed, reason, count = _call_validate(tmp_path, "camera_chest")
        assert passed is False
        assert "depth_maps" in reason
        assert "missing" in reason

    def test_3_5_seg_count_mismatch(self, tmp_path):
        """seg has 30 files, others have 50 → FAIL with counts in reason."""
        ids_50 = self._ids(50)
        ids_30 = self._ids(30)
        _create_camera_dir(tmp_path, "camera_chest", frame_ids=ids_50, seg_ids=ids_30, depth_ids=ids_50)
        passed, reason, count = _call_validate(tmp_path, "camera_chest")
        assert passed is False
        assert "file count mismatch" in reason
        assert "50" in reason
        assert "30" in reason

    def test_3_6_depth_count_mismatch(self, tmp_path):
        """depth has 45 files, others have 50 → FAIL with counts in reason."""
        ids_50 = self._ids(50)
        ids_45 = self._ids(45)
        _create_camera_dir(tmp_path, "camera_chest", frame_ids=ids_50, seg_ids=ids_50, depth_ids=ids_45)
        passed, reason, count = _call_validate(tmp_path, "camera_chest")
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
        passed, reason, count = _call_validate(tmp_path, "camera_chest")
        assert passed is False
        assert "not aligned" in reason

    def test_3_8_non_contiguous_ids_all_match(self, tmp_path):
        """Non-contiguous IDs (000000, 000005, 000010) all match → PASS."""
        ids = ["000000", "000005", "000010"]
        _create_camera_dir(tmp_path, "camera_chest", frame_ids=ids, seg_ids=ids, depth_ids=ids)
        passed, reason, count = _call_validate(tmp_path, "camera_chest")
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
        _, reason, _ = _call_validate(tmp_path / "a", "camera_chest")
        assert "camera_chest" in reason

        # depth missing
        _create_camera_dir(tmp_path / "b", "camera_head", frame_ids=ids, seg_ids=ids, depth_ids=ids, skip_depth_dir=True)
        _, reason, _ = _call_validate(tmp_path / "b", "camera_head")
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
        train_sessions = {}
        for i in range(24):
            train_sessions[f"s{i}"] = {"status": "cleaned"}
        for i in range(24, 27):
            train_sessions[f"s{i}"] = {"status": "skipped"}
        for i in range(27, 560):
            train_sessions[f"s{i}"] = {"status": "pending"}
        progress = {"sessions": {"train": train_sessions, "test": {}}}

        result = format_resume_summary(progress, 560)
        assert "24 / 560 sessions completed" in result
        assert "3 skipped" in result


# =========================================================================
# 6. Clear Stale Pending Sessions
# =========================================================================


class TestClearStalePending:
    """Test clear_stale_pending_sessions: Feature 1."""

    def _config(self):
        return {"split": "train"}

    def _progress(self, train_sessions: dict) -> dict:
        return {"sessions": {"train": train_sessions, "test": {}}}

    def test_6_1_clean_pending_not_touched(self):
        """Pending with only 'status' key → unchanged, returns 0."""
        progress = self._progress({"s1": {"status": "pending"}})
        count = clear_stale_pending_sessions(progress, self._config())
        assert count == 0
        assert progress["sessions"]["train"]["s1"] == {"status": "pending"}

    def test_6_2_stale_pending_extra_keys_cleared(self):
        """Pending with extra keys → stripped to {status: pending}, returns 1."""
        progress = self._progress({
            "s1": {
                "status": "pending",
                "downloaded_at": "2026-04-01T10:00:00+00:00",
                "skip_reason": "old reason",
            }
        })
        count = clear_stale_pending_sessions(progress, self._config())
        assert count == 1
        assert progress["sessions"]["train"]["s1"] == {"status": "pending"}

    def test_6_3_non_pending_sessions_not_touched(self):
        """downloaded/validated sessions are never modified."""
        progress = self._progress({
            "s1": {"status": "downloaded", "downloaded_at": "2026-04-01T10:00:00+00:00"},
            "s2": {"status": "validated", "validated_at": "2026-04-01T10:05:00+00:00"},
        })
        count = clear_stale_pending_sessions(progress, self._config())
        assert count == 0
        assert "downloaded_at" in progress["sessions"]["train"]["s1"]
        assert "validated_at" in progress["sessions"]["train"]["s2"]

    def test_6_4_multiple_stale_pending_sessions(self):
        """Three stale pending sessions → all cleaned, returns 3."""
        progress = self._progress({
            f"s{i}": {"status": "pending", "downloaded_at": "old"} for i in range(3)
        })
        count = clear_stale_pending_sessions(progress, self._config())
        assert count == 3
        for i in range(3):
            assert progress["sessions"]["train"][f"s{i}"] == {"status": "pending"}

    def test_6_5_mixed_sessions(self):
        """Mix of stale pending, clean pending, and downloaded → only stale pending cleared."""
        progress = self._progress({
            "stale": {"status": "pending", "downloaded_at": "old"},
            "clean": {"status": "pending"},
            "done": {"status": "downloaded", "downloaded_at": "2026-04-01T10:00:00+00:00"},
        })
        count = clear_stale_pending_sessions(progress, self._config())
        assert count == 1
        assert progress["sessions"]["train"]["stale"] == {"status": "pending"}
        assert progress["sessions"]["train"]["clean"] == {"status": "pending"}
        assert "downloaded_at" in progress["sessions"]["train"]["done"]


# =========================================================================
# 7. download_session() — Dict Return, Per-Camera Skip Reasons, Depth Handling
# =========================================================================


class TestDownloadSession:
    """Test download_session: Features 2 + 3. Mocks _gcloud_ls, _gcloud_cp, and filesystem."""

    def _config(self, tmp_path: Path) -> dict:
        return {
            "remote": {"gcs_bucket": "gs://test-bucket"},
            "paths": {"raw_dir": str(tmp_path / "raw")},
            "split": "train",
        }

    def _progress(self) -> dict:
        return {"sessions": {"train": {}, "test": {}}}

    def _ls_factory(self, present: set[str]):
        """Returns a _gcloud_ls mock: True if any key in present is a substring of path."""
        def ls(path):
            return any(key in path for key in present)
        return ls

    def test_7_1_clean_chest_only_download(self, tmp_path):
        """All data present, no camera_head → status=downloaded, no extra fields."""
        def ls(path):
            return "camera_head/left/" not in path  # camera_head dir absent
        with patch("data_pipeline.download._gcloud_ls", side_effect=ls), \
             patch("data_pipeline.download._gcloud_cp", return_value=True), \
             patch("data_pipeline.download._integrity_check_post_download", return_value=True):
            result = download_session("sess1", self._progress(), self._config(tmp_path))
        assert result["status"] == "downloaded"
        assert "downloaded_at" in result
        assert "camera_skip_reasons" not in result
        assert "cameras_missing_depth" not in result
        assert "skip_reason" not in result

    def test_7_2_camera_head_missing_video_frames(self, tmp_path):
        """camera_head/left/video_frames absent → camera_skip_reasons + skip_reason set."""
        def ls(path):
            if "camera_head/left/video_frames" in path:
                return False
            return True
        with patch("data_pipeline.download._gcloud_ls", side_effect=ls), \
             patch("data_pipeline.download._gcloud_cp", return_value=True), \
             patch("data_pipeline.download._integrity_check_post_download", return_value=True):
            result = download_session("sess1", self._progress(), self._config(tmp_path))
        assert result["status"] == "downloaded"
        assert result["camera_skip_reasons"] == {"camera_head": "video_frames missing on GCS"}
        assert result["skip_reason"] == "camera_head: video_frames missing on GCS"
        assert "cameras_missing_depth" not in result

    def test_7_3_camera_head_missing_segmentation_masks(self, tmp_path):
        """camera_head/left/segmentation_masks absent → camera_skip_reasons + skip_reason set."""
        def ls(path):
            if "camera_head/left/segmentation_masks" in path:
                return False
            return True
        with patch("data_pipeline.download._gcloud_ls", side_effect=ls), \
             patch("data_pipeline.download._gcloud_cp", return_value=True), \
             patch("data_pipeline.download._integrity_check_post_download", return_value=True):
            result = download_session("sess1", self._progress(), self._config(tmp_path))
        assert result["status"] == "downloaded"
        assert result["camera_skip_reasons"] == {"camera_head": "segmentation_masks missing on GCS"}
        assert "skip_reason" in result

    def test_7_4_camera_head_missing_depth_maps(self, tmp_path):
        """camera_head/left/depth_maps absent → cameras_missing_depth set, status=downloaded."""
        def ls(path):
            if "camera_head/left/depth_maps" in path:
                return False
            return True
        cp_calls = []
        def cp(src, dst):
            cp_calls.append(src)
            return True
        with patch("data_pipeline.download._gcloud_ls", side_effect=ls), \
             patch("data_pipeline.download._gcloud_cp", side_effect=cp), \
             patch("data_pipeline.download._integrity_check_post_download", return_value=True):
            result = download_session("sess1", self._progress(), self._config(tmp_path))
        assert result["status"] == "downloaded"
        assert result["cameras_missing_depth"] == ["camera_head"]
        assert "camera_skip_reasons" not in result
        # depth_maps must NOT have been downloaded for camera_head
        assert not any("camera_head/left/depth_maps" in c for c in cp_calls)
        # depth_maps IS downloaded for camera_chest
        assert any("camera_chest/left/depth_maps" in c for c in cp_calls)

    def test_7_5_all_cameras_missing_required_types(self, tmp_path):
        """Both cameras missing video_frames → session-level skip with camera_skip_reasons."""
        def ls(path):
            if "camera_head/left/" == path.split("test-bucket/sess1/")[-1]:
                return True  # camera_head dir exists
            if "video_frames" in path:
                return False
            return True
        with patch("data_pipeline.download._gcloud_ls", side_effect=ls), \
             patch("data_pipeline.download._gcloud_cp", return_value=True):
            result = download_session("sess1", self._progress(), self._config(tmp_path))
        assert result["status"] == "skipped"
        assert "skip_reason" in result
        assert "skipped_at" in result
        assert "camera_chest" in result["camera_skip_reasons"]
        assert "camera_head" in result["camera_skip_reasons"]

    def test_7_6_integrity_check_fails(self, tmp_path):
        """All GCS present, gcloud_cp succeeds, but integrity check fails → skipped."""
        with patch("data_pipeline.download._gcloud_ls", return_value=True), \
             patch("data_pipeline.download._gcloud_cp", return_value=True), \
             patch("data_pipeline.download._integrity_check_post_download", return_value=False):
            result = download_session("sess1", self._progress(), self._config(tmp_path))
        assert result["status"] == "skipped"
        assert "integrity check failed" in result["skip_reason"]
        assert "skipped_at" in result

    def test_7_7_gcloud_cp_failure(self, tmp_path):
        """gcloud_cp fails → {"status": "error"}."""
        with patch("data_pipeline.download._gcloud_ls", return_value=True), \
             patch("data_pipeline.download._gcloud_cp", return_value=False):
            result = download_session("sess1", self._progress(), self._config(tmp_path))
        assert result == {"status": "error"}

    def test_7_8_camera_skip_reasons_absent_when_empty(self, tmp_path):
        """Clean download → camera_skip_reasons and skip_reason not present in result."""
        def ls(path):
            return "camera_head/left/" not in path
        with patch("data_pipeline.download._gcloud_ls", side_effect=ls), \
             patch("data_pipeline.download._gcloud_cp", return_value=True), \
             patch("data_pipeline.download._integrity_check_post_download", return_value=True):
            result = download_session("sess1", self._progress(), self._config(tmp_path))
        assert "camera_skip_reasons" not in result
        assert "skip_reason" not in result

    def test_7_9_cameras_missing_depth_absent_when_empty(self, tmp_path):
        """All depth maps present → cameras_missing_depth key not present in result."""
        def ls(path):
            return "camera_head/left/" not in path
        with patch("data_pipeline.download._gcloud_ls", side_effect=ls), \
             patch("data_pipeline.download._gcloud_cp", return_value=True), \
             patch("data_pipeline.download._integrity_check_post_download", return_value=True):
            result = download_session("sess1", self._progress(), self._config(tmp_path))
        assert "cameras_missing_depth" not in result

    def test_7_10_both_skip_reasons_and_missing_depth(self, tmp_path):
        """camera_head skipped (missing seg), camera_chest missing depth → both fields present."""
        def ls(path):
            if "camera_head/left/segmentation_masks" in path:
                return False
            if "camera_chest/left/depth_maps" in path:
                return False
            return True
        with patch("data_pipeline.download._gcloud_ls", side_effect=ls), \
             patch("data_pipeline.download._gcloud_cp", return_value=True), \
             patch("data_pipeline.download._integrity_check_post_download", return_value=True):
            result = download_session("sess1", self._progress(), self._config(tmp_path))
        assert result["status"] == "downloaded"
        assert result["camera_skip_reasons"] == {"camera_head": "segmentation_masks missing on GCS"}
        assert result["cameras_missing_depth"] == ["camera_chest"]
        assert "skip_reason" in result

    def test_7_11_chest_skipped_head_valid_status_downloaded(self, tmp_path):
        """camera_chest missing seg, camera_head fully present → status=downloaded (not skipped)."""
        def ls(path):
            if "camera_chest/left/segmentation_masks" in path:
                return False
            return True
        with patch("data_pipeline.download._gcloud_ls", side_effect=ls), \
             patch("data_pipeline.download._gcloud_cp", return_value=True), \
             patch("data_pipeline.download._integrity_check_post_download", return_value=True):
            result = download_session("sess1", self._progress(), self._config(tmp_path))
        assert result["status"] == "downloaded"
        assert result["camera_skip_reasons"] == {"camera_chest": "segmentation_masks missing on GCS"}
        assert result["skip_reason"] == "camera_chest: segmentation_masks missing on GCS"


# =========================================================================
# 8. validate_session() — Partial Camera Skip Reasons
# =========================================================================


class TestValidateSessionPartialSkip:
    """Test that validate_session writes skip_reason even when session is validated."""

    def _config(self, tmp_path: Path) -> dict:
        return {
            "paths": {
                "raw_dir": str(tmp_path / "raw"),
                "progress_file": str(tmp_path / "progress.json"),
            },
            "preprocess": {"cameras": ["camera_chest", "camera_head"]},
            "split": "train",
        }

    def _progress(self, session_id: str) -> dict:
        return {"sessions": {"train": {session_id: {"status": "downloaded"}}, "test": {}}}

    def _ids(self, n: int) -> list[str]:
        return [f"{i:06d}" for i in range(n)]

    def test_8_1_all_cameras_pass_no_skip_reason(self, tmp_path):
        """Both cameras pass validation → skip_reason and camera_skip_reasons absent."""
        session_id = "sess1"
        raw = tmp_path / "raw" / session_id
        ids = self._ids(10)
        _create_camera_dir(raw, "camera_chest", frame_ids=ids, seg_ids=ids, depth_ids=ids)
        _create_camera_dir(raw, "camera_head", frame_ids=ids, seg_ids=ids, depth_ids=ids)

        progress = self._progress(session_id)
        validate_session(session_id, progress, self._config(tmp_path))

        info = progress["sessions"]["train"][session_id]
        assert info["status"] == "validated"
        assert "skip_reason" not in info
        assert "camera_skip_reasons" not in info

    def test_8_2_one_camera_fails_session_validated_with_skip_reason(self, tmp_path):
        """camera_head has count mismatch → status=validated, skip_reason + camera_skip_reasons set."""
        session_id = "sess1"
        raw = tmp_path / "raw" / session_id
        ids_10 = self._ids(10)
        ids_5 = self._ids(5)
        _create_camera_dir(raw, "camera_chest", frame_ids=ids_10, seg_ids=ids_10, depth_ids=ids_10)
        # camera_head: seg has fewer files → count mismatch
        _create_camera_dir(raw, "camera_head", frame_ids=ids_10, seg_ids=ids_5, depth_ids=ids_10)

        progress = self._progress(session_id)
        validate_session(session_id, progress, self._config(tmp_path))

        info = progress["sessions"]["train"][session_id]
        assert info["status"] == "validated"
        assert info["valid_cameras"] == ["camera_chest"]
        assert "skip_reason" in info
        assert "camera_head" in info["skip_reason"]
        assert "camera_skip_reasons" in info
        assert "camera_head" in info["camera_skip_reasons"]

    def test_8_3_all_cameras_fail_session_skipped(self, tmp_path):
        """Both cameras fail → status=skipped, skip_reason set (existing behavior unchanged)."""
        session_id = "sess1"
        raw = tmp_path / "raw" / session_id
        ids = self._ids(10)
        # Both cameras: seg dir missing
        _create_camera_dir(raw, "camera_chest", frame_ids=ids, seg_ids=ids, depth_ids=ids, skip_seg_dir=True)
        _create_camera_dir(raw, "camera_head", frame_ids=ids, seg_ids=ids, depth_ids=ids, skip_seg_dir=True)

        progress = self._progress(session_id)
        validate_session(session_id, progress, self._config(tmp_path))

        info = progress["sessions"]["train"][session_id]
        assert info["status"] == "skipped"
        assert "skip_reason" in info
        assert "skipped_at" in info

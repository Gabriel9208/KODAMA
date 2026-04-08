"""
Integration tests: FeatureExtractor (YOLO26n-seg) → SementicDecoder pipeline.

Requires the real model weights at model_info/yolo26n-seg.pt.
Marked `integration` so they can be skipped in CI with: pytest -m "not integration"
"""
import pytest
import torch
from pathlib import Path

MODEL_PATH = Path("model_info/yolo26n-seg.pt")

# FPN layer indices in YOLO26n backbone
P3_LAYER, P4_LAYER, P5_LAYER = 16, 19, 22

# Channel widths confirmed by probing the live model
P3_CH, P4_CH, P5_CH = 64, 128, 256
NUM_CLASSES = 15

# Input spatial dimensions (H × W) used across all tests
INPUT_H, INPUT_W = 480, 640


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def yolo_model():
    """Load the YOLO26n-seg model once per test module."""
    pytest.importorskip("ultralytics", reason="ultralytics not installed")
    if not MODEL_PATH.exists():
        pytest.skip(f"Model weights not found at {MODEL_PATH}")
    from ultralytics import YOLO
    return YOLO(str(MODEL_PATH))


@pytest.fixture(scope="module")
def feature_extractor(yolo_model):
    """FeatureExtractor wrapping the YOLO model, shared across tests."""
    from src.models.feature_extractor import FeatureExtractor
    fe = FeatureExtractor(yolo_model, layers=[P3_LAYER, P4_LAYER, P5_LAYER])
    yield fe
    del fe  # triggers handle cleanup via __del__


@pytest.fixture(scope="module")
def fpn_features(feature_extractor):
    """Run one forward pass (CPU input — YOLO places it on its device) and cache."""
    x = torch.zeros(1, 3, INPUT_H, INPUT_W)
    with torch.no_grad():
        return feature_extractor(x)


@pytest.fixture(scope="module")
def device(fpn_features):
    """Actual device the YOLO model uses, detected from extracted features."""
    return fpn_features[P3_LAYER].device


@pytest.fixture(scope="module")
def decoder(device):
    from src.models.semantic_decoder import SementicDecoder
    return SementicDecoder(
        num_classes=NUM_CLASSES,
        p3_channel=P3_CH,
        p4_channel=P4_CH,
        p5_channel=P5_CH,
    ).to(device)


# ---------------------------------------------------------------------------
# FeatureExtractor tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestFeatureExtractor:
    """Verify the FeatureExtractor produces the expected FPN tensors."""

    def test_returns_all_three_layers(self, fpn_features):
        assert set(fpn_features.keys()) == {P3_LAYER, P4_LAYER, P5_LAYER}

    def test_p3_shape(self, fpn_features):
        # Layer 16 → 1/8 of input: 480/8=60, 640/8=80
        assert fpn_features[P3_LAYER].shape == (1, P3_CH, INPUT_H // 8, INPUT_W // 8)

    def test_p4_shape(self, fpn_features):
        assert fpn_features[P4_LAYER].shape == (1, P4_CH, INPUT_H // 16, INPUT_W // 16)

    def test_p5_shape(self, fpn_features):
        assert fpn_features[P5_LAYER].shape == (1, P5_CH, INPUT_H // 32, INPUT_W // 32)

    def test_features_are_tensors(self, fpn_features):
        for feat in fpn_features.values():
            assert isinstance(feat, torch.Tensor)

    def test_features_no_nan(self, fpn_features):
        for feat in fpn_features.values():
            assert not torch.isnan(feat).any()

    def test_context_manager_removes_hooks(self, yolo_model):
        """__exit__ must remove all forward hooks."""
        from src.models.feature_extractor import FeatureExtractor
        with FeatureExtractor(yolo_model, [P3_LAYER]) as fe:
            assert len(fe.handles) == 1
        assert len(fe.handles) == 0

    def test_features_reset_between_calls(self, feature_extractor):
        """features dict must be fresh on each forward call."""
        x = torch.zeros(1, 3, INPUT_H, INPUT_W)
        with torch.no_grad():
            f1 = feature_extractor(x)
            f2 = feature_extractor(x)
        # Different dict objects, but same keys
        assert f1 is not f2
        assert f1.keys() == f2.keys()


# ---------------------------------------------------------------------------
# End-to-end pipeline tests: FeatureExtractor → SementicDecoder
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestFeatureExtractorToDecoder:
    """Full pipeline: YOLO FPN → SementicDecoder → segmentation logits."""

    @pytest.fixture(scope="class")
    def seg_output(self, fpn_features, decoder):
        p3 = fpn_features[P3_LAYER]
        p4 = fpn_features[P4_LAYER]
        p5 = fpn_features[P5_LAYER]
        with torch.no_grad():
            return decoder(p3, p4, p5)

    def test_output_shape(self, seg_output):
        assert seg_output.shape == (1, NUM_CLASSES, INPUT_H, INPUT_W)

    def test_output_channels_match_num_classes(self, seg_output):
        assert seg_output.shape[1] == NUM_CLASSES

    def test_output_spatial_matches_input(self, seg_output):
        assert seg_output.shape[2] == INPUT_H
        assert seg_output.shape[3] == INPUT_W

    def test_output_no_nan(self, seg_output):
        assert not torch.isnan(seg_output).any()

    def test_output_no_inf(self, seg_output):
        assert not torch.isinf(seg_output).any()

    def test_output_is_float(self, seg_output):
        assert seg_output.dtype == torch.float32

    def test_batch_size_propagated(self, feature_extractor, decoder):
        """Batch size > 1 flows end-to-end correctly."""
        B = 2
        x = torch.zeros(B, 3, INPUT_H, INPUT_W)
        with torch.no_grad():
            feats = feature_extractor(x)
            out = decoder(feats[P3_LAYER], feats[P4_LAYER], feats[P5_LAYER])
        assert out.shape == (B, NUM_CLASSES, INPUT_H, INPUT_W)

    def test_different_inputs_produce_different_outputs(self, feature_extractor, decoder):
        """Non-zero input should produce different logits than all-zeros."""
        x_zero = torch.zeros(1, 3, INPUT_H, INPUT_W)
        x_rand = torch.randn(1, 3, INPUT_H, INPUT_W)
        with torch.no_grad():
            f_zero = feature_extractor(x_zero)
            out_zero = decoder(f_zero[P3_LAYER], f_zero[P4_LAYER], f_zero[P5_LAYER])
            f_rand = feature_extractor(x_rand)
            out_rand = decoder(f_rand[P3_LAYER], f_rand[P4_LAYER], f_rand[P5_LAYER])
        assert not torch.allclose(out_zero, out_rand)

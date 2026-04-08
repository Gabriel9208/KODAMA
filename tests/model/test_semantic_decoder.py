import pytest
import torch
from model.semantic_decoder import UpsampleBlock, SementicDecoder


# ---------------------------------------------------------------------------
# UpsampleBlock
# ---------------------------------------------------------------------------

class TestUpsampleBlock:
    """Unit tests for the UpsampleBlock building block."""

    @pytest.fixture
    def block(self):
        # in_channels=32, out_channels=32 (must be divisible by 16 for GroupNorm)
        return UpsampleBlock(in_channels=32, out_channels=32)

    def test_output_shape_doubles_spatial(self, block):
        x = torch.randn(1, 32, 10, 10)
        out = block(x)
        assert out.shape == (1, 32, 20, 20)

    def test_channel_projection(self):
        block = UpsampleBlock(in_channels=64, out_channels=32)
        x = torch.randn(1, 64, 8, 8)
        out = block(x)
        assert out.shape[1] == 32

    def test_batch_dimension_preserved(self, block):
        x = torch.randn(4, 32, 6, 6)
        out = block(x)
        assert out.shape[0] == 4

    def test_forward_no_nan(self, block):
        x = torch.randn(1, 32, 5, 5)
        out = block(x)
        assert not torch.isnan(out).any()

    def test_is_nn_module(self, block):
        assert isinstance(block, torch.nn.Module)

    @pytest.mark.parametrize("H,W", [(4, 4), (8, 6), (20, 15)])
    def test_various_spatial_sizes(self, block, H, W):
        x = torch.randn(1, 32, H, W)
        out = block(x)
        assert out.shape == (1, 32, H * 2, W * 2)


# ---------------------------------------------------------------------------
# SementicDecoder — channel configuration matching YOLO FPN defaults
# ---------------------------------------------------------------------------

P3_CH, P4_CH, P5_CH = 64, 128, 256
NUM_CLASSES = 30

# SANPO YOLO input spatial dimensions
P3_H, P3_W = 80, 60
P4_H, P4_W = 40, 30
P5_H, P5_W = 20, 15


@pytest.fixture
def decoder():
    return SementicDecoder(
        num_classes=NUM_CLASSES,
        p3_channel=P3_CH,
        p4_channel=P4_CH,
        p5_channel=P5_CH,
    )


@pytest.fixture
def fpn_inputs():
    return (
        torch.randn(1, P3_CH, P3_H, P3_W),
        torch.randn(1, P4_CH, P4_H, P4_W),
        torch.randn(1, P5_CH, P5_H, P5_W),
    )


class TestSementicDecoderOutputShape:
    """Tests focused on output tensor dimensions."""

    def test_output_channels_equals_num_classes(self, decoder, fpn_inputs):
        out = decoder(*fpn_inputs)
        assert out.shape[1] == NUM_CLASSES

    def test_output_spatial_is_8x_p3(self, decoder, fpn_inputs):
        """Final upsample ×8 relative to P3 (80×60 → 640×480)."""
        out = decoder(*fpn_inputs)
        assert out.shape[2] == P3_H * 8
        assert out.shape[3] == P3_W * 8

    def test_output_shape_full(self, decoder, fpn_inputs):
        out = decoder(*fpn_inputs)
        assert out.shape == (1, NUM_CLASSES, 640, 480)

    def test_batch_size_preserved(self, decoder):
        B = 3
        p3 = torch.randn(B, P3_CH, P3_H, P3_W)
        p4 = torch.randn(B, P4_CH, P4_H, P4_W)
        p5 = torch.randn(B, P5_CH, P5_H, P5_W)
        out = decoder(p3, p4, p5)
        assert out.shape[0] == B


class TestSementicDecoderNumerics:
    """Numerical sanity checks."""

    def test_forward_no_nan(self, decoder, fpn_inputs):
        out = decoder(*fpn_inputs)
        assert not torch.isnan(out).any()

    def test_forward_no_inf(self, decoder, fpn_inputs):
        out = decoder(*fpn_inputs)
        assert not torch.isinf(out).any()

    def test_output_requires_grad(self, decoder, fpn_inputs):
        p3, p4, p5 = fpn_inputs
        p3.requires_grad_(True)
        out = decoder(p3, p4, p5)
        assert out.requires_grad

    def test_gradients_flow(self, decoder, fpn_inputs):
        p3, p4, p5 = fpn_inputs
        p3 = p3.requires_grad_(True)
        p4 = p4.requires_grad_(True)
        p5 = p5.requires_grad_(True)
        out = decoder(p3, p4, p5)
        out.sum().backward()
        assert p3.grad is not None
        assert p4.grad is not None
        assert p5.grad is not None


class TestSementicDecoderConstruction:
    """Tests for constructor behaviour and stored attributes."""

    def test_num_classes_stored(self, decoder):
        assert decoder.num_classes == NUM_CLASSES

    def test_is_nn_module(self, decoder):
        assert isinstance(decoder, torch.nn.Module)

    @pytest.mark.parametrize("num_classes", [2, 10, 30, 80])
    def test_custom_num_classes(self, num_classes):
        dec = SementicDecoder(num_classes, P3_CH, P4_CH, P5_CH)
        p3 = torch.randn(1, P3_CH, P3_H, P3_W)
        p4 = torch.randn(1, P4_CH, P4_H, P4_W)
        p5 = torch.randn(1, P5_CH, P5_H, P5_W)
        out = dec(p3, p4, p5)
        assert out.shape[1] == num_classes

    @pytest.mark.parametrize("p3_ch,p4_ch,p5_ch", [
        (32, 64, 128),
        (64, 128, 256),
    ])
    def test_custom_channel_widths(self, p3_ch, p4_ch, p5_ch):
        dec = SementicDecoder(NUM_CLASSES, p3_ch, p4_ch, p5_ch)
        p3 = torch.randn(1, p3_ch, P3_H, P3_W)
        p4 = torch.randn(1, p4_ch, P4_H, P4_W)
        p5 = torch.randn(1, p5_ch, P5_H, P5_W)
        out = dec(p3, p4, p5)
        assert out.shape[1] == NUM_CLASSES


class TestSementicDecoderFPNFusion:
    """Tests that all three FPN paths contribute to the output."""

    def test_p3_contributes_to_output(self, decoder):
        """Zeroing P3 should change the output."""
        p3_zero = torch.zeros(1, P3_CH, P3_H, P3_W)
        p3_rand = torch.randn(1, P3_CH, P3_H, P3_W)
        p4 = torch.randn(1, P4_CH, P4_H, P4_W)
        p5 = torch.randn(1, P5_CH, P5_H, P5_W)
        out_zero = decoder(p3_zero, p4, p5)
        out_rand = decoder(p3_rand, p4, p5)
        assert not torch.allclose(out_zero, out_rand)

    def test_p4_contributes_to_output(self, decoder):
        p3 = torch.randn(1, P3_CH, P3_H, P3_W)
        p4_zero = torch.zeros(1, P4_CH, P4_H, P4_W)
        p4_rand = torch.randn(1, P4_CH, P4_H, P4_W)
        p5 = torch.randn(1, P5_CH, P5_H, P5_W)
        out_zero = decoder(p3, p4_zero, p5)
        out_rand = decoder(p3, p4_rand, p5)
        assert not torch.allclose(out_zero, out_rand)

    def test_p5_contributes_to_output(self, decoder):
        p3 = torch.randn(1, P3_CH, P3_H, P3_W)
        p4 = torch.randn(1, P4_CH, P4_H, P4_W)
        p5_zero = torch.zeros(1, P5_CH, P5_H, P5_W)
        p5_rand = torch.randn(1, P5_CH, P5_H, P5_W)
        out_zero = decoder(p3, p4, p5_zero)
        out_rand = decoder(p3, p4, p5_rand)
        assert not torch.allclose(out_zero, out_rand)

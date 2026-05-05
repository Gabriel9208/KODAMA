from src.datasets.SANPO_dataset import SANPO__dataset
import torch

def test_SANPO__dataset():

    data_dir = "data/processed/-5OCPnbrwJdu3jH70ieU7pUiFsOJQoeG/camera_chest/left"

    dataset = SANPO__dataset(data_dir=data_dir)

    assert dataset is not None
    assert dataset[0] is not None, "First sample should not be None"

    rgb, seg, depth = dataset[0]
    assert rgb is not None, "RGB should not be None"
    assert seg is not None, "Seg should not be None"
    assert depth is not None, "Depth should not be None"

    assert rgb.shape == (3, 640, 640), "Sample shape should be (3, 640, 640)"
    assert seg.shape == (2, 640, 640), "Sample shape should be (2, 640, 640)"
    assert depth.shape == (640, 640), "Sample shape should be (640, 640)"

    assert rgb.dtype == torch.int32, "RGB should be int32"
    assert seg.dtype == torch.int32, "Seg should be int32"
    assert depth.dtype == torch.float32, "Depth should be float32"    

    assert not torch.isnan(rgb).any(), "RGB should not contain NaN"
    assert not torch.isnan(seg).any(), "Seg should not contain NaN"
    assert not torch.isnan(depth).any(), "Depth should not contain NaN"

    assert dataset[0][0].min() >= 0 and dataset[0][0].max() <= 255, "RGB values should be in range [0, 255]"
    assert dataset[0][1].min() >= 0 and dataset[0][1][0].max() <= 30, "Seg values should be in range [0, 65535]"
    assert dataset[0][1].min() >= 0 and dataset[0][1][1].max() <= 65535, "Seg values should be in range [0, 65535]"
    assert dataset[0][2].min() >= 0, "Depth values should be >= 0"
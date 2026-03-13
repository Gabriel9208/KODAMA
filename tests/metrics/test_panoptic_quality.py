import pytest
import torch
from src.metrics.panoptic_quality import calculate_pq

def test_pq_perfect_match():
    shape = (100, 100, 2) # h, w, (sementic id, instance id)
    ground_truth = torch.zeros(shape, dtype=torch.int32)
    
    # A background
    ground_truth[:, :, 0] = 0
    
    # A single instance of class 1
    ground_truth[20:40, 30:60, 0] = 1
    ground_truth[20:40, 30:60, 1] = 1
    
    predictions = ground_truth.clone()
    
    result = calculate_pq(ground_truth, predictions)
    
    assert result.classes[0].PQ == pytest.approx(1.0, rel=1e-5), f"Expected PQ of 1.0, got {result.classes[0].PQ}"
    
    assert result.mPQ == pytest.approx(1.0, rel=1e-5), f"Expected PQ of 1.0, got {result.mPQ}"
    assert result.mSQ == pytest.approx(1.0, rel=1e-5), f"Expected SQ of 1.0, got {result.mSQ}"
    assert result.mRQ == pytest.approx(1.0, rel=1e-5), f"Expected RQ of 1.0, got {result.mRQ}"


def test_pq_division_by_zero_prevention(): # |TP| = 0
    shape = (10, 10, 2) # h, w, (semantic id, instance id)
    ground_truth = torch.zeros(shape, dtype=torch.int32)
    predictions = torch.ones(shape, dtype=torch.int32)
    
    result = calculate_pq(ground_truth, predictions)
    
    assert result.classes[0].PQ == pytest.approx(0.0, rel=1e-5), f"Expected PQ of 0.0, got {result.classes[0].PQ}"
    
    assert result.mPQ == pytest.approx(0.0, rel=1e-5), f"Expected PQ of 0.0, got {result.mPQ}"
    assert result.mSQ == pytest.approx(0.0, rel=1e-5), f"Expected SQ of 0.0, got {result.mSQ}"   
    assert result.mRQ == pytest.approx(0.0, rel=1e-5), f"Expected RQ of 0.0, got {result.mRQ}"
    
    
def test_pq_partial_match():
    shape = (100, 100, 2) # h, w, (sementic id, instance id)
    ground_truth = torch.zeros(shape, dtype=torch.int32)
    predictions = ground_truth.clone()

    # A background
    ground_truth[:, :, 0] = 0
    
    # A single instance of class 1
    ground_truth[:50, :, 0] = 1
    ground_truth[:50, :, 1] = 1
    

    predictions[10:60, :, 0] = 1
    predictions[10:60, :, 1] = 1
    
    result = calculate_pq(ground_truth, predictions)
    
    assert result.mPQ == pytest.approx(2/3, rel=1e-5), f"Expected PQ of 2/3, got {result.mPQ}"
    assert result.mSQ == pytest.approx(2/3, rel=1e-5), f"Expected SQ of 2/3, got {result.mSQ}"
    assert result.mRQ == pytest.approx(1.0, rel=1e-5), f"Expected RQ of 1.0, got {result.mRQ}"
    

def test_pq_over_segmentation():
    shape = (100, 100, 2) # h, w, (semantic id, instance id)
    ground_truth = torch.zeros(shape, dtype=torch.int32)
    predictions = ground_truth.clone()

    # A background
    ground_truth[:, :, 0] = 0
    
    # Two overlapping instances of class 1
    ground_truth[:, 30:60, 0] = 1
    ground_truth[:, 30:60, 1] = 1
    
    predictions[:, 30:50, 0] = 1
    predictions[:, 30:50, 1] = 1
    predictions[:, 50:60, 0] = 1
    predictions[:, 50:60, 1] = 2
    
    result = calculate_pq(ground_truth, predictions)
    
    assert result.classes[1].PQ == pytest.approx(4/9, rel=1e-5), f"Expected PQ of 4/9, got {result.classes[1].PQ}"
    
    assert result.mPQ == pytest.approx(13/18, rel=1e-5), f"Expected PQ of 13/18, got {result.mPQ}"
    assert result.mSQ == pytest.approx(5/6, rel=1e-5), f"Expected SQ of 5/6, got {result.mSQ}"
    assert result.mRQ == pytest.approx(5/6, rel=1e-5), f"Expected RQ of 5/6, got {result.mRQ}"
    
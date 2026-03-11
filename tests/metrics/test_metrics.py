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
    
    pq, sq, rq = calculate_pq(ground_truth, predictions) 
    
    assert pq == pytest.approx(1.0, rel=1e-5), f"Expected PQ of 1.0, got {pq}"
    assert sq == pytest.approx(1.0, rel=1e-5), f"Expected SQ of 1.0, got {sq}"
    assert rq == pytest.approx(1.0, rel=1e-5), f"Expected RQ of 1.0, got {rq}"


def test_pq_division_by_zero_prevention():
    shape = (10, 10, 2) # h, w, (semantic id, instance id)
    ground_truth = torch.zeros(shape, dtype=torch.int32)
    predictions = torch.ones(shape, dtype=torch.int32)
    
    pq, sq, rq = calculate_pq(ground_truth, predictions)
    
    assert pq == pytest.approx(0.0, rel=1e-5), f"Expected PQ of 0.0, got {pq}"
    assert sq == pytest.approx(0.0, rel=1e-5), f"Expected SQ of 0.0, got {sq}"   
    assert rq == pytest.approx(0.0, rel=1e-5), f"Expected RQ of 0.0, got {rq}"
    
def test_pq_partial_match():
    shape = (100, 100, 2) # h, w, (sementic id, instance id)
    ground_truth = torch.zeros(shape, dtype=torch.int32)
    
    # A background
    ground_truth[:, :, 0] = 0
    
    # A single instance of class 1
    ground_truth[:50, :, 0] = 1
    ground_truth[:50, :, 1] = 1
    
    predictions = ground_truth.clone()

    predictions[10:60, :, 0] = 1
    predictions[10:60, :, 1] = 1
    
    pq, sq, rq = calculate_pq(ground_truth, predictions)
    
    assert pq == pytest.approx(2/3, rel=1e-5), f"Expected PQ of 2/3, got {pq}"
    assert sq == pytest.approx(2/3, rel=1e-5), f"Expected SQ of 2/3, got {sq}"
    assert rq == pytest.approx(2/3, rel=1e-5), f"Expected RQ of 2/3, got {rq}"
    
    
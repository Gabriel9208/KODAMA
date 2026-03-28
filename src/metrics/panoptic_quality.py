from dataclasses import dataclass
import torch
import numpy as np

@dataclass
class PanopticQuality:
    PQ: float
    SQ: float
    RQ: float
    TP: int
    FP: int
    FN: int
    
@dataclass
class PanopticQualityResult:
    mPQ: float
    mSQ: float
    mRQ: float
    PQ_th: float  # Things # TODO
    PQ_st: float  # Stuff only # TODO
    classes: dict[int, PanopticQuality]
    

def calculate_pq(g: torch.Tensor, p: torch.Tensor) -> PanopticQualityResult:
    # g: (H, W, 2) - (semantic id, instance id)
    # p: (H, W, 2) - (semantic id, instance id)
    
    g = g.detach().cpu().numpy().reshape(-1, 2)
    p = p.detach().cpu().numpy().reshape(-1, 2)
    
    # Unique class ids in g and p
    unique_classes = np.sort(np.unique(np.concatenate((g[:, 0], p[:, 0]))))
       
    # Turn (semantic id, instance id) into a single id for easier processing, ex. 2001 for class 2, instance 1
    max_instance_id = max(g[:, 1].max(), p[:, 1].max())
    OFFSET = 10 ** len(str(max_instance_id))
    
    g_ids = np.sum(g * np.array([OFFSET, 1]), axis=-1)
    p_ids = np.sum(p * np.array([OFFSET, 1]), axis=-1)
    
    # Calculate intersection ids for each pair of (g, p), 
    # ex. 20011001 for class 2, instance 1 in g and class 1, instance 1 in p
    max_class_id = max(g_ids.max(), p_ids.max())
    INTERSECT_OFFSET = 10 ** len(str(max_class_id))
    
    intersect_ids = g_ids * INTERSECT_OFFSET + p_ids
    
    # Calculate areas for g, p, and intersection
    g_ids, g_area = np.unique(g_ids, return_counts=True)
    p_ids, p_area = np.unique(p_ids, return_counts=True)
    intersect_ids, intersect_area = np.unique(intersect_ids, return_counts=True)
    
    g_area = dict(zip(g_ids, g_area))
    p_area = dict(zip(p_ids, p_area))

    result = PanopticQualityResult(mPQ=0.0, mSQ=0.0, mRQ=0.0, classes={})
    
    N_g = {}
    N_p = {}
    for i in g_ids:
        N_g[i // OFFSET] = N_g.get(i // OFFSET, 0) + 1
    for i in p_ids:
        N_p[i // OFFSET] = N_p.get(i // OFFSET, 0) + 1

    for class_id in unique_classes:
        
        result.classes[class_id] = PanopticQuality(PQ=0.0, SQ=0.0, RQ=0.0, TP=0, FP=0, FN=0)
        
        n_g = N_g.get(class_id, 0)
        n_p = N_p.get(class_id, 0)
        iou_sum = 0.0
        
        for i in range(len(intersect_ids)):
            
            gid = intersect_ids[i] // INTERSECT_OFFSET
            pid = intersect_ids[i] % INTERSECT_OFFSET
            
            if gid // OFFSET == class_id and pid // OFFSET == class_id:
                    
                intersection = intersect_area[i]
                
                union = g_area[gid] + p_area[pid] - intersection
                iou = float(intersection) / union
                
                if iou > 0.5:
                    iou_sum += iou
                    result.classes[class_id].TP += 1
                                
        result.classes[class_id].FP = n_p - result.classes[class_id].TP
        result.classes[class_id].FN = n_g - result.classes[class_id].TP
        result.classes[class_id].SQ = float(iou_sum / result.classes[class_id].TP if result.classes[class_id].TP > 0 else 0.0)
        result.classes[class_id].RQ = float(result.classes[class_id].TP / (result.classes[class_id].TP + 0.5 * result.classes[class_id].FP + 0.5 * result.classes[class_id].FN))
        result.classes[class_id].PQ = float(result.classes[class_id].SQ * result.classes[class_id].RQ)
        
        result.mPQ += result.classes[class_id].PQ
        result.mSQ += result.classes[class_id].SQ
        result.mRQ += result.classes[class_id].RQ
    
    result.mPQ /= len(unique_classes)
    result.mSQ /= len(unique_classes)
    result.mRQ /= len(unique_classes)
    
    return result
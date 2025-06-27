import numpy as np
import torch

def IOU(true: np.ndarray, pred: np.ndarray, num_classes: int, FINETUNE_DATASET) -> dict:
    true = true.cpu().numpy() if isinstance(true, torch.Tensor) else true
    pred = pred.cpu().numpy() if isinstance(pred, torch.Tensor) else pred
    if true.shape != pred.shape:
        if len(true.shape) == 3:
            true = true[:len(pred), :len(pred[0]), :len(pred[0][0])]
        else:
            true = true[:len(pred), :len(pred[0])]
    labels = np.array(range(num_classes))
    ious = []
    for label in labels:
        intersection = np.sum(np.logical_and(np.equal(true, label), np.equal(pred, label)))
        union = np.sum(np.logical_or(np.equal(true, label), np.equal(pred, label)))
        ious.append(intersection * 1.0 / union if union > 0 else 0)
    iou_dict = {"IOU_{}".format(label): iou for label, iou in zip(labels, ious)}
    
    if FINETUNE_DATASET == "metaldam":
        # Remove the minority class (class 3)
        iou_dict["mean_IOU"] = np.mean(np.delete(ious, 3))
    elif FINETUNE_DATASET == "aachen":
        iou_dict["mean_IOU"] = np.mean(ious)
    else:
        iou_dict["mean_IOU"] = np.mean(ious)
        
        
    # iou_dict["mean_IOU"] = np.mean(ious)
    return iou_dict


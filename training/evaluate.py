"""
DeepShield AI — Evaluation Metrics

Computes AUC, F1, Precision, Recall, ROC curve, and EER
for binary deepfake detection (0=real, 1=fake).
"""

import json
import numpy as np
from pathlib import Path

from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    roc_curve, confusion_matrix, average_precision_score,
)


def compute_metrics(y_true: list, y_prob: list, threshold: float = 0.5) -> dict:
    """
    Compute standard deepfake detection metrics.

    Parameters
    ----------
    y_true    : list of int (0=real, 1=fake)
    y_prob    : list of float (probability of FAKE class, 0–1)
    threshold : decision threshold (default 0.5)

    Returns
    -------
    dict with: auc, f1, precision, recall, eer, ap, confusion_matrix
    """
    y_true  = np.array(y_true)
    y_prob  = np.array(y_prob)
    y_pred  = (y_prob >= threshold).astype(int)

    auc = float(roc_auc_score(y_true, y_prob))
    ap  = float(average_precision_score(y_true, y_prob))
    f1  = float(f1_score(y_true, y_pred, zero_division=0))
    pre = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))

    # EER (Equal Error Rate) — lower is better
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = float(fpr[eer_idx])

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "auc":              round(auc, 4),
        "average_precision": round(ap,  4),
        "f1":               round(f1,  4),
        "precision":        round(pre, 4),
        "recall":           round(rec, 4),
        "eer":              round(eer, 4),
        "threshold":        threshold,
        "confusion_matrix": cm,          # [[TN, FP], [FN, TP]]
        "total_samples":    len(y_true),
        "fake_count":       int(y_true.sum()),
        "real_count":       int((1 - y_true).sum()),
    }


def evaluate_checkpoint(checkpoint_path: str, data_dir: str, model_name: str = "efficientnet_b4"):
    """
    Standalone evaluation from a saved checkpoint.

    Usage:
        python training/evaluate.py --checkpoint models/efficientnet_b4_deepfake.pth
                                    --data_dir data/datasets --dataset ff++
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import torch
    from torch.utils.data import DataLoader
    from backend.detection.model_zoo import get_model
    from training.datasets.unified_dataset import UnifiedDeepfakeDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = get_model(model_name, pretrained=False).to(device).eval()

    state = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)

    dataset = UnifiedDeepfakeDataset(data_dir, datasets=["ff++"], augment=False)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    metrics = compute_metrics(all_labels, all_probs)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir",   default="./data/datasets")
    p.add_argument("--model",      default="efficientnet_b4")
    args = p.parse_args()
    evaluate_checkpoint(args.checkpoint, args.data_dir, args.model)

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, recall_score


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred)
    return {
        "acc": acc,
        "sn": sn,
        "sp": sp,
        "mcc": mcc,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "sn_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }
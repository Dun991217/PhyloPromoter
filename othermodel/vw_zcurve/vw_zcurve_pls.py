import os
import argparse
import itertools
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

ALPHABET = "ACGT"


def read_fasta_sequences(fasta_path: str):
    seqs = []
    names = []

    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    name = None
    seq_chunks = []

    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    seq = "".join(seq_chunks).upper()
                    seqs.append(seq)
                    names.append(name)
                name = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line)

    if name is not None:
        seq = "".join(seq_chunks).upper()
        seqs.append(seq)
        names.append(name)

    return names, seqs


def clean_sequence(seq: str):
    seq = seq.upper().replace("U", "T")
    allowed = set(ALPHABET)
    if not set(seq).issubset(allowed):
        return None
    return seq


def generate_all_kmers(k: int):
    return ["".join(p) for p in itertools.product(ALPHABET, repeat=k)]


def kmer_frequencies(seq: str, k: int):
    total = len(seq) - k + 1
    all_kmers = generate_all_kmers(k)

    if total <= 0:
        return {kmer: 0.0 for kmer in all_kmers}

    counts = Counter()
    valid_total = 0

    for i in range(total):
        kmer = seq[i:i + k]
        if set(kmer).issubset(set(ALPHABET)):
            counts[kmer] += 1
            valid_total += 1

    if valid_total == 0:
        return {kmer: 0.0 for kmer in all_kmers}

    return {kmer: counts.get(kmer, 0) / valid_total for kmer in all_kmers}


def zcurve_group_features(freqs, prefix: str):
    a = freqs.get(prefix + "A", 0.0)
    c = freqs.get(prefix + "C", 0.0)
    g = freqs.get(prefix + "G", 0.0)
    t = freqs.get(prefix + "T", 0.0)

    x = (a + g) - (c + t)
    y = (a + c) - (g + t)
    z = (a + t) - (c + g)
    return [x, y, z]


def vw_zcurve_features(seq: str, max_w: int = 6):
    features = []

    for w in range(1, max_w + 1):
        freqs = kmer_frequencies(seq, w)

        if w == 1:
            features.extend(zcurve_group_features(freqs, ""))
        else:
            prefixes = generate_all_kmers(w - 1)
            for prefix in prefixes:
                features.extend(zcurve_group_features(freqs, prefix))

    return np.asarray(features, dtype=np.float32)


def build_dataset(pos_fasta: str, neg_fasta: str, max_w: int = 6):
    pos_names, pos_seqs = read_fasta_sequences(pos_fasta)
    neg_names, neg_seqs = read_fasta_sequences(neg_fasta)

    X = []
    y = []
    sample_names = []
    raw_seqs = []

    dropped = 0

    for name, seq in zip(pos_names, pos_seqs):
        seq = clean_sequence(seq)
        if seq is None:
            dropped += 1
            continue
        X.append(vw_zcurve_features(seq, max_w=max_w))
        y.append(1)
        sample_names.append(name)
        raw_seqs.append(seq)

    for name, seq in zip(neg_names, neg_seqs):
        seq = clean_sequence(seq)
        if seq is None:
            dropped += 1
            continue
        X.append(vw_zcurve_features(seq, max_w=max_w))
        y.append(0)
        sample_names.append(name)
        raw_seqs.append(seq)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)

    return X, y, sample_names, raw_seqs, dropped


class VWZCurvePLSClassifier:
    def __init__(self, n_components=20, scale=True):
        self.n_components = n_components
        self.scale = scale
        self.scaler = StandardScaler() if scale else None
        self.model = PLSRegression(n_components=n_components)

    def fit(self, X, y):
        # PLS 使用 -1 / +1 标签
        y_pls = np.where(y == 1, 1.0, -1.0)

        if self.scale:
            X = self.scaler.fit_transform(X)

        self.model.fit(X, y_pls)
        return self

    def decision_function(self, X):
        if self.scale:
            X = self.scaler.transform(X)
        scores = self.model.predict(X).ravel()
        return scores

    def predict(self, X):
        scores = self.decision_function(X)
        return (scores > 0).astype(np.int32)

    def predict_proba_like(self, X):
        # 不是严格概率，但可用于 AUC / AUPRC
        return self.decision_function(X)


def evaluate_binary(y_true, y_pred, y_score):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)  # sensitivity
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_acc,
        "mcc": matthews_corrcoef(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_score),
        "auprc": average_precision_score(y_true, y_score),
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true, y_pred, digits=4, output_dict=True
        ),
    }
    return metrics


def save_prediction_table(out_csv, names, seqs, y_true, y_pred, y_score):
    df = pd.DataFrame({
        "name": names,
        "sequence": seqs,
        "y_true": y_true,
        "y_pred": y_pred,
        "score": y_score,
    })
    df.to_csv(out_csv, index=False)


def main():
    parser = argparse.ArgumentParser(description="vw Z-curve + PLS for promoter prediction")
    parser.add_argument("--train_pos", required=True, help="Training positive FASTA")
    parser.add_argument("--train_neg", required=True, help="Training negative FASTA")
    parser.add_argument("--test_pos", required=True, help="Testing positive FASTA")
    parser.add_argument("--test_neg", required=True, help="Testing negative FASTA")
    parser.add_argument("--max_w", type=int, default=6, help="Maximum window size for vw Z-curve")
    parser.add_argument("--n_components", type=int, default=20, help="PLS n_components")
    parser.add_argument("--no_scale", action="store_true", help="Disable StandardScaler")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio split from training set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[INFO] Building full training dataset...")
    X_full, y_full, full_names, full_seqs, dropped_train = build_dataset(
        args.train_pos, args.train_neg, max_w=args.max_w
    )

    print("[INFO] Building testing dataset...")
    X_test, y_test, test_names, test_seqs, dropped_test = build_dataset(
        args.test_pos, args.test_neg, max_w=args.max_w
    )

    train_idx, val_idx = train_test_split(
        np.arange(len(y_full)),
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=y_full,
    )

    X_train = X_full[train_idx]
    y_train = y_full[train_idx]

    X_val = X_full[val_idx]
    y_val = y_full[val_idx]
    val_names = [full_names[i] for i in val_idx]
    val_seqs = [full_seqs[i] for i in val_idx]

    print(f"[INFO] Full train shape : X={X_full.shape}, y={y_full.shape}, dropped={dropped_train}")
    print(f"[INFO] Train split shape: X={X_train.shape}, y={y_train.shape}")
    print(f"[INFO] Val split shape  : X={X_val.shape}, y={y_val.shape}")
    print(f"[INFO] Test shape       : X={X_test.shape}, y={y_test.shape}, dropped={dropped_test}")

    clf = VWZCurvePLSClassifier(
        n_components=args.n_components,
        scale=(not args.no_scale),
    )

    print("[INFO] Training model...")
    clf.fit(X_train, y_train)

    print("[INFO] Predicting validation set...")
    val_score = clf.predict_proba_like(X_val)
    val_pred = clf.predict(X_val)
    val_metrics = evaluate_binary(y_val, val_pred, val_score)

    print("\n===== Validation Metrics =====")
    print(f"Accuracy          : {val_metrics['accuracy']:.6f}")
    print(f"Balanced Accuracy : {val_metrics['balanced_accuracy']:.6f}")
    print(f"MCC               : {val_metrics['mcc']:.6f}")
    print(f"AUC               : {val_metrics['auc']:.6f}")
    print(f"AUPRC             : {val_metrics['auprc']:.6f}")
    print(f"Precision         : {val_metrics['precision']:.6f}")
    print(f"Recall            : {val_metrics['recall']:.6f}")
    print(f"Specificity       : {val_metrics['specificity']:.6f}")
    print(f"F1                : {val_metrics['f1']:.6f}")
    print(f"Confusion Matrix  : {val_metrics['confusion_matrix']}")

    print("[INFO] Predicting test set...")
    y_score = clf.predict_proba_like(X_test)
    y_pred = clf.predict(X_test)
    test_metrics = evaluate_binary(y_test, y_pred, y_score)

    print("\n===== Test Metrics =====")
    print(f"Accuracy          : {test_metrics['accuracy']:.6f}")
    print(f"Balanced Accuracy : {test_metrics['balanced_accuracy']:.6f}")
    print(f"MCC               : {test_metrics['mcc']:.6f}")
    print(f"AUC               : {test_metrics['auc']:.6f}")
    print(f"AUPRC             : {test_metrics['auprc']:.6f}")
    print(f"Precision         : {test_metrics['precision']:.6f}")
    print(f"Recall            : {test_metrics['recall']:.6f}")
    print(f"Specificity       : {test_metrics['specificity']:.6f}")
    print(f"F1                : {test_metrics['f1']:.6f}")
    print(f"Confusion Matrix  : {test_metrics['confusion_matrix']}")

    val_metrics_simple = {
        "accuracy": val_metrics["accuracy"],
        "balanced_accuracy": val_metrics["balanced_accuracy"],
        "mcc": val_metrics["mcc"],
        "auc": val_metrics["auc"],
        "auprc": val_metrics["auprc"],
        "precision": val_metrics["precision"],
        "recall": val_metrics["recall"],
        "sensitivity": val_metrics["sensitivity"],
        "specificity": val_metrics["specificity"],
        "f1": val_metrics["f1"],
        "confusion_matrix": str(val_metrics["confusion_matrix"]),
        "n_train": len(y_train),
        "n_val": len(y_val),
        "max_w": args.max_w,
        "n_components": args.n_components,
        "scale": not args.no_scale,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
    }
    pd.DataFrame([val_metrics_simple]).to_csv(
        os.path.join(args.out_dir, "val_metrics.csv"),
        index=False
    )

    test_metrics_simple = {
        "accuracy": test_metrics["accuracy"],
        "balanced_accuracy": test_metrics["balanced_accuracy"],
        "mcc": test_metrics["mcc"],
        "auc": test_metrics["auc"],
        "auprc": test_metrics["auprc"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "sensitivity": test_metrics["sensitivity"],
        "specificity": test_metrics["specificity"],
        "f1": test_metrics["f1"],
        "confusion_matrix": str(test_metrics["confusion_matrix"]),
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
        "max_w": args.max_w,
        "n_components": args.n_components,
        "scale": not args.no_scale,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
    }
    pd.DataFrame([test_metrics_simple]).to_csv(
        os.path.join(args.out_dir, "test_metrics.csv"),
        index=False
    )

    report_df = pd.DataFrame(test_metrics["classification_report"]).transpose()
    report_df.to_csv(os.path.join(args.out_dir, "classification_report.csv"))

    save_prediction_table(
        os.path.join(args.out_dir, "val_predictions.csv"),
        val_names,
        val_seqs,
        y_val,
        val_pred,
        val_score,
    )

    save_prediction_table(
        os.path.join(args.out_dir, "test_predictions.csv"),
        test_names,
        test_seqs,
        y_test,
        y_pred,
        y_score,
    )

    print(f"[OK] Results saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
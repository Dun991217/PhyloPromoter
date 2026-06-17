from __future__ import annotations

import argparse
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from dataset import SequenceDataset, build_classification_dataset, build_identification_dataset
from evaluate import binary_metrics, multiclass_metrics
from feature_extractor import KmerWord2VecExtractor, Word2VecConfig
from model import LSTMAttentionClassifier
from utils import load_yaml, save_json, set_seed


def fit_one_fold(model, train_loader, valid_loader, device, lr, epochs, num_classes):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = -1.0
    best_y_true, best_y_pred = None, None

    for _ in range(epochs):
        model.train()
        for x, lengths, y in train_loader:
            x = x.to(device)
            lengths = lengths.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits, _ = model(x, lengths)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        model.eval()
        fold_true, fold_pred = [], []
        with torch.no_grad():
            for x, lengths, y in valid_loader:
                x = x.to(device)
                lengths = lengths.to(device)
                logits, _ = model(x, lengths)
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                fold_pred.extend(pred.tolist())
                fold_true.extend(y.numpy().tolist())

        y_true = np.asarray(fold_true)
        y_pred = np.asarray(fold_pred)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_y_true = y_true.copy()
            best_y_pred = y_pred.copy()

    if num_classes == 2:
        return binary_metrics(best_y_true, best_y_pred), best_y_true, best_y_pred
    return multiclass_metrics(best_y_true, best_y_pred), best_y_true, best_y_pred


def main(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(cfg["train"]["seed"])

    task = cfg["task"]
    if task == "identification":
        sequences, labels = build_identification_dataset(
            cfg["data"]["promoter_path"],
            cfg["data"]["non_promoter_path"],
        )
    elif task == "classification":
        sequences, labels = build_classification_dataset(
            cfg["data"]["strong_path"],
            cfg["data"]["weak_path"],
        )
    else:
        raise ValueError(f"Unsupported task: {task}")

    extractor_cfg = Word2VecConfig(
        kmer=cfg["feature"]["kmer"],
        stride=cfg["feature"]["stride"],
        vector_size=cfg["feature"]["vector_size"],
        window=cfg["feature"]["window"],
        min_count=cfg["feature"]["min_count"],
        sg=cfg["feature"]["sg"],
        workers=cfg["feature"]["workers"],
        epochs=cfg["feature"]["epochs"],
        seed=cfg["train"]["seed"],
    )

    extractor = KmerWord2VecExtractor(extractor_cfg)
    extractor.fit(sequences)
    x, lengths = extractor.encode(sequences)
    emb = extractor.embedding_matrix()
    labels = np.asarray(labels, dtype=np.int64)

    device_name = cfg["train"]["device"]
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else device_name if device_name != "auto" else "cpu")

    skf = StratifiedKFold(
        n_splits=cfg["train"]["n_splits"],
        shuffle=True,
        random_state=cfg["train"]["seed"],
    )

    fold_metrics = []
    all_true, all_pred = [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(x, labels), start=1):
        train_ds = SequenceDataset(x[tr_idx], lengths[tr_idx], labels[tr_idx])
        valid_ds = SequenceDataset(x[va_idx], lengths[va_idx], labels[va_idx])

        train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
        valid_loader = DataLoader(valid_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

        model = LSTMAttentionClassifier(
            vocab_size=emb.shape[0],
            embed_dim=emb.shape[1],
            hidden_dim=cfg["model"]["hidden_dim"],
            num_layers=cfg["model"]["num_layers"],
            num_classes=cfg["model"]["num_classes"],
            bidirectional=cfg["model"]["bidirectional"],
            dropout=cfg["model"]["dropout"],
            embedding_matrix=emb,
            freeze_embedding=cfg["model"]["freeze_embedding"],
        ).to(device)

        metrics, y_true_fold, y_pred_fold = fit_one_fold(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            device=device,
            lr=cfg["train"]["lr"],
            epochs=cfg["train"]["epochs"],
            num_classes=cfg["model"]["num_classes"],
        )
        metrics["fold"] = fold
        fold_metrics.append(metrics)
        all_true.extend(y_true_fold.tolist())
        all_pred.extend(y_pred_fold.tolist())
        print(f"Fold {fold}: {metrics}")

    all_true = np.asarray(all_true)
    all_pred = np.asarray(all_pred)

    if cfg["model"]["num_classes"] == 2:
        overall = binary_metrics(all_true, all_pred)
    else:
        overall = multiclass_metrics(all_true, all_pred)

    result = {
        "task": task,
        "feature_config": asdict(extractor_cfg),
        "model_config": cfg["model"],
        "train_config": cfg["train"],
        "fold_metrics": fold_metrics,
        "overall": overall,
    }

    print("\nOverall:", overall)
    save_json(result, cfg["output"]["result_path"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to yaml config")
    args = parser.parse_args()
    main(args.config)
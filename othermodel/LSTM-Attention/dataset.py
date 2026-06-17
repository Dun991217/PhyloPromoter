from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import load_sequences_auto


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, lengths: np.ndarray, y: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.long)
        self.lengths = torch.as_tensor(lengths, dtype=torch.long)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.lengths[idx], self.y[idx]


def build_identification_dataset(promoter_path: str, non_promoter_path: str):
    promoters = load_sequences_auto(promoter_path)
    non_promoters = load_sequences_auto(non_promoter_path)
    seqs = promoters + non_promoters
    labels = [1] * len(promoters) + [0] * len(non_promoters)
    return seqs, labels


def build_classification_dataset(strong_path: str, weak_path: str):
    strong = load_sequences_auto(strong_path)
    weak = load_sequences_auto(weak_path)
    seqs = strong + weak
    labels = [1] * len(strong) + [0] * len(weak)
    return seqs, labels
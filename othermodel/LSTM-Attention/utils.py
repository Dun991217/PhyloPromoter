from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean_dna(seq: str) -> str:
    seq = seq.strip().upper().replace("U", "T")
    allowed = set("ATCGN")
    return "".join(ch for ch in seq if ch in allowed)


def read_fasta(path: str | Path):
    sequences = []
    current = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    sequences.append(clean_dna("".join(current)))
                    current = []
            else:
                current.append(line)
        if current:
            sequences.append(clean_dna("".join(current)))
    return sequences


def read_txt_sequences(path: str | Path):
    seqs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = clean_dna(line)
            if line:
                seqs.append(line)
    return seqs


def load_sequences_auto(path: str | Path):
    path = str(path)
    if path.lower().endswith((".fa", ".fasta", ".fna")):
        return read_fasta(path)
    return read_txt_sequences(path)


def load_yaml(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
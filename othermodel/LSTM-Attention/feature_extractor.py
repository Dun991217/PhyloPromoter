from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from gensim.models import Word2Vec


def make_kmers(seq: str, k: int, stride: int = 1):
    if len(seq) < k:
        return []
    return [seq[i:i + k] for i in range(0, len(seq) - k + 1, stride)]


@dataclass
class Word2VecConfig:
    kmer: int = 1
    stride: int = 1
    vector_size: int = 100
    window: int = 5
    min_count: int = 1
    sg: int = 1
    workers: int = 4
    epochs: int = 20
    seed: int = 42


class KmerWord2VecExtractor:
    def __init__(self, config: Word2VecConfig):
        self.config = config
        self.model: Optional[Word2Vec] = None

    def tokenize(self, sequences: Sequence[str]):
        return [make_kmers(seq, self.config.kmer, self.config.stride) for seq in sequences]

    def fit(self, sequences: Sequence[str]):
        tokenized = self.tokenize(sequences)
        self.model = Word2Vec(
            sentences=tokenized,
            vector_size=self.config.vector_size,
            window=self.config.window,
            min_count=self.config.min_count,
            sg=self.config.sg,
            workers=self.config.workers,
            epochs=self.config.epochs,
            seed=self.config.seed,
        )
        return self

    def vocab(self) -> Dict[str, int]:
        if self.model is None:
            raise RuntimeError("Word2Vec model is not fitted yet.")
        stoi = {"<PAD>": 0, "<UNK>": 1}
        for i, token in enumerate(self.model.wv.index_to_key, start=2):
            stoi[token] = i
        return stoi

    def embedding_matrix(self) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Word2Vec model is not fitted yet.")
        vocab = self.vocab()
        emb = np.zeros((len(vocab), self.config.vector_size), dtype=np.float32)
        if len(self.model.wv.index_to_key) > 0:
            emb[1] = np.mean(self.model.wv.vectors, axis=0)
        for token, idx in vocab.items():
            if token in {"<PAD>", "<UNK>"}:
                continue
            emb[idx] = self.model.wv[token]
        return emb

    def encode(self, sequences: Sequence[str], max_len: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Word2Vec model is not fitted yet.")
        vocab = self.vocab()
        tokenized = self.tokenize(sequences)
        if max_len is None:
            max_len = max(len(tokens) for tokens in tokenized)

        x = np.zeros((len(tokenized), max_len), dtype=np.int64)
        lengths = np.zeros((len(tokenized),), dtype=np.int64)

        for i, tokens in enumerate(tokenized):
            ids = [vocab.get(tok, 1) for tok in tokens[:max_len]]
            x[i, :len(ids)] = ids
            lengths[i] = len(ids)
        return x, lengths
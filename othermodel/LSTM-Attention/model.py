from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int, bidirectional: bool = True):
        super().__init__()
        self.out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.score = nn.Linear(self.out_dim, 1)

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        attn_logits = self.score(h).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(attn_logits, dim=-1)
        context = torch.sum(h * attn_weights.unsqueeze(-1), dim=1)
        return context, attn_weights


class LSTMAttentionClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int = 100,
        num_layers: int = 2,
        num_classes: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        embedding_matrix: np.ndarray | None = None,
        freeze_embedding: bool = False,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.tensor(embedding_matrix))
        self.embedding.weight.requires_grad = not freeze_embedding

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )

        self.attn = AttentionPooling(hidden_dim, bidirectional=bidirectional)
        feat_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        max_t = out.size(1)
        mask = torch.arange(max_t, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
        context, attn_weights = self.attn(out, mask)
        logits = self.classifier(context)
        return logits, attn_weights
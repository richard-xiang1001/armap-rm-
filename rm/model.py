#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn

from rm.pooling import pool_hidden


class CharTokenizer:
    """Minimal character tokenizer.

    - No external deps
    - Stable for open-sourcing and reproducibility

    Notes:
    - For real tasks, swap to a standard tokenizer (e.g., Llama/Qwen tokenizer).
    """

    def __init__(self):
        # Basic printable ASCII + some common unicode punctuation used in traces.
        base = [chr(i) for i in range(32, 127)]
        extra = list("。，“”‘’—…→✓✗")
        vocab = ["<pad>", "<unk>", "<bos>", "<eos>"] + base + extra
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = vocab
        self.pad_id = self.stoi["<pad>"]
        self.unk_id = self.stoi["<unk>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str, max_len: int) -> torch.Tensor:
        ids = [self.bos_id]
        for ch in text:
            ids.append(self.stoi.get(ch, self.unk_id))
            if len(ids) >= max_len - 1:
                break
        ids.append(self.eos_id)
        if len(ids) < max_len:
            ids += [self.pad_id] * (max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)


class TextRewardModel(nn.Module):
    """A minimal text-only reward model.

    Architecture:
    - char embedding
    - BiGRU encoder
    - configurable pooling (mean/last/last_k_step)
    - scalar head -> reward

    This matches the *shape* of ARMAP RM: backbone -> scalar head.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
        pad_id: int = 0,
        pooling: str = "mean",
        last_k_steps_pool: int = 3,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.pooling = pooling
        self.last_k_steps_pool = max(int(last_k_steps_pool), 1)
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.rnn = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size * 2, 1)

    def forward(self, input_ids: torch.Tensor, step_mask: torch.Tensor | None = None) -> torch.Tensor:
        # input_ids: [B, T]
        mask = (input_ids != self.pad_id).float()  # [B, T]
        x = self.emb(input_ids)  # [B, T, D]
        h, _ = self.rnn(x)  # [B, T, 2H]
        h = self.dropout(h)

        pooled = pool_hidden(h=h, mask=mask, strategy=self.pooling, step_mask=step_mask)
        reward = self.head(pooled).squeeze(-1)  # [B]
        return reward

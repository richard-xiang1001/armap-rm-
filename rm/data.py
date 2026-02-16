#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterator, List

import torch
from torch.utils.data import Dataset


@dataclass
class PairExample:
    instruction: str
    pos: str
    neg: str
    meta: Dict


def _read_jsonl(path: str) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


class PreferencePairDataset(Dataset):
    def __init__(self, path: str, use_refined_instruction: bool = True):
        self.items: List[PairExample] = []
        for obj in _read_jsonl(path):
            instr = obj.get("instruction_refined") if use_refined_instruction else obj.get("instruction_raw")
            self.items.append(
                PairExample(
                    instruction=instr,
                    pos=obj["traj_pos"],
                    neg=obj["traj_neg"],
                    meta=obj.get("meta", {}),
                )
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> PairExample:
        return self.items[idx]


class PairCollator:
    """Builds two sequences per example: (instruction + pos_traj) and (instruction + neg_traj)."""

    def __init__(self, tokenizer, max_len: int = 512, sep: str = "\n\n---\n\n"):
        self.tok = tokenizer
        self.max_len = max_len
        self.sep = sep

    def __call__(self, batch: List[PairExample]) -> Dict[str, torch.Tensor]:
        pos_texts = [b.instruction + self.sep + b.pos for b in batch]
        neg_texts = [b.instruction + self.sep + b.neg for b in batch]

        pos_ids = torch.stack([self.tok.encode(t, self.max_len) for t in pos_texts], dim=0)
        neg_ids = torch.stack([self.tok.encode(t, self.max_len) for t in neg_texts], dim=0)

        return {"pos_ids": pos_ids, "neg_ids": neg_ids}

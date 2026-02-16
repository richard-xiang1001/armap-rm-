#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterator, List, Tuple

import torch
from torch.utils.data import Dataset

from rm.formatters.stepwise import format_instruction, format_trajectory_stepwise, get_last_k_step_char_start


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

    def __init__(
        self,
        tokenizer,
        max_len: int = 512,
        sep: str = "\n\n---\n\n",
        input_format: str = "flat",
        last_k_steps_pool: int = 3,
    ):
        self.tok = tokenizer
        self.max_len = max_len
        self.sep = sep
        self.input_format = input_format
        self.last_k_steps_pool = max(int(last_k_steps_pool), 1)

    def _format(self, instruction: str, traj: str) -> Tuple[str, int]:
        instruction = str(instruction or "")
        traj = str(traj or "")

        if self.input_format == "stepwise":
            instruction_text = format_instruction(instruction)
            traj_text = format_trajectory_stepwise(traj)
        else:
            instruction_text = instruction
            traj_text = traj

        text = instruction_text + self.sep + traj_text
        traj_start = len(instruction_text + self.sep)
        region_start = traj_start + get_last_k_step_char_start(traj_text, last_k_steps=self.last_k_steps_pool)
        return text, region_start

    def _step_mask(self, text: str, region_start: int) -> torch.Tensor:
        keep_chars = max(self.max_len - 2, 0)
        visible = min(len(text), keep_chars)
        mask = torch.zeros(self.max_len, dtype=torch.float32)
        if visible <= 0:
            return mask

        start = max(0, min(region_start, visible))
        if start < visible:
            mask[(start + 1) : (visible + 1)] = 1.0  # +1 for BOS offset
            eos_idx = min(visible + 1, self.max_len - 1)
            mask[eos_idx] = 1.0
        return mask

    def __call__(self, batch: List[PairExample]) -> Dict[str, torch.Tensor]:
        pos_payloads = [self._format(b.instruction, b.pos) for b in batch]
        neg_payloads = [self._format(b.instruction, b.neg) for b in batch]

        pos_ids = torch.stack([self.tok.encode(text, self.max_len) for text, _ in pos_payloads], dim=0)
        neg_ids = torch.stack([self.tok.encode(text, self.max_len) for text, _ in neg_payloads], dim=0)

        pos_step_mask = torch.stack([self._step_mask(text, start) for text, start in pos_payloads], dim=0)
        neg_step_mask = torch.stack([self._step_mask(text, start) for text, start in neg_payloads], dim=0)

        return {
            "pos_ids": pos_ids,
            "neg_ids": neg_ids,
            "pos_step_mask": pos_step_mask,
            "neg_step_mask": neg_step_mask,
        }

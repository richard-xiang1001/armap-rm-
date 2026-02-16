#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from rm.data import PairCollator, PreferencePairDataset
from rm.model import CharTokenizer, TextRewardModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--valid_path", type=str, required=True)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--input_format", type=str, default="", choices=["", "flat", "stepwise"])
    ap.add_argument("--use_refined_instruction", type=str, default="", choices=["", "true", "false"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["config"]

    tok = CharTokenizer()
    model = TextRewardModel(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        pad_id=cfg["pad_id"],
        pooling=cfg.get("pooling", "mean"),
        last_k_steps_pool=cfg.get("last_k_steps_pool", 3),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    use_refined = cfg.get("use_refined_instruction", True)
    if args.use_refined_instruction:
        use_refined = args.use_refined_instruction == "true"
    input_format = args.input_format or cfg.get("input_format", "flat")

    ds = PreferencePairDataset(args.valid_path, use_refined_instruction=use_refined)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=PairCollator(tok, max_len=args.max_len, input_format=input_format, last_k_steps_pool=cfg.get("last_k_steps_pool", 3)),
        pin_memory=torch.cuda.is_available(),
    )

    correct = 0
    total = 0
    gap_sum = 0.0
    deltas = []

    with torch.no_grad():
        for batch in loader:
            pos_ids = batch["pos_ids"].to(device)
            neg_ids = batch["neg_ids"].to(device)
            pos_step_mask = batch.get("pos_step_mask")
            neg_step_mask = batch.get("neg_step_mask")
            if pos_step_mask is not None:
                pos_step_mask = pos_step_mask.to(device)
            if neg_step_mask is not None:
                neg_step_mask = neg_step_mask.to(device)
            r_pos = model(pos_ids, step_mask=pos_step_mask)
            r_neg = model(neg_ids, step_mask=neg_step_mask)
            delta = (r_pos - r_neg).detach().cpu().numpy()
            correct += (r_pos > r_neg).sum().item()
            total += r_pos.numel()
            gap_sum += float(delta.sum())
            deltas.append(delta)

    pair_acc = correct / max(total, 1)
    avg_gap = gap_sum / max(total, 1)
    if deltas:
        all_delta = np.concatenate(deltas)
        gap_p10 = float(np.percentile(all_delta, 10))
        gap_p50 = float(np.percentile(all_delta, 50))
        gap_p90 = float(np.percentile(all_delta, 90))
    else:
        gap_p10 = 0.0
        gap_p50 = 0.0
        gap_p90 = 0.0
    print(
        {
            "pair_accuracy": pair_acc,
            "avg_reward_gap": avg_gap,
            "gap_p10": gap_p10,
            "gap_p50": gap_p50,
            "gap_p90": gap_p90,
            "n": total,
        }
    )


if __name__ == "__main__":
    main()

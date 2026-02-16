#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from rm.data import PairCollator, PreferencePairDataset
from rm.model import CharTokenizer, TextRewardModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pairwise_loss(r_pos: torch.Tensor, r_neg: torch.Tensor) -> torch.Tensor:
    # -log sigmoid(r_pos - r_neg)
    return F.softplus(-(r_pos - r_neg)).mean()


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    gaps = []
    losses = []

    for batch in loader:
        pos_ids = batch["pos_ids"].to(device)
        neg_ids = batch["neg_ids"].to(device)
        r_pos = model(pos_ids)
        r_neg = model(neg_ids)
        loss = pairwise_loss(r_pos, r_neg)
        gap = (r_pos - r_neg).detach().cpu().numpy()
        gaps.append(gap)
        losses.append(loss.item())

        correct += (r_pos > r_neg).sum().item()
        total += r_pos.numel()

    gaps_np = np.concatenate(gaps) if gaps else np.array([0.0])
    return {
        "pair_accuracy": float(correct / max(total, 1)),
        "avg_reward_gap": float(gaps_np.mean()),
        "std_reward_gap": float(gaps_np.std()),
        "avg_loss": float(np.mean(losses) if losses else 0.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", type=str, required=True)
    ap.add_argument("--valid_path", type=str, required=True)
    ap.add_argument("--save_dir", type=str, required=True)

    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2)

    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--hidden_size", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)

    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = CharTokenizer()
    model = TextRewardModel(
        vocab_size=tok.vocab_size,
        d_model=args.d_model,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pad_id=tok.pad_id,
    ).to(device)

    train_ds = PreferencePairDataset(args.train_path)
    valid_ds = PreferencePairDataset(args.valid_path)

    collate = PairCollator(tok, max_len=args.max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # logging
    metrics_path = os.path.join(args.save_dir, "metrics.jsonl")
    ckpt_path = os.path.join(args.save_dir, "rm.pt")

    global_step = 0
    best_valid = -1.0

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in pbar:
            pos_ids = batch["pos_ids"].to(device)
            neg_ids = batch["neg_ids"].to(device)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                r_pos = model(pos_ids)
                r_neg = model(neg_ids)
                loss = pairwise_loss(r_pos, r_neg)

            scaler.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            global_step += 1
            if global_step % 50 == 0:
                with torch.no_grad():
                    acc = (r_pos > r_neg).float().mean().item()
                    gap = (r_pos - r_neg).mean().item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "pair_acc": f"{acc:.3f}", "gap": f"{gap:.3f}"})

        valid_metrics = evaluate(model, valid_loader, device)
        record: Dict[str, float | int] = {
            "epoch": epoch,
            "global_step": global_step,
            "elapsed_sec": round(time.time() - t0, 2),
            **valid_metrics,
        }
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # save best
        if valid_metrics["pair_accuracy"] > best_valid:
            best_valid = valid_metrics["pair_accuracy"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "tokenizer": {"type": "CharTokenizer"},
                    "config": {
                        "vocab_size": tok.vocab_size,
                        "d_model": args.d_model,
                        "hidden_size": args.hidden_size,
                        "num_layers": args.num_layers,
                        "dropout": args.dropout,
                        "pad_id": tok.pad_id,
                        "max_len": args.max_len,
                    },
                },
                ckpt_path,
            )

        print(f"[valid] {valid_metrics}")
        print(f"[best_pair_acc] {best_valid:.4f}")

    print(f"Done. Saved best checkpoint to: {ckpt_path}")


if __name__ == "__main__":
    main()

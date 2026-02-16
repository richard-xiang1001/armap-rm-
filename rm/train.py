#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from contextlib import nullcontext
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


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_amp_dtype(amp_dtype: str, use_cuda: bool) -> str:
    if not use_cuda:
        return "none"
    if amp_dtype == "none":
        return "none"
    if amp_dtype == "fp16":
        return "fp16"
    if amp_dtype == "bf16":
        if torch.cuda.is_bf16_supported():
            return "bf16"
        return "fp16"
    return "none"


def parse_auto_bool(value: str, auto_value: bool) -> bool:
    if value == "auto":
        return auto_value
    return value == "true"


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
    ap.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--amp_dtype", type=str, choices=["none", "bf16", "fp16"], default="bf16")
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=0, help="0 means full epoch training.")
    ap.add_argument("--pin_memory", type=str, choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--persistent_workers", type=str, choices=["auto", "true", "false"], default="auto")

    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--hidden_size", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)

    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    set_seed(args.seed)

    if args.grad_accum_steps < 1:
        raise ValueError("--grad_accum_steps must be >= 1")
    if args.max_steps < 0:
        raise ValueError("--max_steps must be >= 0")

    device = resolve_device(args.device)
    use_cuda = device.type == "cuda"
    amp_dtype_used = resolve_amp_dtype(args.amp_dtype, use_cuda=use_cuda)
    pin_memory = parse_auto_bool(args.pin_memory, auto_value=use_cuda)
    persistent_workers = parse_auto_bool(args.persistent_workers, auto_value=args.num_workers > 0)

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

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["persistent_workers"] = persistent_workers

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        **loader_kwargs,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        **loader_kwargs,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype_used == "fp16")
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_dtype_used == "fp16")

    # logging
    metrics_path = os.path.join(args.save_dir, "metrics.jsonl")
    ckpt_path = os.path.join(args.save_dir, "rm.pt")

    global_step = 0
    best_valid = -1.0

    t0 = time.time()
    total_samples_seen = 0
    total_tokens_seen = 0
    reached_max_steps = False
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        opt.zero_grad(set_to_none=True)
        accum_counter = 0
        last_batch_pair_acc = 0.0
        last_batch_gap = 0.0
        last_batch_loss = 0.0
        for batch_idx, batch in enumerate(pbar):
            pos_ids = batch["pos_ids"].to(device)
            neg_ids = batch["neg_ids"].to(device)

            batch_samples = pos_ids.size(0)
            batch_tokens = int((pos_ids != tok.pad_id).sum().item() + (neg_ids != tok.pad_id).sum().item())
            total_samples_seen += batch_samples
            total_tokens_seen += batch_tokens

            if amp_dtype_used == "bf16":
                autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            elif amp_dtype_used == "fp16":
                autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
            else:
                autocast_ctx = nullcontext()

            with autocast_ctx:
                r_pos = model(pos_ids)
                r_neg = model(neg_ids)
                raw_loss = pairwise_loss(r_pos, r_neg)
                loss = raw_loss / args.grad_accum_steps

            if amp_dtype_used == "fp16":
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_counter += 1
            is_last_batch = batch_idx == (len(train_loader) - 1)
            if accum_counter >= args.grad_accum_steps or is_last_batch:
                if args.grad_clip and args.grad_clip > 0:
                    if amp_dtype_used == "fp16":
                        scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if amp_dtype_used == "fp16":
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)

                global_step += 1
                accum_counter = 0
                with torch.no_grad():
                    last_batch_pair_acc = (r_pos > r_neg).float().mean().item()
                    last_batch_gap = (r_pos - r_neg).mean().item()
                    last_batch_loss = raw_loss.item()

                if global_step % 50 == 0:
                    elapsed = max(time.time() - t0, 1e-6)
                    pbar.set_postfix(
                        {
                            "loss": f"{last_batch_loss:.4f}",
                            "pair_acc": f"{last_batch_pair_acc:.3f}",
                            "gap": f"{last_batch_gap:.3f}",
                            "samples/s": f"{(total_samples_seen / elapsed):.1f}",
                        }
                    )

                if args.max_steps > 0 and global_step >= args.max_steps:
                    reached_max_steps = True
                    break

        valid_metrics = evaluate(model, valid_loader, device)
        elapsed = max(time.time() - t0, 1e-6)
        record: Dict[str, float | int | str] = {
            "epoch": epoch,
            "global_step": global_step,
            "elapsed_sec": round(elapsed, 2),
            "samples_per_sec": float(total_samples_seen / elapsed),
            "tokens_per_sec": float(total_tokens_seen / elapsed),
            "amp_dtype_used": amp_dtype_used,
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
        if reached_max_steps:
            print(f"[early_stop] reached max_steps={args.max_steps}")
            break

    print(f"Done. Saved best checkpoint to: {ckpt_path}")


if __name__ == "__main__":
    main()

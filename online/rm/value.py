from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple

import torch

from rm.formatters.stepwise import format_instruction, format_trajectory_stepwise, get_last_k_step_char_start
from rm.model import CharTokenizer, TextRewardModel

PAIR_SEP = "\n\n---\n\n"


@lru_cache(maxsize=8)
def _load_bundle(ckpt: str, device_key: str):
    map_device = "cpu" if device_key == "auto" else device_key
    ckpt_obj = torch.load(ckpt, map_location="cpu")
    cfg = dict(ckpt_obj["config"])

    if device_key == "cpu":
        device = torch.device("cpu")
    elif device_key == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested cuda device but CUDA is unavailable.")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = CharTokenizer()
    model = TextRewardModel(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        pad_id=cfg["pad_id"],
        pooling=str(cfg.get("pooling", "mean")),
        last_k_steps_pool=int(cfg.get("last_k_steps_pool", 3)),
    ).to(device)
    model.load_state_dict(ckpt_obj["state_dict"], strict=True)
    model.eval()
    return tok, model, cfg, device


def _to_text(item: Dict, input_format: str, stepwise_style: str) -> Tuple[str, int]:
    instruction = str(item.get("instruction_refined") or "").strip()
    prefix = str(item.get("trajectory_prefix_text") or "").strip()
    action_text = str(item.get("action_text") or "").strip()
    rollout_text = str(item.get("rollout_text") or "").strip()

    tail = "\n".join([x for x in [action_text, rollout_text] if x]).strip()
    traj = "\n".join([x for x in [prefix, tail] if x]).strip()

    if input_format == "stepwise":
        instruction_text = format_instruction(instruction, style=stepwise_style)
        traj_text = format_trajectory_stepwise(traj, style=stepwise_style)
    else:
        instruction_text = instruction
        traj_text = traj

    full_text = instruction_text + PAIR_SEP + traj_text
    traj_start = len(instruction_text + PAIR_SEP)
    region_start = traj_start + get_last_k_step_char_start(traj_text, last_k_steps=3)
    return full_text, region_start


def _step_mask(text: str, region_start: int, max_len: int) -> torch.Tensor:
    keep_chars = max(max_len - 2, 0)
    visible = min(len(text), keep_chars)
    mask = torch.zeros(max_len, dtype=torch.float32)
    if visible <= 0:
        return mask
    start = max(0, min(region_start, visible))
    if start < visible:
        mask[(start + 1) : (visible + 1)] = 1.0
        eos_idx = min(visible + 1, max_len - 1)
        mask[eos_idx] = 1.0
    return mask


def score_action_batch(ckpt: str, items: list[dict], device: str, max_len: int, batch_size: int) -> list[float]:
    if not items:
        return []

    tok, model, cfg, torch_device = _load_bundle(str(ckpt), str(device))
    max_len_eff = int(max_len) if int(max_len) > 0 else int(cfg.get("max_len", 512))
    input_format = str(cfg.get("input_format", "flat"))
    pooling = str(cfg.get("pooling", "mean"))
    stepwise_style = "xml_v1"

    payload: List[Dict] = []
    for row in items:
        text, region_start = _to_text(row, input_format=input_format, stepwise_style=stepwise_style)
        payload.append({"text": text, "region_start": region_start})

    out: List[float] = []
    with torch.no_grad():
        for start in range(0, len(payload), max(int(batch_size), 1)):
            batch = payload[start : start + max(int(batch_size), 1)]
            ids = torch.stack([tok.encode(x["text"], max_len=max_len_eff) for x in batch], dim=0).to(torch_device)
            if pooling == "last_k_step":
                masks = torch.stack([_step_mask(x["text"], x["region_start"], max_len=max_len_eff) for x in batch], dim=0).to(torch_device)
                rewards = model(ids, step_mask=masks)
            else:
                rewards = model(ids, step_mask=None)
            out.extend([float(v) for v in rewards.detach().cpu().tolist()])
    return out

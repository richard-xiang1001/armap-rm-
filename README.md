# ARMAP Reward Model (Text-only) — Minimal Repro

This repo provides a **minimal runnable** text-only Reward Model (RM) training pipeline inspired by **ARMAP: Scaling Autonomous Agents via Automatic Reward Modeling and Planning**.

Scope (this repro):
- ✅ Build an RM that scores (instruction, trajectory) pairs.
- ✅ Train with **pairwise preference loss** on (pos, neg) trajectory pairs.
- ✅ Provide an auditable, open-source-friendly training protocol (`docs/training_details.md`).
- ❌ Not targeting full benchmark parity with the paper.

## Quickstart

### 1) Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

### 2) Generate a tiny dataset

```bash
python3 scripts/prepare_data.py \
  --out_dir data \
  --n_train 20000 \
  --n_valid 2000 \
  --seed 42
```

### 3) Train RM

```bash
python3 -m rm.train \
  --train_path data/train.jsonl \
  --valid_path data/valid.jsonl \
  --save_dir results/exp1 \
  --max_len 512 \
  --batch_size 64 \
  --lr 3e-4 \
  --epochs 3 \
  --seed 42
```

### 4) Lint Data

```bash
python3 scripts/lint_data.py \
  --path data/train.jsonl \
  --max_len 512 \
  --last_k_steps 3 \
  --step_markers "step,action:,observation:,obs:"
```

This reports:
- required-field completeness
- empty trajectory ratio
- length statistics
- keyword leakage hits (`reward/success/done/score` by default)
- `Final:` truncation risk at the target `max_len`
- tail-step truncation risk (`truncation_risk_last_k_steps`)

Optional leak-term extension:

```bash
python3 scripts/lint_data.py \
  --path data/train.jsonl \
  --leak_terms_file configs/leak_terms.txt \
  --fail_on_leak
```

### 5) Evaluate

```bash
python3 -m rm.eval \
  --ckpt results/exp1/rm.pt \
  --valid_path data/valid.jsonl \
  --max_len 512
```

Notes:
- Keep `max_len` large enough to include the trajectory tail (where `Final: ...` often appears). For this synthetic data, `max_len=512` is safe.
- In restricted macOS/sandbox environments, set `--num_workers 0` to avoid shared-memory worker errors.
- Use `docs/reproducibility.md` and `CHANGELOG.md` as the canonical record for experiment and release metadata.

You should see:
- `pair_accuracy` > 0.5 (better than random)
- a positive average `reward_gap = R(pos)-R(neg)`
- `gap_p10/p50/p90` to inspect reward-gap distribution stability

## What is being learned?
The RM learns a scalar scoring function **R(x, h)** such that for the same instruction *x*,

- R(x, h⁺) > R(x, h⁻)

and is trained with the pairwise logistic objective.

See `docs/training_details.md` for a full engineering-level protocol.

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

GPU/4090-friendly flags (optional):

```bash
python3 -m rm.train \
  --train_path data/train.jsonl \
  --valid_path data/valid.jsonl \
  --save_dir results/smoke_gpu \
  --device cuda \
  --amp_dtype bf16 \
  --batch_size 256 \
  --grad_accum_steps 1 \
  --max_steps 200 \
  --num_workers 4 \
  --pin_memory auto \
  --prefetch_factor 2 \
  --persistent_workers auto
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
  --fail_on_leak \
  --fail_on_missing \
  --fail_on_truncation_risk \
  --max_truncation_risk_last_k_steps 0.05
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

## v0.2.0 Real-Trajectory Pipeline

This repo now includes an ingest pipeline:

1. `ingest/export_pairs.py`: Episode JSONL -> RM pair JSONL
2. `ingest/sanitize_traj.py`: leak masking + step normalization
3. strict lint gate (`scripts/lint_data.py`)
4. `scripts/run_v020_pipeline.py`: one-command end-to-end flow

Generate a minimal de-identified sample:

```bash
python3 scripts/generate_real_episode_sample.py \
  --out_path data/raw/example_episodes.jsonl \
  --n_tasks 50 \
  --rollouts_per_task 3 \
  --seed 42
```

Run full v0.2.0 flow:

```bash
python3 scripts/run_v020_pipeline.py \
  --raw_path data/raw/example_episodes.jsonl \
  --adapter generic_jsonl \
  --work_dir data/real \
  --results_dir results/v020_real \
  --max_len 512 \
  --batch_size 128 \
  --epochs 2 \
  --max_steps 200 \
  --device auto \
  --amp_dtype bf16
```

## v0.2.1 Acceptance (5000/500 + CPU/GPU parity)

Use one command to run export/sanitize/gate/train/eval and parity checks:

```bash
python3 scripts/run_v021_acceptance.py \
  --raw_path data/raw/episodes_real.jsonl \
  --adapter generic_jsonl \
  --work_dir data/real \
  --results_dir results/v021_real \
  --max_pairs 5500 \
  --target_train_size 5000 \
  --target_valid_size 500 \
  --max_len 512 \
  --batch_size 256 \
  --epochs 3 \
  --max_steps 4000 \
  --device cuda \
  --amp_dtype bf16 \
  --num_workers 4
```

Report output:
- `data/real/reports/v021_acceptance.json`

Artifacts:
- `data/real/pairs.unsanitized.jsonl`
- `data/real/pairs.sanitized.jsonl`
- `data/real/train.jsonl`
- `data/real/valid.jsonl`
- `data/real/reports/*.json`
- `results/v020_real/*`

You should see:
- `pair_accuracy` > 0.5 (better than random)
- a positive average `reward_gap = R(pos)-R(neg)`
- `gap_p10/p50/p90` to inspect reward-gap distribution stability

## What is being learned?
The RM learns a scalar scoring function **R(x, h)** such that for the same instruction *x*,

- R(x, h⁺) > R(x, h⁻)

and is trained with the pairwise logistic objective.

See `docs/training_details.md` for a full engineering-level protocol.

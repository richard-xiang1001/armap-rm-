# Reproducibility

This file is the canonical template for reporting RM training runs in this repo.

## Run Record Template

Use the following checklist for every public run:

- Date (YYYY-MM-DD):
- Git commit hash:
- Host OS:
- Python version:
- Torch version:
- CUDA available:
- CUDA device count:
- CUDA device model:
- Dataset path and size:
- Data source type (synthetic / real episodes):
- Train command:
- Eval command:
- Data lint command:
- Seed:
- Key outputs:
  - `pair_accuracy`:
  - `avg_reward_gap`:
  - `gap_p10/p50/p90`:
  - `samples_per_sec`:
  - `tokens_per_sec`:
  - `amp_dtype_used`:
- CPU/GPU parity check (`|pair_acc_gpu - pair_acc_cpu|`):
- Known constraints:
- Notes:

## v0.1.0 Reference Run (2026-02-16)

- Date: 2026-02-16
- Git commit hash: (fill during release tag)
- Host OS: macOS 15.6.1 (Darwin arm64, sandboxed)
- Python version: 3.13.2
- Torch version: 2.9.1
- CUDA available: false
- CUDA device count: 0
- Dataset path and size:
  - `data/train.jsonl` = 20,000 rows
  - `data/valid.jsonl` = 2,000 rows
- Train command:
  - `python3 -m rm.train --train_path data/train.jsonl --valid_path data/valid.jsonl --save_dir results/exp1 --max_len 512 --batch_size 64 --lr 3e-4 --epochs 3 --num_workers 0 --seed 42`
- Eval command:
  - `python3 -m rm.eval --ckpt results/exp1/rm.pt --valid_path data/valid.jsonl --max_len 512 --num_workers 0`
- Data lint command:
  - `python3 scripts/lint_data.py --path data/train.jsonl --max_len 512 --last_k_steps 3 --step_markers "step,action:,observation:,obs:"`
- Seed: 42
- Key outputs:
  - `results/exp1/metrics.jsonl` (3 epochs)
  - `results/exp1/rm.pt`
  - final eval:
    - `pair_accuracy=0.7155`
    - `avg_reward_gap=0.9959664268493652`
    - `gap_p10=-0.460713654756546`
    - `gap_p50=0.4952836036682129`
    - `gap_p90=3.0919368267059326`
    - `n=2000`
- Known constraints:
  - In restricted macOS/sandbox environments, default worker settings can fail with `torch_shm_manager ... Operation not permitted`; use `--num_workers 0`.
- Notes:
  - A smoke run with `max_len=128` can truncate the `Final:` line and collapse preference signal; use `max_len=512` for this dataset.

## v0.2.0 Reference Run Template (4090 + Real Episodes)

Use this template for the v0.2.0 release report:

- Date: YYYY-MM-DD
- Git commit hash:
- Host OS:
- Python version:
- Torch version:
- CUDA available: true
- CUDA device count:
- CUDA device model: RTX 4090 (24GB)
- Dataset path and size:
  - `data/real/pairs.unsanitized.jsonl`:
  - `data/real/pairs.sanitized.jsonl`:
  - `data/real/train.jsonl`:
  - `data/real/valid.jsonl`:
- Export command:
  - `python3 ingest/export_pairs.py ...`
- Sanitize command:
  - `python3 ingest/sanitize_traj.py ...`
- Lint gate command:
  - `python3 scripts/lint_data.py --fail_on_leak --fail_on_missing --fail_on_truncation_risk --max_truncation_risk_last_k_steps 0.05 ...`
- Train command:
  - `python3 -m rm.train --device cuda --amp_dtype bf16 --max_steps 200 ...`
- Eval command:
  - `python3 -m rm.eval ...`
- Seed:
- Gate checks:
  - `rows_with_missing = 0`
  - `leak_hit_count = 0`
  - `truncation_risk_last_k_steps <= 0.05`
- Key outputs:
  - `pair_accuracy`:
  - `avg_reward_gap`:
  - `gap_p10/p50/p90`:
  - `samples_per_sec`:
  - `tokens_per_sec`:
  - `amp_dtype_used`:
- CPU/GPU parity:
  - same seed/same valid set
  - `abs(pair_acc_gpu - pair_acc_cpu) <= 0.03`
- Known constraints:
  - If CUDA wheel installation is blocked by network policy, use offline wheels and `pip install --no-index --find-links wheels -r requirements.txt`.

## v0.2.1 Acceptance Command (5000/500 + Parity)

Run this acceptance script and archive its report:

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

Expected acceptance report path:
- `data/real/reports/v021_acceptance.json`

## v0.2.1 Reference Run (2026-02-16, AutoDL 4090)

- Date: 2026-02-16
- Git commit hash: (fill with release commit)
- Host: AutoDL
- CPU / RAM: 16 vCPU / 120GB
- Python version: 3.10 (conda `armap-rm`)
- CUDA available: true
- CUDA device count: 1
- CUDA device model: NVIDIA GeForce RTX 4090
- NVIDIA driver / CUDA (system): 560.35.03 / 12.6
- Dataset path and size:
  - `data/raw/episodes_real.jsonl`: 12,000 episodes
  - `data/real/pairs.unsanitized.jsonl`: 5,500 rows
  - `data/real/pairs.sanitized.jsonl`: 5,500 rows
  - `data/real/train.jsonl`: 5,000 rows
  - `data/real/valid.jsonl`: 500 rows
- Acceptance command:
  - `python3 scripts/run_v021_acceptance.py --raw_path data/raw/episodes_real.jsonl --adapter generic_jsonl --work_dir data/real --results_dir results/v021_real --max_pairs 5500 --target_train_size 5000 --target_valid_size 500 --max_len 512 --batch_size 256 --epochs 3 --max_steps 4000 --device cuda --amp_dtype bf16 --num_workers 4`
- Gate checks (from `data/real/reports/train_lint.json`):
  - `rows_with_missing = 0`
  - `leak_hit_count = 0`
  - `truncation_risk_last_k_steps = 0.0`
- Key outputs:
  - best valid `pair_accuracy = 0.618`
  - eval:
    - `pair_accuracy = 0.618`
    - `avg_reward_gap = 0.004093139052391052`
    - `gap_p10 = -0.016657302156090736`
    - `gap_p50 = 0.005510792136192322`
    - `gap_p90 = 0.023761045187711716`
    - `n = 500`
- CPU/GPU parity:
  - GPU eval `pair_accuracy = 0.618`
  - CPU eval `pair_accuracy = 0.618`
  - `abs(pair_acc_gpu - pair_acc_cpu) = 0.0 <= 0.03`
- Acceptance result:
  - `data/real/reports/v021_acceptance.json`
  - `passed = true`

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
- Dataset path and size:
- Train command:
- Eval command:
- Data lint command:
- Seed:
- Key outputs:
  - `pair_accuracy`:
  - `avg_reward_gap`:
  - `gap_p10/p50/p90`:
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

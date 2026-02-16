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

## v0.3.0 Acceptance Command (ARMAP-style negatives + parity)

Run this acceptance script and archive its report:

```bash
python3 scripts/run_v030_acceptance.py \
  --raw_path data/raw/episodes_real.jsonl \
  --adapter generic_jsonl \
  --work_dir data/real \
  --results_dir results/v030_real \
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
- `data/real/reports/v030_acceptance.json`

New gate checks in v0.3.0:
- `neg_replay_success_rate >= 0.8`
- `env_judge_consistency >= 0.95`

Remote execution handbook:
- `docs/runbooks/v030_acceptance_remote.md`

## v0.3.0 Reference Run (2026-02-16, AutoDL 4090, Precheck Dataset)

- Date: 2026-02-16
- Git commit hash: `c474f1a`
- Host: AutoDL
- Python version: 3.10 (`.venv`)
- CUDA available: true
- CUDA device count: 1
- CUDA device model: NVIDIA GeForce RTX 4090
- NVIDIA driver / CUDA (system): 560.35.03 / 12.6
- Dataset path and size:
  - `data/raw/example_episodes_v030.jsonl`: 12,000 episodes
  - `data/real_v030_precheck/pairs.unsanitized.jsonl`: 5,500 rows
  - `data/real_v030_precheck/pairs.sanitized.jsonl`: 5,500 rows
  - `data/real_v030_precheck/train.jsonl`: 5,000 rows
  - `data/real_v030_precheck/valid.jsonl`: 500 rows
- Acceptance command:
  - `python3 scripts/run_v030_acceptance.py --raw_path data/raw/example_episodes_v030.jsonl --adapter generic_jsonl --work_dir data/real_v030_precheck --results_dir results/v030_precheck --max_pairs 5500 --target_train_size 5000 --target_valid_size 500 --max_len 512 --batch_size 256 --epochs 3 --max_steps 4000 --device cuda --amp_dtype bf16 --num_workers 4 --min_pair_accuracy 0.55 --min_gap_p50 0.0 --min_replay_ok_rate 0.8 --min_env_judge_consistency 0.95`
- Gate checks (`data/real_v030_precheck/reports/train_lint.json`):
  - `rows_with_missing = 0`
  - `leak_hit_count = 0`
  - `truncation_risk_last_k_steps = 0.0`
  - `neg_replay_success_rate = 1.0`
  - `env_judge_consistency = 1.0`
- Key outputs (`results/v030_precheck` + eval):
  - best valid `pair_accuracy = 0.578`
  - eval:
    - `pair_accuracy = 0.582`
    - `avg_reward_gap = 0.0035885519981384275`
    - `gap_p10 = -0.01506007555872202`
    - `gap_p50 = 0.0026562009006738663`
    - `gap_p90 = 0.02276899665594101`
    - `n = 500`
- CPU/GPU parity:
  - GPU eval `pair_accuracy = 0.582`
  - CPU eval `pair_accuracy = 0.582`
  - `abs(pair_acc_gpu - pair_acc_cpu) = 0.0`
- Acceptance result:
  - `data/real_v030_precheck/reports/v030_acceptance.json`
  - `passed = true`

Note:
- This record is a 4090 precheck run using generated `example_episodes_v030.jsonl`.
- Official publication-grade evidence should additionally include a run with `data/raw/episodes_real.jsonl` once available.

## v0.3.1 Reference Run (2026-02-16, AutoDL 4090, Input-Format/Pooling Ablation)

- Date: 2026-02-16
- Git commit hash: `c474f1a` (training run)
- Host: AutoDL
- Python version: 3.10 (`.venv`)
- CUDA available: true
- CUDA device count: 1
- CUDA device model: NVIDIA GeForce RTX 4090
- NVIDIA driver / CUDA (system): 560.35.03 / 12.6
- Dataset path and size:
  - train: `data/real_v030_precheck/train.jsonl` (5,000)
  - valid: `data/real_v030_precheck/valid.jsonl` (500)
- Common train settings:
  - `epochs=3`, `max_steps=4000`, `batch_size=256`, `max_len=512`, `seed=42`, `device=cuda`, `amp_dtype=bf16`, `num_workers=4`

Runs:
- `flat + mean`:
  - save dir: `results/v031_flat_mean_512`
  - best valid `pair_accuracy = 0.582`
- `stepwise + mean`:
  - save dir: `results/v031_stepwise_mean_512`
  - best valid `pair_accuracy = 0.582`
- `stepwise + last_k_step (k=3)`:
  - save dir: `results/v031_stepwise_lastk_512`
  - best valid `pair_accuracy = 0.586`
- truncation check (`stepwise + last_k_step`, `max_len=384`):
  - save dir: `results/v031_stepwise_lastk_384`
  - best valid `pair_accuracy = 0.588`

Eval (same valid split, `num_workers=0`):
- `results/v031_stepwise_lastk_512/rm.pt`, `max_len=512`:
  - `pair_accuracy = 0.586`
  - `avg_reward_gap = 0.008283441185951233`
  - `gap_p10 = -0.03620664402842522`
  - `gap_p50 = 0.006407078355550766`
  - `gap_p90 = 0.05261464789509773`
  - `n = 500`
- `results/v031_stepwise_lastk_384/rm.pt`, `max_len=384`:
  - `pair_accuracy = 0.588`
  - `avg_reward_gap = 0.00826094126701355`
  - `gap_p10 = -0.03577784076333046`
  - `gap_p50 = 0.005827473476529121`
  - `gap_p90 = 0.052338626235723495`
  - `n = 500`

Summary:
- `stepwise + mean` is non-regressive vs `flat + mean`.
- `stepwise + last_k_step` improves pair accuracy on this setup.
- `max_len 512 -> 384` does not degrade pair accuracy in this run.

## v0.3.2 Reference Run (2026-02-16, AutoDL 4090, Webshop-like Scaffold Smoke)

- Date: 2026-02-16
- Git commit hash: `c474f1a` (runtime branch baseline)
- Host: AutoDL
- Python version: 3.10 (`.venv`)
- CUDA available: true
- CUDA device count: 1
- CUDA device model: NVIDIA GeForce RTX 4090
- NVIDIA driver / CUDA (system): 560.35.03 / 12.6

Data generation:
- command:
  - `python3 scripts/generate_webshop_like_sample.py --out_path data/raw/webshop_like_sample.jsonl --n_tasks 200 --rollouts_per_task 3 --seed 42`
- output:
  - `data/raw/webshop_like_sample.jsonl` with `600` episodes

Scaffold smoke command:
- `python3 scripts/build_pairs_v030.py --input_path data/raw/webshop_like_sample.jsonl --output_path data/webshop_like/pairs.unsanitized.jsonl --adapter webshop_like --max_pairs 500 --stats_path data/webshop_like/reports/export_stats_v030.json`

Smoke outputs (`data/webshop_like/reports/export_stats_v030.json`):
- `episodes_in = 600`
- `success_candidates = 295`
- `pairs_out = 295`
- `neg_type_counts = {"truncate_last_k": 295}`
- `neg_attempted_counts = {"replay_corrupt": 295, "swap_step": 295, "truncate_last_k": 295}`
- `neg_replay_success_rate = 0.0`
- `env_judge_consistency = 0.0`

Interpretation:
- Adapter + pair-construction path is functional for Webshop-like trajectory schema.
- Arithmetic env judge is not expected to validate Webshop-like tasks, so fallback to `truncate_last_k` is expected in this lightweight scaffold phase.

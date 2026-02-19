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

## v0.4.0 Acceptance Command (Real Evidence MVP)

Run this acceptance script and archive its report:

```bash
python3 scripts/run_v040_real_acceptance.py \
  --raw_path data/raw/episodes_real_v040.jsonl \
  --adapter real_logs_v1 \
  --env_judge real_logs \
  --work_dir data/real_v040 \
  --results_dir results/v040_real \
  --max_pairs 1200 \
  --target_train_size 1000 \
  --target_valid_size 200 \
  --max_len 512 \
  --batch_size 128 \
  --epochs 3 \
  --max_steps 2000 \
  --device cuda \
  --amp_dtype bf16
```

Expected acceptance report path:
- `data/real_v040/reports/v040_real_acceptance.json`

v0.4.0 gate additions:
- `judge_unknown_rate <= 0.05` (or temporarily `<= 0.10` for early real-log migration, documented in release notes)

## v0.4.2 Ablation Command (neg/refine/format/pooling)

```bash
python3 scripts/run_ablation_v04x.py \
  --raw_path data/raw/episodes_real_v040.jsonl \
  --adapter real_logs_v1 \
  --env_judge real_logs \
  --work_dir data/ablation/v04x \
  --results_dir results/ablation/v04x
```

Expected outputs:
- `results/ablation/v04x/v04x_ablation.csv`
- `results/ablation/v04x/v04x_ablation.md`

## v0.4.0 Reference Runs (2026-02-16)

### Release-grade attempt (blocked)

- Command:
  - `python3 scripts/run_v040_real_acceptance.py --raw_path data/raw/episodes_real_v040.jsonl --adapter real_logs_v1 --env_judge real_logs --work_dir data/real_v040 --results_dir results/v040_real --max_pairs 1200 --target_train_size 1000 --target_valid_size 200 --device cpu --amp_dtype none --epochs 1 --max_steps 50 --batch_size 64 --num_workers 0 --min_replay_ok_rate 0.0`
- Blocker evidence:
  - `data/real_v040/reports/export_stats_v040.json`: `pairs_trainable=31`
  - `data/real_v040/reports/train_lint.json`: `env_judge_consistency=1.0`, `judge_unknown_rate=0.0`
  - `data/real_v040/reports/v040_real_acceptance.json`: `passed=false` with only scale checks failing (`export_pairs_sufficient=false`, `split_train_exact=false`, `split_valid_exact=false`)
  - `data/real_v040/reports/precheck_pairs_summary.json`: `LT_300_DATA_SCALE_BLOCKER`

### Precheck run (non-release)

- Command:
  - `python3 scripts/run_v040_real_acceptance.py --raw_path data/raw/episodes_real_v040.jsonl --adapter real_logs_v1 --env_judge real_logs --work_dir data/real_v040_precheck --results_dir results/v040_real_precheck --max_pairs 31 --target_train_size 20 --target_valid_size 11 --device cpu --amp_dtype none --epochs 1 --max_steps 100 --batch_size 64 --num_workers 0 --min_replay_ok_rate 0.0 --min_env_judge_consistency 0.0 --max_judge_unknown_rate 1.0 --min_pair_accuracy 0.0 --min_gap_p50 -1.0`
- Result:
  - `data/real_v040_precheck/reports/v040_real_acceptance.json`: `passed=true`
  - `pair_accuracy=1.0`, `gap_p50=0.003827570006251335`, `judge_unknown_rate=0.0`, parity diff `0.0`

## v0.4.1 Protocol Migration Evidence (2026-02-16)

- Arithmetic judge run:
  - report: `data/v041_arithmetic/reports/v030_acceptance.json`
  - `passed=true`, `judge_unknown_rate=0.0`, `env_judge_consistency=1.0`
- Real-logs judge run:
  - report: `data/v041_real_logs/reports/v030_acceptance.json`
  - `passed=true`, `judge_unknown_rate=0.0`, `env_judge_consistency=1.0`

## v0.4.2 Ablation Evidence (2026-02-16)

- Command:
  - `python3 scripts/run_ablation_v04x.py --raw_path data/raw/episodes_real_v040.jsonl --adapter real_logs_v1 --env_judge real_logs --work_dir data/ablation/v04x --results_dir results/ablation/v04x --max_pairs 31 --target_valid_size 11 --device cpu --amp_dtype none --epochs 1 --max_steps 30 --batch_size 64 --num_workers 0 --bucket_sampling true`
- Outputs:
  - `results/ablation/v04x/v04x_ablation.csv`
  - `results/ablation/v04x/v04x_ablation.md`
- Matrix summary:
  - total `24` runs, `24` successful, `0` failed
  - `truncate_last_k`: `8` success, mean `pair_accuracy=1.0`
  - `swap_step`: `8` success, mean `pair_accuracy=0.6364` (primary weak bucket)
  - `replay_corrupt`: `8` success, mean `pair_accuracy=1.0`
- mismatch dumps:
  - generated files: `24`
  - non-empty mismatch files: `8` (all under `swap_step`)

### 4090 retry on full release-grade command (2026-02-16)

- Host: AutoDL RTX 4090
- Command:
  - `python3 scripts/run_v040_real_acceptance.py --raw_path data/raw/episodes_real_v040.jsonl --adapter real_logs_v1 --env_judge real_logs --work_dir data/real_v040 --results_dir results/v040_real --max_pairs 1200 --target_train_size 1000 --target_valid_size 200 --device cuda --amp_dtype bf16`
- Report:
  - `data/real_v040/reports/v040_real_acceptance.json`
- Outcome:
  - `passed=false`
  - failed checks: `export_pairs_sufficient`, `split_train_exact`, `split_valid_exact`
- Scale evidence:
  - `pairs_trainable=31` (target >= 1200)
- Quality evidence:
  - `judge_unknown_rate=0.0`
  - `env_judge_consistency=1.0`
  - `pair_accuracy=1.0`
  - `gap_p50≈0.004`
  - parity diff `<=0.03`
- Final status:
  - `LT_300_DATA_SCALE_BLOCKER` (release-grade still pending larger real logs)

## v0.5.0 TaskMix No-Agent Data Loop

### Episode Factory Command

```bash
python3 scripts/generate_taskmix_episodes_v050.py \
  --out_path data/raw/episodes_taskmix_v050.jsonl \
  --n_tasks 4000 \
  --rollouts_per_task 3 \
  --max_steps 8 \
  --seed 42 \
  --family_mix 1:1:1 \
  --success_policy balanced
```

Expected scale:
- `episodes = 12000`
- family distribution near `1:1:1` (`arithmetic/retrieval/constraint`)

### Episode Quality Gate Command

```bash
python3 scripts/validate_taskmix_episodes_v050.py \
  --input_path data/raw/episodes_taskmix_v050.jsonl \
  --report_path data/real_v050/reports/episode_factory_stats_v050.json \
  --min_episodes 12000 \
  --min_success_rate 0.30 \
  --max_success_rate 0.80 \
  --min_avg_steps 3.0 \
  --min_diversity 0.20 \
  --require_families arithmetic,retrieval,constraint \
  --require_done_reasons goal_satisfied,wrong_answer,invalid_action,constraint_violation,timeout_step_cap \
  --fail_on_gate
```

Gate report keys:
- `episodes_in`
- `success_rate`
- `avg_steps`
- `diversity`
- `family_distribution`
- `done_reason_distribution`

### v0.5.0 Acceptance Command (release-grade target unchanged)

```bash
python3 scripts/run_v050_taskmix_acceptance.py \
  --raw_path data/raw/episodes_taskmix_v050.jsonl \
  --adapter taskmix_v050 \
  --env_judge taskmix \
  --work_dir data/real_v050 \
  --results_dir results/v050_real \
  --max_pairs 1200 \
  --target_train_size 1000 \
  --target_valid_size 200 \
  --device cuda \
  --amp_dtype bf16
```

Expected acceptance report path:
- `data/real_v050/reports/v050_taskmix_acceptance.json`

### v0.5.1 Shortcut Audit Command

```bash
python3 scripts/audit_shortcuts_v050.py \
  --path data/real_v050/train.jsonl \
  --path data/real_v050/valid.jsonl \
  --forbidden_terms_file configs/leak_terms_v050.txt \
  --report_path data/real_v050/reports/shortcut_audit_v050.json \
  --fail_on_detected
```

Expected shortcut audit fields:
- `forbidden_term_hits`
- `hit_examples`
- `meta_text_overlap_rate`
- `passed`

### v0.5.x Ablation Command

```bash
python3 scripts/run_ablation_v05x.py \
  --raw_path data/raw/episodes_taskmix_v050.jsonl \
  --adapter taskmix_v050 \
  --env_judge taskmix \
  --work_dir data/ablation/v05x \
  --results_dir results/ablation/v05x
```

Expected outputs:
- `results/ablation/v05x/v05x_ablation.csv`
- `results/ablation/v05x/v05x_ablation.md`
- mismatch stats by family included in CSV/Markdown

## v0.5.1 Full-Scale Reference Run (2026-02-18, no-agent)

- Command:
  - `python3 scripts/run_v050_taskmix_acceptance.py --raw_path data/raw/episodes_taskmix_v050.jsonl --generate_if_missing true --force_generate true --n_tasks 4000 --rollouts_per_task 3 --family_mix 1:1:1 --success_policy balanced --work_dir data/real_v050_fullscale_v2 --results_dir results/v051_real_fullscale_v2 --max_pairs 1200 --target_train_size 1000 --target_valid_size 200 --device cpu --amp_dtype none --num_workers 0`
- Archived report:
  - `data/real_v050/reports/v050_taskmix_acceptance_v051_fullscale.json`
- Archived shortcut audit:
  - `data/real_v050/reports/shortcut_audit_v050_v051_fullscale.json`
- Archived scored pairs / mismatches:
  - `data/real_v050/reports/valid_scored_pairs_gpu_v051_fullscale.jsonl`
  - `data/real_v050/reports/valid_mismatches_gpu_v051_fullscale.jsonl`

Outcome summary:
- `passed=true`
- `pairs_trainable=1200`
- split exact: `train=1000`, `valid=200`
- `judge_unknown_rate=0.0`
- `env_judge_consistency=1.0`
- shortcut audit: `forbidden_term_hits=0`, `meta_text_overlap_rate=0.0`, `passed=true`
- eval: `pair_accuracy=1.0`, `gap_p50=0.01840946264564991`

Family breakdown (`family_metrics`, valid=200):
- `arithmetic`: `n=72`, `pair_accuracy=1.0`, `gap_p50=0.010292798280715942`
- `retrieval`: `n=65`, `pair_accuracy=1.0`, `gap_p50=0.02361416071653366`
- `constraint`: `n=63`, `pair_accuracy=1.0`, `gap_p50=0.01844429224729538`
- `unknown`: `n=0`

Mismatch family counts:
- `arithmetic=0`, `retrieval=0`, `constraint=0`, `unknown=0`

## v0.5.2 Curriculum + Counterfactual Audit (2026-02-18, no-agent)

### Full-Scale Acceptance Command

```bash
python3 scripts/run_v052_taskmix_acceptance.py \
  --raw_path data/raw/episodes_taskmix_v052_fullscale.jsonl \
  --generate_if_missing true \
  --force_generate true \
  --n_tasks 4000 \
  --rollouts_per_task 3 \
  --family_mix 1:1:1 \
  --success_policy balanced \
  --work_dir data/real_v052 \
  --results_dir results/v052_real_fullscale \
  --max_pairs 1200 \
  --target_train_size 1000 \
  --target_valid_size 200 \
  --device cpu \
  --amp_dtype none \
  --num_workers 0 \
  --report_path data/real_v052/reports/v052_taskmix_acceptance_fullscale.json
```

### Counterfactual Audit Command

```bash
python3 scripts/audit_counterfactual_format_v052.py \
  --ckpt results/v052_real_fullscale/rm.pt \
  --valid_path data/real_v052/stages/stage_3/valid.jsonl \
  --primary_format flat \
  --counterfactual_format stepwise \
  --max_len 512 \
  --num_workers 0 \
  --threshold 0.95 \
  --report_path data/real_v052/reports/counterfactual_audit_v052_fullscale.json \
  --primary_scored_path data/real_v052/reports/valid_scored_pairs_primary_fullscale.jsonl \
  --counterfactual_scored_path data/real_v052/reports/valid_scored_pairs_counterfactual_fullscale.jsonl
```

Archived artifacts:
- `data/real_v052/reports/v052_taskmix_acceptance_fullscale.json`
- `data/real_v052/reports/counterfactual_audit_v052_fullscale.json`
- `data/real_v052/reports/valid_scored_pairs_primary_fullscale.jsonl`
- `data/real_v052/reports/valid_scored_pairs_counterfactual_fullscale.jsonl`
- `data/real_v052/reports/valid_mismatches_gpu_fullscale.jsonl`

Outcome summary:
- `passed=true`
- `pairs_trainable=1200`
- split exact: `train=1000`, `valid=200`
- `judge_unknown_rate=0.0`
- `env_judge_consistency=1.0`
- shortcut audit: `forbidden_term_hits=0`, `meta_text_overlap_rate=0.0`, `passed=true`
- eval(primary): `pair_accuracy=1.0`, `gap_p50=0.021306009963154793`

Curriculum summary (`curriculum_mode=stage3`, ratio `40:30:30`):
- stage steps: `800/600/600`
- stage_1 (`truncate_last_k`): `pair_accuracy=1.0`
- stage_2 (`swap_step`): `pair_accuracy=0.75`
- stage_3 (`replay_corrupt`): `pair_accuracy=1.0`

Counterfactual audit summary (soft gate, threshold `0.95`):
- `sign_consistency_rate=0.685`
- `sign_flip_count=63`
- `passed_soft=false` (recorded in `advisory_checks`, not blocking `passed` under soft mode)

Family breakdown (`family_metrics`, valid=200):
- `arithmetic`: `n=72`, `pair_accuracy=1.0`, `gap_p50=0.012525983154773712`
- `retrieval`: `n=65`, `pair_accuracy=1.0`, `gap_p50=0.024283483624458313`
- `constraint`: `n=63`, `pair_accuracy=1.0`, `gap_p50=0.021319493651390076`
- `unknown`: `n=0`

## v0.5.3 Flip Analysis + Format-Aug Training (2026-02-18, no-agent)

### Full-Scale Acceptance Command

```bash
python3 scripts/run_v053_taskmix_acceptance.py \
  --raw_path data/raw/episodes_taskmix_v053_fullscale.jsonl \
  --generate_if_missing true \
  --force_generate true \
  --n_tasks 4000 \
  --rollouts_per_task 3 \
  --family_mix 1:1:1 \
  --success_policy balanced \
  --work_dir data/real_v053 \
  --results_dir results/v053_real_fullscale \
  --max_pairs 1200 \
  --target_train_size 1000 \
  --target_valid_size 200 \
  --format_augmentation concat \
  --counterfactual_warn_threshold 0.80 \
  --counterfactual_block_threshold 0.65 \
  --counterfactual_gate_mode soft \
  --device cpu \
  --amp_dtype none \
  --num_workers 0 \
  --report_path data/real_v053/reports/v053_taskmix_acceptance_fullscale.json
```

### Flip Analysis Command

```bash
python3 scripts/analyze_counterfactual_flips_v053.py \
  --primary_scored_path data/real_v053/reports/valid_scored_pairs_primary_fullscale.jsonl \
  --counterfactual_scored_path data/real_v053/reports/valid_scored_pairs_counterfactual_fullscale.jsonl \
  --report_json data/real_v053/reports/counterfactual_flip_analysis_v053_fullscale.json \
  --report_csv data/real_v053/reports/counterfactual_flip_analysis_v053_fullscale.csv \
  --report_md data/real_v053/reports/counterfactual_flip_analysis_v053_fullscale.md \
  --boundary_bins 0.002,0.01
```

Archived artifacts:
- `data/real_v053/reports/v053_taskmix_acceptance_fullscale.json`
- `data/real_v053/reports/counterfactual_audit_v053_fullscale.json`
- `data/real_v053/reports/counterfactual_flip_analysis_v053_fullscale.json`
- `data/real_v053/reports/counterfactual_flip_analysis_v053_fullscale.csv`
- `data/real_v053/reports/counterfactual_flip_analysis_v053_fullscale.md`
- `data/real_v053/reports/valid_scored_pairs_primary_fullscale.jsonl`
- `data/real_v053/reports/valid_scored_pairs_counterfactual_fullscale.jsonl`

Outcome summary:
- `passed=true`
- `pairs_trainable=1200`
- split exact: `train=1000`, `valid=200`
- `judge_unknown_rate=0.0`
- `env_judge_consistency=1.0`
- shortcut audit: `forbidden_term_hits=0`, `meta_text_overlap_rate=0.0`, `passed=true`
- eval(primary): `pair_accuracy=1.0`, `gap_p50=0.044837988913059235`

Counterfactual thresholds (`soft` mode):
- `warn_threshold=0.80`
- `block_threshold=0.65`
- observed `sign_consistency_rate=0.685`
- advisory: `counterfactual_warn_ok=false`, `counterfactual_block_ok=true`

Flip analysis summary:
- `n_pairs=200`
- `stable_correct_count=137`
- `stable_error_count=0`
- `flip_count=63` (`flip_rate=0.315`)
- `flip_by_family={"constraint":63}`
- `flip_by_neg_type={"replay_corrupt":63}`
- `flip_by_gap_bin={"[0,0.002)":63,"[0.002,0.01)":0,"[0.01,+inf)":0}`

Curriculum summary (`curriculum_mode=stage3`, ratio `40:30:30`):
- stage steps: `800/600/600`
- stage_1 (`truncate_last_k`): `pair_accuracy=1.0`
- stage_2 (`swap_step`): `pair_accuracy=0.87`
- stage_3 (`replay_corrupt`): `pair_accuracy=1.0`

## v0.5.3 A/B Counterfactual Consistency Compare (2026-02-18)

Goal:
- isolate one variable (`format_augmentation`) and answer whether `concat+auto` improves counterfactual sign consistency versus `off+flat`.

Fixed invariants:
- `seed=42`
- `curriculum_mode=stage3`, `curriculum_ratio=40:30:30`
- `input_format=flat`
- `counterfactual_gate_mode=soft`
- thresholds: `warn=0.80`, `block=0.65`
- same raw episodes file for A/B in each scale

### Smoke A/B Commands

Generate smoke raw once:

```bash
python3 scripts/generate_taskmix_episodes_v050.py \
  --out_path data/raw/episodes_taskmix_v053_ab_smoke.jsonl \
  --n_tasks 300 \
  --rollouts_per_task 3 \
  --max_steps 8 \
  --seed 42 \
  --family_mix 1:1:1 \
  --success_policy balanced
```

Run A (baseline: off+flat):

```bash
python3 scripts/run_v053_taskmix_acceptance.py \
  --raw_path data/raw/episodes_taskmix_v053_ab_smoke.jsonl \
  --generate_if_missing false \
  --work_dir data/real_v053_ab/smoke_baseline \
  --results_dir results/v053_ab/smoke_baseline \
  --format_augmentation off \
  --input_format flat \
  --max_pairs 240 \
  --target_train_size 200 \
  --target_valid_size 40 \
  --min_episodes 900 \
  --device cpu \
  --amp_dtype none \
  --num_workers 0 \
  --max_steps 400 \
  --counterfactual_warn_threshold 0.80 \
  --counterfactual_block_threshold 0.65 \
  --counterfactual_gate_mode soft \
  --report_path data/real_v053_ab/smoke_baseline/reports/v053_taskmix_acceptance_smoke_baseline.json
```

Run B (aug: concat+auto):

```bash
python3 scripts/run_v053_taskmix_acceptance.py \
  --raw_path data/raw/episodes_taskmix_v053_ab_smoke.jsonl \
  --generate_if_missing false \
  --work_dir data/real_v053_ab/smoke_aug \
  --results_dir results/v053_ab/smoke_aug \
  --format_augmentation concat \
  --input_format flat \
  --max_pairs 240 \
  --target_train_size 200 \
  --target_valid_size 40 \
  --min_episodes 900 \
  --device cpu \
  --amp_dtype none \
  --num_workers 0 \
  --max_steps 400 \
  --counterfactual_warn_threshold 0.80 \
  --counterfactual_block_threshold 0.65 \
  --counterfactual_gate_mode soft \
  --report_path data/real_v053_ab/smoke_aug/reports/v053_taskmix_acceptance_smoke_aug.json
```

Build smoke compare report:

```bash
python3 scripts/compare_v053_ab_reports.py \
  --baseline_report data/real_v053_ab/smoke_baseline/reports/v053_taskmix_acceptance_smoke_baseline.json \
  --aug_report data/real_v053_ab/smoke_aug/reports/v053_taskmix_acceptance_smoke_aug.json \
  --output_json data/real_v053_ab/reports/v053_ab_compare_smoke.json \
  --label smoke
```

### Full-Scale A/B Commands

Generate full-scale raw once:

```bash
python3 scripts/generate_taskmix_episodes_v050.py \
  --out_path data/raw/episodes_taskmix_v053_ab_fullscale.jsonl \
  --n_tasks 4000 \
  --rollouts_per_task 3 \
  --max_steps 8 \
  --seed 42 \
  --family_mix 1:1:1 \
  --success_policy balanced
```

Run A (baseline: off+flat):

```bash
python3 scripts/run_v053_taskmix_acceptance.py \
  --raw_path data/raw/episodes_taskmix_v053_ab_fullscale.jsonl \
  --generate_if_missing false \
  --work_dir data/real_v053_ab/full_baseline \
  --results_dir results/v053_ab/full_baseline \
  --format_augmentation off \
  --input_format flat \
  --max_pairs 1200 \
  --target_train_size 1000 \
  --target_valid_size 200 \
  --device cpu \
  --amp_dtype none \
  --num_workers 0 \
  --counterfactual_warn_threshold 0.80 \
  --counterfactual_block_threshold 0.65 \
  --counterfactual_gate_mode soft \
  --report_path data/real_v053_ab/full_baseline/reports/v053_taskmix_acceptance_fullscale_baseline.json
```

Run B (aug: concat+auto):

```bash
python3 scripts/run_v053_taskmix_acceptance.py \
  --raw_path data/raw/episodes_taskmix_v053_ab_fullscale.jsonl \
  --generate_if_missing false \
  --work_dir data/real_v053_ab/full_aug \
  --results_dir results/v053_ab/full_aug \
  --format_augmentation concat \
  --input_format flat \
  --max_pairs 1200 \
  --target_train_size 1000 \
  --target_valid_size 200 \
  --device cpu \
  --amp_dtype none \
  --num_workers 0 \
  --counterfactual_warn_threshold 0.80 \
  --counterfactual_block_threshold 0.65 \
  --counterfactual_gate_mode soft \
  --report_path data/real_v053_ab/full_aug/reports/v053_taskmix_acceptance_fullscale_aug.json
```

Build full-scale compare artifacts:

```bash
python3 scripts/compare_v053_ab_reports.py \
  --baseline_report data/real_v053_ab/full_baseline/reports/v053_taskmix_acceptance_fullscale_baseline.json \
  --aug_report data/real_v053_ab/full_aug/reports/v053_taskmix_acceptance_fullscale_aug.json \
  --output_json data/real_v053_ab/reports/v053_ab_compare_fullscale.json \
  --output_md data/real_v053_ab/reports/v053_ab_compare_fullscale.md \
  --label fullscale
```

### Archived A/B Artifacts

- `data/real_v053_ab/reports/v053_ab_compare_smoke.json`
- `data/real_v053_ab/reports/v053_ab_compare_fullscale.json`
- `data/real_v053_ab/reports/v053_ab_compare_fullscale.md`
- `data/real_v053_ab/full_baseline/reports/v053_taskmix_acceptance_fullscale_baseline.json`
- `data/real_v053_ab/full_aug/reports/v053_taskmix_acceptance_fullscale_aug.json`

### Compare Rules and Full-Scale Result

Rules:
- `effective`: `delta_sign_consistency >= 0.03` and `delta_flip_count <= -10` and no hard-metric regression
- `weak_effective`: `delta_sign_consistency >= 0.01` and no hard-metric regression
- `invalid`: `-0.01 < delta_sign_consistency < 0.01`
- `negative`: `delta_sign_consistency <= -0.01` or hard-metric regression

Hard-metric non-regression:
- `passed=true`
- `split_train_exact=true`, `split_valid_exact=true`
- `judge_unknown_rate` non-worse
- `env_judge_consistency` non-worse

Observed full-scale A/B:
- baseline sign consistency: `0.685`
- aug sign consistency: `0.685`
- `delta_sign_consistency=0.0`
- baseline flip count: `63`
- aug flip count: `63`
- `delta_flip_count=0`
- verdict: `无效`
- next step: enter v0.5.4 training constraints (consistency regularization first, format distillation second)

## v0.5.4 A/B Consistency Regularization Compare (2026-02-18)

Goal:
- isolate one variable and test whether training-level consistency regularization (`sign_hinge`) improves counterfactual consistency.

Experiment lock:
- fixed: `seed=42`, `curriculum_mode=stage3`, `curriculum_ratio=40:30:30`, `input_format=flat`, `format_augmentation=concat`, `counterfactual_gate_mode=soft`, `warn=0.80`, `block=0.65`
- only variable: `--consistency_lambda 0.0` (A) vs `--consistency_lambda 0.10` (B)
- fixed scope: `--consistency_apply_scope stage3_only`
- same raw file for each scale (`smoke` and `full-scale`)

### Smoke A/B Commands

Generate smoke raw once:

```bash
python3 scripts/generate_taskmix_episodes_v050.py \
  --out_path data/raw/episodes_taskmix_v054_ab_smoke.jsonl \
  --n_tasks 300 \
  --rollouts_per_task 3 \
  --max_steps 8 \
  --seed 42 \
  --family_mix 1:1:1 \
  --success_policy balanced
```

Run A (baseline: `consistency_lambda=0.0`):

```bash
python3 scripts/run_v054_taskmix_acceptance.py \
  --raw_path data/raw/episodes_taskmix_v054_ab_smoke.jsonl \
  --generate_if_missing false \
  --work_dir data/real_v054_ab/smoke_baseline \
  --results_dir results/v054_ab/smoke_baseline \
  --input_format flat \
  --format_augmentation concat \
  --consistency_lambda 0.0 \
  --consistency_mode sign_hinge \
  --consistency_margin 0.0 \
  --consistency_apply_scope stage3_only \
  --curriculum_mode stage3 \
  --curriculum_ratio 40:30:30 \
  --max_pairs 240 \
  --target_train_size 200 \
  --target_valid_size 40 \
  --min_episodes 900 \
  --max_steps 400 \
  --device cpu \
  --amp_dtype none \
  --num_workers 0 \
  --counterfactual_warn_threshold 0.80 \
  --counterfactual_block_threshold 0.65 \
  --counterfactual_gate_mode soft \
  --report_path data/real_v054_ab/smoke_baseline/reports/v054_taskmix_acceptance_smoke_baseline.json
```

Run B (treatment: `consistency_lambda=0.10`):

```bash
python3 scripts/run_v054_taskmix_acceptance.py \
  --raw_path data/raw/episodes_taskmix_v054_ab_smoke.jsonl \
  --generate_if_missing false \
  --work_dir data/real_v054_ab/smoke_treatment \
  --results_dir results/v054_ab/smoke_treatment \
  --input_format flat \
  --format_augmentation concat \
  --consistency_lambda 0.10 \
  --consistency_mode sign_hinge \
  --consistency_margin 0.0 \
  --consistency_apply_scope stage3_only \
  --curriculum_mode stage3 \
  --curriculum_ratio 40:30:30 \
  --max_pairs 240 \
  --target_train_size 200 \
  --target_valid_size 40 \
  --min_episodes 900 \
  --max_steps 400 \
  --device cpu \
  --amp_dtype none \
  --num_workers 0 \
  --counterfactual_warn_threshold 0.80 \
  --counterfactual_block_threshold 0.65 \
  --counterfactual_gate_mode soft \
  --report_path data/real_v054_ab/smoke_treatment/reports/v054_taskmix_acceptance_smoke_treatment.json
```

Build smoke compare:

```bash
python3 scripts/compare_v053_ab_reports.py \
  --baseline_report data/real_v054_ab/smoke_baseline/reports/v054_taskmix_acceptance_smoke_baseline.json \
  --aug_report data/real_v054_ab/smoke_treatment/reports/v054_taskmix_acceptance_smoke_treatment.json \
  --output_json data/real_v054_ab/reports/v054_ab_compare_smoke.json \
  --label smoke
```

### Full-Scale A/B Commands

Generate full-scale raw once:

```bash
python3 scripts/generate_taskmix_episodes_v050.py \
  --out_path data/raw/episodes_taskmix_v054_ab_fullscale.jsonl \
  --n_tasks 4000 \
  --rollouts_per_task 3 \
  --max_steps 8 \
  --seed 42 \
  --family_mix 1:1:1 \
  --success_policy balanced
```

Run A (baseline: `consistency_lambda=0.0`):

```bash
python3 scripts/run_v054_taskmix_acceptance.py \
  --raw_path data/raw/episodes_taskmix_v054_ab_fullscale.jsonl \
  --generate_if_missing false \
  --work_dir data/real_v054_ab/full_baseline \
  --results_dir results/v054_ab/full_baseline \
  --input_format flat \
  --format_augmentation concat \
  --consistency_lambda 0.0 \
  --consistency_mode sign_hinge \
  --consistency_margin 0.0 \
  --consistency_apply_scope stage3_only \
  --curriculum_mode stage3 \
  --curriculum_ratio 40:30:30 \
  --max_pairs 1200 \
  --target_train_size 1000 \
  --target_valid_size 200 \
  --device cpu \
  --amp_dtype none \
  --num_workers 0 \
  --counterfactual_warn_threshold 0.80 \
  --counterfactual_block_threshold 0.65 \
  --counterfactual_gate_mode soft \
  --report_path data/real_v054_ab/full_baseline/reports/v054_taskmix_acceptance_fullscale_baseline.json
```

Run B (treatment: `consistency_lambda=0.10`):

```bash
python3 scripts/run_v054_taskmix_acceptance.py \
  --raw_path data/raw/episodes_taskmix_v054_ab_fullscale.jsonl \
  --generate_if_missing false \
  --work_dir data/real_v054_ab/full_treatment \
  --results_dir results/v054_ab/full_treatment \
  --input_format flat \
  --format_augmentation concat \
  --consistency_lambda 0.10 \
  --consistency_mode sign_hinge \
  --consistency_margin 0.0 \
  --consistency_apply_scope stage3_only \
  --curriculum_mode stage3 \
  --curriculum_ratio 40:30:30 \
  --max_pairs 1200 \
  --target_train_size 1000 \
  --target_valid_size 200 \
  --device cpu \
  --amp_dtype none \
  --num_workers 0 \
  --counterfactual_warn_threshold 0.80 \
  --counterfactual_block_threshold 0.65 \
  --counterfactual_gate_mode soft \
  --report_path data/real_v054_ab/full_treatment/reports/v054_taskmix_acceptance_fullscale_treatment.json
```

Build full-scale compare artifacts:

```bash
python3 scripts/compare_v053_ab_reports.py \
  --baseline_report data/real_v054_ab/full_baseline/reports/v054_taskmix_acceptance_fullscale_baseline.json \
  --aug_report data/real_v054_ab/full_treatment/reports/v054_taskmix_acceptance_fullscale_treatment.json \
  --output_json data/real_v054_ab/reports/v054_ab_compare_fullscale.json \
  --output_md data/real_v054_ab/reports/v054_ab_compare_fullscale.md \
  --label fullscale
```

### Archived A/B Artifacts

- `data/real_v054_ab/reports/v054_ab_compare_smoke.json`
- `data/real_v054_ab/reports/v054_ab_compare_fullscale.json`
- `data/real_v054_ab/reports/v054_ab_compare_fullscale.md`
- `data/real_v054_ab/full_baseline/reports/v054_taskmix_acceptance_fullscale_baseline.json`
- `data/real_v054_ab/full_treatment/reports/v054_taskmix_acceptance_fullscale_treatment.json`

### Compare Rules and Observed Result

Rules (unchanged from v0.5.3):
- `effective`: `delta_sign_consistency >= 0.03` and `delta_flip_count <= -10` and no hard-metric regression
- `weak_effective`: `delta_sign_consistency >= 0.01` and no hard-metric regression
- `invalid`: `-0.01 < delta_sign_consistency < 0.01`
- `negative`: `delta_sign_consistency <= -0.01` or hard-metric regression

Smoke result:
- baseline `sign_consistency_rate=0.65`, treatment `sign_consistency_rate=0.65`
- `delta_sign_consistency=0.0`, `delta_flip_count=0`
- verdict: `无效`

Full-scale result:
- baseline `sign_consistency_rate=0.685`, treatment `sign_consistency_rate=0.685`
- `delta_sign_consistency=0.0`
- baseline `flip_count=63`, treatment `flip_count=63`, `delta_flip_count=0`
- hard metrics non-regression: pass (`passed=true`, split exact, `judge_unknown_rate` non-worse, `env_judge_consistency` non-worse)
- verdict: `无效`

Consistency activation evidence (treatment, stage_3):
- `consistency_lambda=0.1`
- `consistency_pairs=1000`
- `consistency_loss=0.00017727833437675145`

Decision:
- `sign_hinge` regularization (`lambda=0.10`, `stage3_only`) is active but does not improve full-scale counterfactual consistency under current setup.
- next step: move to v0.5.5 training constraints (format distillation / stronger consistency objective), not more data-only variants.

## v0.5.5 A/B gap_l2 + Smoke Gate (2026-02-18)

Goal:
- keep v0.5.4 pipeline/compare policy unchanged and test whether switching treatment consistency objective to `gap_l2` improves counterfactual sign consistency.
- add training diagnostics to verify whether consistency pressure is actually active on paired views.

### Code Changes

- `rm/train.py` now writes extra consistency diagnostics to `metrics.jsonl`:
  - `consistency_violation_nonpos_rate`
  - `consistency_gap_abs_mean`
  - `consistency_gap_abs_p50`
  - `consistency_gap_abs_p90`
  - `consistency_groups_used`
- new runner: `scripts/run_v055_taskmix_acceptance.py`
  - defaults changed to v055 paths
  - default `--consistency_mode gap_l2`
  - keeps v0.5.4 gate semantics

### Experiment Lock

- fixed: `seed=42`, `curriculum_mode=stage3`, `curriculum_ratio=40:30:30`, `format_augmentation=concat`, `input_format=flat`, counterfactual gate `soft`, thresholds `warn=0.80`, `block=0.65`
- only variable:
  - A baseline: `consistency_lambda=0.0`
  - B treatment: `consistency_lambda=0.10`, `consistency_mode=gap_l2`
- smoke gate rule:
  - must satisfy `non_regression.all_ok=true`
  - and verdict in `{弱有效, 有效}`
  - otherwise do not run full-scale

### Smoke A/B Commands

Generate smoke raw once:

```bash
python3 scripts/generate_taskmix_episodes_v050.py \
  --out_path data/raw/episodes_taskmix_v055_ab_smoke.jsonl \
  --n_tasks 300 \
  --rollouts_per_task 3 \
  --max_steps 8 \
  --seed 42 \
  --family_mix 1:1:1 \
  --success_policy balanced
```

Run A (baseline):

```bash
python3 scripts/run_v055_taskmix_acceptance.py \
  --raw_path data/raw/episodes_taskmix_v055_ab_smoke.jsonl \
  --generate_if_missing false \
  --work_dir data/real_v055_ab/smoke_baseline \
  --results_dir results/v055_ab/smoke_baseline \
  --input_format flat \
  --format_augmentation concat \
  --consistency_lambda 0.0 \
  --consistency_mode gap_l2 \
  --consistency_margin 0.0 \
  --consistency_apply_scope stage3_only \
  --curriculum_mode stage3 \
  --curriculum_ratio 40:30:30 \
  --max_pairs 240 \
  --target_train_size 200 \
  --target_valid_size 40 \
  --min_episodes 900 \
  --max_steps 400 \
  --device cpu \
  --amp_dtype none \
  --num_workers 0 \
  --counterfactual_warn_threshold 0.80 \
  --counterfactual_block_threshold 0.65 \
  --counterfactual_gate_mode soft \
  --report_path data/real_v055_ab/smoke_baseline/reports/v055_taskmix_acceptance_smoke_baseline.json
```

Run B (treatment):

```bash
python3 scripts/run_v055_taskmix_acceptance.py \
  --raw_path data/raw/episodes_taskmix_v055_ab_smoke.jsonl \
  --generate_if_missing false \
  --work_dir data/real_v055_ab/smoke_treatment \
  --results_dir results/v055_ab/smoke_treatment \
  --input_format flat \
  --format_augmentation concat \
  --consistency_lambda 0.10 \
  --consistency_mode gap_l2 \
  --consistency_margin 0.0 \
  --consistency_apply_scope stage3_only \
  --curriculum_mode stage3 \
  --curriculum_ratio 40:30:30 \
  --max_pairs 240 \
  --target_train_size 200 \
  --target_valid_size 40 \
  --min_episodes 900 \
  --max_steps 400 \
  --device cpu \
  --amp_dtype none \
  --num_workers 0 \
  --counterfactual_warn_threshold 0.80 \
  --counterfactual_block_threshold 0.65 \
  --counterfactual_gate_mode soft \
  --report_path data/real_v055_ab/smoke_treatment/reports/v055_taskmix_acceptance_smoke_treatment.json
```

Build smoke compare:

```bash
python3 scripts/compare_v053_ab_reports.py \
  --baseline_report data/real_v055_ab/smoke_baseline/reports/v055_taskmix_acceptance_smoke_baseline.json \
  --aug_report data/real_v055_ab/smoke_treatment/reports/v055_taskmix_acceptance_smoke_treatment.json \
  --output_json data/real_v055_ab/reports/v055_ab_compare_smoke.json \
  --label smoke
```

### Smoke Gate Decision (executed)

Decision artifact:
- `data/real_v055_ab/reports/v055_smoke_gate_decision.json`

Observed smoke A/B:
- `delta_sign_consistency=0.0`
- `delta_flip_count=0`
- compare verdict: `无效`
- `non_regression.all_ok=true`
- gate result: `proceed_fullscale=false`

Treatment stage_3 diagnostics (from `consistency_train_metrics_by_stage.stage_3`):
- `consistency_lambda=0.1`
- `consistency_mode=gap_l2`
- `consistency_pairs=200`
- `consistency_loss=0.00021737196948379278`
- `consistency_violation_nonpos_rate=0.16`
- `consistency_gap_abs_mean=0.010916537679731846`
- `consistency_gap_abs_p50=0.00936073251068592`
- `consistency_gap_abs_p90=0.021951111778616906`
- `consistency_groups_used=200`

### Full-Scale Commands (conditional, not executed in this run)

Run only if smoke gate passes (`proceed_fullscale=true`):

```bash
python3 scripts/generate_taskmix_episodes_v050.py \
  --out_path data/raw/episodes_taskmix_v055_ab_fullscale.jsonl \
  --n_tasks 4000 \
  --rollouts_per_task 3 \
  --max_steps 8 \
  --seed 42 \
  --family_mix 1:1:1 \
  --success_policy balanced
```

```bash
python3 scripts/run_v055_taskmix_acceptance.py \
  --raw_path data/raw/episodes_taskmix_v055_ab_fullscale.jsonl \
  --generate_if_missing false \
  --work_dir data/real_v055_ab/full_baseline \
  --results_dir results/v055_ab/full_baseline \
  --input_format flat \
  --format_augmentation concat \
  --consistency_lambda 0.0 \
  --consistency_mode gap_l2 \
  --consistency_apply_scope stage3_only \
  --curriculum_mode stage3 \
  --curriculum_ratio 40:30:30 \
  --max_pairs 1200 \
  --target_train_size 1000 \
  --target_valid_size 200 \
  --device cpu \
  --amp_dtype none \
  --num_workers 0 \
  --counterfactual_warn_threshold 0.80 \
  --counterfactual_block_threshold 0.65 \
  --counterfactual_gate_mode soft \
  --report_path data/real_v055_ab/full_baseline/reports/v055_taskmix_acceptance_fullscale_baseline.json
```

```bash
python3 scripts/run_v055_taskmix_acceptance.py \
  --raw_path data/raw/episodes_taskmix_v055_ab_fullscale.jsonl \
  --generate_if_missing false \
  --work_dir data/real_v055_ab/full_treatment \
  --results_dir results/v055_ab/full_treatment \
  --input_format flat \
  --format_augmentation concat \
  --consistency_lambda 0.10 \
  --consistency_mode gap_l2 \
  --consistency_apply_scope stage3_only \
  --curriculum_mode stage3 \
  --curriculum_ratio 40:30:30 \
  --max_pairs 1200 \
  --target_train_size 1000 \
  --target_valid_size 200 \
  --device cpu \
  --amp_dtype none \
  --num_workers 0 \
  --counterfactual_warn_threshold 0.80 \
  --counterfactual_block_threshold 0.65 \
  --counterfactual_gate_mode soft \
  --report_path data/real_v055_ab/full_treatment/reports/v055_taskmix_acceptance_fullscale_treatment.json
```

```bash
python3 scripts/compare_v053_ab_reports.py \
  --baseline_report data/real_v055_ab/full_baseline/reports/v055_taskmix_acceptance_fullscale_baseline.json \
  --aug_report data/real_v055_ab/full_treatment/reports/v055_taskmix_acceptance_fullscale_treatment.json \
  --output_json data/real_v055_ab/reports/v055_ab_compare_fullscale.json \
  --output_md data/real_v055_ab/reports/v055_ab_compare_fullscale.md \
  --label fullscale
```

## v0.5.6 A/B gap_l2 + Flip Hard-Mining + Strict Smoke Gate (2026-02-18)

Goal:
- keep v0.5.5 compare rules and gate semantics unchanged.
- add stage-3 targeted hard-mining so consistency training focuses on flip-prone groups.

### Code Changes

- new flip mining script: `scripts/mine_flip_groups_v056.py`
  - inputs: `primary_scored_path`, `counterfactual_scored_path`, `source_pairs_path`
  - outputs: `flip_groups_v056.json` + report with `n_rows/flip_rows/flip_groups/flip_rate`
- new hard-mined train builder: `scripts/build_hardmined_train_v056.py`
  - input: `train.format_aug.jsonl` + `flip_group_ids`
  - output: `train.format_aug.hardmined.jsonl` (`rows_out == rows_in`, group-paired)
- new runner: `scripts/run_v056_taskmix_acceptance.py`
  - defaults: `hard_mining_mode=flip80`, `hard_mining_flip_fraction=0.80`, `hard_mining_mining_source=stage2_ckpt`
  - hard-mining applies to stage_3 only (default scope alignment with consistency regularization)

### Experiment Lock

- fixed: `seed=42`, `curriculum_mode=stage3`, `curriculum_ratio=40:30:30`, `format_augmentation=concat`, `input_format=flat`, counterfactual gate `soft`, thresholds `warn=0.80` / `block=0.65`
- only variable:
  - A baseline: `consistency_lambda=0.0`, `hard_mining_mode=off`
  - B treatment: `consistency_lambda=0.10`, `consistency_mode=gap_l2`, `hard_mining_mode=flip80`
- smoke gate rule:
  - must satisfy `non_regression.all_ok=true`
  - and verdict in `{弱有效, 有效}`
  - otherwise stop (`proceed_fullscale=false`)

### Smoke A/B Commands

Generate smoke raw once:

```bash
python3 scripts/generate_taskmix_episodes_v050.py \
  --out_path data/raw/episodes_taskmix_v056_ab_smoke.jsonl \
  --n_tasks 300 \
  --rollouts_per_task 3 \
  --max_steps 8 \
  --seed 42 \
  --family_mix 1:1:1 \
  --success_policy balanced
```

Run A (baseline):

```bash
python3 scripts/run_v056_taskmix_acceptance.py \
  --raw_path data/raw/episodes_taskmix_v056_ab_smoke.jsonl \
  --generate_if_missing false \
  --work_dir data/real_v056_ab/smoke_baseline \
  --results_dir results/v056_ab/smoke_baseline \
  --input_format flat \
  --format_augmentation concat \
  --consistency_lambda 0.0 \
  --consistency_mode gap_l2 \
  --consistency_apply_scope stage3_only \
  --hard_mining_mode off \
  --curriculum_mode stage3 \
  --curriculum_ratio 40:30:30 \
  --max_pairs 240 \
  --target_train_size 200 \
  --target_valid_size 40 \
  --min_episodes 900 \
  --max_steps 400 \
  --device cpu \
  --amp_dtype none \
  --num_workers 0 \
  --counterfactual_warn_threshold 0.80 \
  --counterfactual_block_threshold 0.65 \
  --counterfactual_gate_mode soft \
  --report_path data/real_v056_ab/smoke_baseline/reports/v056_taskmix_acceptance_smoke_baseline.json
```

Run B (treatment):

```bash
python3 scripts/run_v056_taskmix_acceptance.py \
  --raw_path data/raw/episodes_taskmix_v056_ab_smoke.jsonl \
  --generate_if_missing false \
  --work_dir data/real_v056_ab/smoke_treatment \
  --results_dir results/v056_ab/smoke_treatment \
  --input_format flat \
  --format_augmentation concat \
  --consistency_lambda 0.10 \
  --consistency_mode gap_l2 \
  --consistency_apply_scope stage3_only \
  --hard_mining_mode flip80 \
  --hard_mining_flip_fraction 0.80 \
  --hard_mining_mining_source stage2_ckpt \
  --curriculum_mode stage3 \
  --curriculum_ratio 40:30:30 \
  --max_pairs 240 \
  --target_train_size 200 \
  --target_valid_size 40 \
  --min_episodes 900 \
  --max_steps 400 \
  --device cpu \
  --amp_dtype none \
  --num_workers 0 \
  --counterfactual_warn_threshold 0.80 \
  --counterfactual_block_threshold 0.65 \
  --counterfactual_gate_mode soft \
  --report_path data/real_v056_ab/smoke_treatment/reports/v056_taskmix_acceptance_smoke_treatment.json
```

Build smoke compare:

```bash
python3 scripts/compare_v053_ab_reports.py \
  --baseline_report data/real_v056_ab/smoke_baseline/reports/v056_taskmix_acceptance_smoke_baseline.json \
  --aug_report data/real_v056_ab/smoke_treatment/reports/v056_taskmix_acceptance_smoke_treatment.json \
  --output_json data/real_v056_ab/reports/v056_ab_compare_smoke.json \
  --label v0.5.6-smoke
```

### Smoke Gate Decision (executed)

Decision artifact:
- `data/real_v056_ab/reports/v056_smoke_gate_decision.json`

Observed smoke A/B:
- `delta_sign_consistency=0.0`
- `delta_flip_count=0`
- compare verdict: `无效`
- `non_regression.all_ok=true`
- gate result: `proceed_fullscale=false`

Treatment hard-mining stage_3 artifacts:
- `data/real_v056_ab/smoke_treatment/stages/stage_3/reports/flip_groups_v056.json`
- `data/real_v056_ab/smoke_treatment/stages/stage_3/reports/hard_mining_v056.json`
- `data/real_v056_ab/smoke_treatment/stages/stage_3/train.format_aug.hardmined.jsonl`

Treatment stage_3 diagnostics:
- `consistency_lambda=0.1`
- `consistency_mode=gap_l2`
- `consistency_pairs=200`
- `consistency_loss=0.00021036774342064746`
- `consistency_violation_nonpos_rate=0.205`
- `consistency_gap_abs_mean=0.012153038457036018`
- `consistency_gap_abs_p90=0.02231992147862911`

### Full-Scale Commands (conditional)

Run only if smoke gate passes (`proceed_fullscale=true`):

```bash
python3 scripts/generate_taskmix_episodes_v050.py \
  --out_path data/raw/episodes_taskmix_v056_ab_fullscale.jsonl \
  --n_tasks 4000 \
  --rollouts_per_task 3 \
  --max_steps 8 \
  --seed 42 \
  --family_mix 1:1:1 \
  --success_policy balanced
```

```bash
python3 scripts/run_v056_taskmix_acceptance.py \
  --raw_path data/raw/episodes_taskmix_v056_ab_fullscale.jsonl \
  --generate_if_missing false \
  --work_dir data/real_v056_ab/full_baseline \
  --results_dir results/v056_ab/full_baseline \
  --input_format flat \
  --format_augmentation concat \
  --consistency_lambda 0.0 \
  --consistency_mode gap_l2 \
  --consistency_apply_scope stage3_only \
  --hard_mining_mode off \
  --curriculum_mode stage3 \
  --curriculum_ratio 40:30:30 \
  --max_pairs 1200 \
  --target_train_size 1000 \
  --target_valid_size 200 \
  --device cpu \
  --amp_dtype none \
  --num_workers 0 \
  --counterfactual_warn_threshold 0.80 \
  --counterfactual_block_threshold 0.65 \
  --counterfactual_gate_mode soft \
  --report_path data/real_v056_ab/full_baseline/reports/v056_taskmix_acceptance_fullscale_baseline.json
```

```bash
python3 scripts/run_v056_taskmix_acceptance.py \
  --raw_path data/raw/episodes_taskmix_v056_ab_fullscale.jsonl \
  --generate_if_missing false \
  --work_dir data/real_v056_ab/full_treatment \
  --results_dir results/v056_ab/full_treatment \
  --input_format flat \
  --format_augmentation concat \
  --consistency_lambda 0.10 \
  --consistency_mode gap_l2 \
  --consistency_apply_scope stage3_only \
  --hard_mining_mode flip80 \
  --hard_mining_flip_fraction 0.80 \
  --hard_mining_mining_source stage2_ckpt \
  --curriculum_mode stage3 \
  --curriculum_ratio 40:30:30 \
  --max_pairs 1200 \
  --target_train_size 1000 \
  --target_valid_size 200 \
  --device cpu \
  --amp_dtype none \
  --num_workers 0 \
  --counterfactual_warn_threshold 0.80 \
  --counterfactual_block_threshold 0.65 \
  --counterfactual_gate_mode soft \
  --report_path data/real_v056_ab/full_treatment/reports/v056_taskmix_acceptance_fullscale_treatment.json
```

```bash
python3 scripts/compare_v053_ab_reports.py \
  --baseline_report data/real_v056_ab/full_baseline/reports/v056_taskmix_acceptance_fullscale_baseline.json \
  --aug_report data/real_v056_ab/full_treatment/reports/v056_taskmix_acceptance_fullscale_treatment.json \
  --output_json data/real_v056_ab/reports/v056_ab_compare_fullscale.json \
  --output_md data/real_v056_ab/reports/v056_ab_compare_fullscale.md \
  --label v0.5.6-fullscale
```

## v0.5.7 A/B Matrix + Strict Smoke Gate (2026-02-18)

Goal:
- keep compare/gate semantics unchanged.
- run a 2x2 hard-mining matrix on smoke:
  - `flip_fraction={0.5,1.0}`
  - `sampling_mode={with_replacement,without_replacement}`
- use mining split as default mining source and promote only best treatment to full-scale if strict smoke gate passes.

### Code Changes

- new hard-mined builder with sampling mode:
  - `scripts/build_hardmined_train_v057.py`
  - output-side group ids are rewritten per sampled pair (`meta.format_group_id`) and source id is kept in `meta.format_group_source_id`, so each output group remains exactly `flat+stepwise` even under `with_replacement`.
- new mining split builder:
  - `scripts/build_mining_split_v057.py`
- new runner:
  - `scripts/run_v057_taskmix_acceptance.py`
  - new args:
    - `--hard_mining_sampling_mode {with_replacement,without_replacement}`
    - `--hard_mining_mining_data_source {train_aug,mining_split}` (default `mining_split`)
    - `--hard_mining_mining_group_fraction` (default `0.20`)
    - `--hard_mining_matrix` (default `0.5:with_replacement,0.5:without_replacement,1.0:with_replacement,1.0:without_replacement`)
- new matrix summarizer:
  - `scripts/summarize_v057_matrix.py`
- backward-compatible enhancement:
  - `scripts/mine_flip_groups_v056.py` now also exports `flip_signatures` and `flip_signature_counts`

### Smoke Raw Command (shared)

```bash
python3 scripts/generate_taskmix_episodes_v050.py \
  --out_path data/raw/episodes_taskmix_v057_ab_smoke.jsonl \
  --n_tasks 300 \
  --rollouts_per_task 3 \
  --max_steps 8 \
  --seed 42 \
  --family_mix 1:1:1 \
  --success_policy balanced
```

### Smoke Runs (executed)

Baseline:

```bash
python3 scripts/run_v057_taskmix_acceptance.py \
  --raw_path data/raw/episodes_taskmix_v057_ab_smoke.jsonl \
  --generate_if_missing false \
  --work_dir data/real_v057_ab/smoke_baseline \
  --results_dir results/v057_ab/smoke_baseline \
  --consistency_lambda 0.0 \
  --hard_mining_mode off \
  --report_path data/real_v057_ab/smoke_baseline/reports/v057_taskmix_acceptance_smoke_baseline.json
```

Treatments:
- `f05_wr`: `flip_fraction=0.5`, `sampling_mode=with_replacement`
- `f05_wor`: `flip_fraction=0.5`, `sampling_mode=without_replacement`
- `f10_wr`: `flip_fraction=1.0`, `sampling_mode=with_replacement`
- `f10_wor`: `flip_fraction=1.0`, `sampling_mode=without_replacement`

All treatment runs keep:
- `consistency_lambda=0.10`
- `consistency_mode=gap_l2`
- `hard_mining_mode=flip80`
- `hard_mining_mining_data_source=mining_split`
- `hard_mining_mining_group_fraction=0.20`

### Compare + Matrix Summary (executed)

Per-arm compare outputs:
- `data/real_v057_ab/reports/v057_compare_smoke_f05_wr.json`
- `data/real_v057_ab/reports/v057_compare_smoke_f05_wor.json`
- `data/real_v057_ab/reports/v057_compare_smoke_f10_wr.json`
- `data/real_v057_ab/reports/v057_compare_smoke_f10_wor.json`

Matrix summary outputs:
- `data/real_v057_ab/reports/v057_smoke_matrix_summary.json`
- `data/real_v057_ab/reports/v057_smoke_matrix_summary.md`
- `data/real_v057_ab/reports/v057_smoke_gate_decision.json`

### Observed Smoke Results

Baseline:
- `passed=true`
- `sign_consistency_rate=0.65`
- `flip_count=14`

All 4 treatments:
- compare verdict: `无效`
- `delta_sign_consistency=0.0`
- `delta_flip_count=0`
- `non_regression.all_ok=true`

Matrix gate:
- `best_arm=f05_wor`
- `best_verdict=无效`
- `proceed_fullscale=false`
- reason: `strict_smoke_gate_not_met`

Stage_3 hard-mining reports:
- `actual_flip_fraction=1.0` for all 4 treatments
- `flip_selection_source=signature_match`

### Full-Scale Commands (conditional)

Run only if `data/real_v057_ab/reports/v057_smoke_gate_decision.json` has `proceed_fullscale=true`:

```bash
python3 scripts/generate_taskmix_episodes_v050.py \
  --out_path data/raw/episodes_taskmix_v057_ab_fullscale.jsonl \
  --n_tasks 4000 \
  --rollouts_per_task 3 \
  --max_steps 8 \
  --seed 42 \
  --family_mix 1:1:1 \
  --success_policy balanced
```

- then run:
  - `data/real_v057_ab/full_baseline/`
  - `data/real_v057_ab/full_treat_best_<arm>/`
- and compare:
  - `data/real_v057_ab/reports/v057_compare_fullscale_best.json`
  - `data/real_v057_ab/reports/v057_compare_fullscale_best.md`

## v0.5.8 Semantic-Equivalent Counterfactual Audit + Strict Smoke Gate (2026-02-18)

Goal:
- freeze v0.5.7 training path (`gap_l2 + stage3 hard-mining 2x2`) and only change counterfactual audit construction.
- replace `flat vs stepwise` with semantic-equivalent `stepwise(xml_v1) vs stepwise(marker_v2)`.
- keep compare/verdict and strict smoke-gate policy unchanged.

### Code Changes

- stepwise formatter style support:
  - `rm/formatters/stepwise.py`
  - styles: `xml_v1` (default backward-compatible), `marker_v2` (marker-only variant)
- collator/eval style pass-through:
  - `rm/data.py` (`PairCollator(stepwise_style=...)`)
  - `rm/eval.py` (`--stepwise_style {xml_v1,marker_v2}`)
- new semantic counterfactual audit:
  - `scripts/audit_counterfactual_semantic_v058.py`
  - adds `gap_delta_mean/p50/p90` while keeping `sign_consistency_rate` and `sign_flip_count`
- new v0.5.8 runner:
  - `scripts/run_v058_taskmix_acceptance.py`
- new matrix summarizer:
  - `scripts/summarize_v058_matrix.py`

### Smoke Raw Command (shared)

```bash
python3 scripts/generate_taskmix_episodes_v050.py \
  --out_path data/raw/episodes_taskmix_v058_ab_smoke.jsonl \
  --n_tasks 300 \
  --rollouts_per_task 3 \
  --max_steps 8 \
  --seed 42 \
  --family_mix 1:1:1 \
  --success_policy balanced
```

### Smoke Runs (executed)

Baseline:

```bash
python3 scripts/run_v058_taskmix_acceptance.py \
  --raw_path data/raw/episodes_taskmix_v058_ab_smoke.jsonl \
  --generate_if_missing false \
  --work_dir data/real_v058_ab/smoke_baseline \
  --results_dir results/v058_ab/smoke_baseline \
  --report_path data/real_v058_ab/smoke_baseline/reports/v058_taskmix_acceptance_smoke_baseline.json \
  --consistency_lambda 0.0 \
  --hard_mining_mode off \
  --primary_format stepwise \
  --primary_stepwise_style xml_v1 \
  --counterfactual_format stepwise \
  --counterfactual_stepwise_style marker_v2 \
  --counterfactual_gate_mode soft \
  --counterfactual_warn_threshold 0.80 \
  --counterfactual_block_threshold 0.65 \
  --max_pairs 240 \
  --target_train_size 200 \
  --target_valid_size 40 \
  --min_episodes 900 \
  --device cpu \
  --amp_dtype none \
  --num_workers 0 \
  --max_steps 400
```

Treatments (2x2 matrix):
- `f05_wr`: `flip_fraction=0.5`, `sampling_mode=with_replacement`
- `f05_wor`: `flip_fraction=0.5`, `sampling_mode=without_replacement`
- `f10_wr`: `flip_fraction=1.0`, `sampling_mode=with_replacement`
- `f10_wor`: `flip_fraction=1.0`, `sampling_mode=without_replacement`

All treatment runs keep:
- `consistency_lambda=0.10`
- `consistency_mode=gap_l2`
- `hard_mining_mode=flip80`
- `hard_mining_mining_data_source=mining_split`
- `hard_mining_mining_group_fraction=0.20`
- semantic counterfactual config:
  - primary `stepwise/xml_v1`
  - counterfactual `stepwise/marker_v2`

### Compare + Matrix Summary (executed)

Per-arm compare outputs:
- `data/real_v058_ab/reports/v058_compare_smoke_f05_wr.json`
- `data/real_v058_ab/reports/v058_compare_smoke_f05_wor.json`
- `data/real_v058_ab/reports/v058_compare_smoke_f10_wr.json`
- `data/real_v058_ab/reports/v058_compare_smoke_f10_wor.json`

Matrix summary outputs:
- `data/real_v058_ab/reports/v058_smoke_matrix_summary.json`
- `data/real_v058_ab/reports/v058_smoke_matrix_summary.md`
- `data/real_v058_ab/reports/v058_smoke_gate_decision.json`

### Observed Smoke Results

Baseline:
- `passed=true`
- `sign_consistency_rate=1.0`
- `flip_count=0`
- semantic gap-delta: `gap_delta_mean=0.0008644227869808674`, `gap_delta_p50=0.000822756439447403`, `gap_delta_p90=0.002008615434169769`

All 4 treatments:
- compare verdict: `无效`
- `delta_sign_consistency=0.0`
- `delta_flip_count=0`
- `non_regression.all_ok=true`

Matrix gate:
- `best_arm=f05_wor`
- `best_verdict=无效`
- `proceed_fullscale=false`
- reason: `strict_smoke_gate_not_met`

Interpretation:
- with semantic-equivalent counterfactual construction, sign flips on smoke collapse to zero for both baseline and all treatments.
- strict smoke-gate remains unchanged; because best verdict is not in `{弱有效, 有效}`, this run is `smoke-gated stop`.

### Full-Scale Commands (conditional)

Run only if `data/real_v058_ab/reports/v058_smoke_gate_decision.json` has `proceed_fullscale=true`:

```bash
python3 scripts/generate_taskmix_episodes_v050.py \
  --out_path data/raw/episodes_taskmix_v058_ab_fullscale.jsonl \
  --n_tasks 4000 \
  --rollouts_per_task 3 \
  --max_steps 8 \
  --seed 42 \
  --family_mix 1:1:1 \
  --success_policy balanced
```

- then run:
  - `data/real_v058_ab/full_baseline/`
  - `data/real_v058_ab/full_treat_best_<arm>/`
- and compare:
  - `data/real_v058_ab/reports/v058_compare_fullscale_best.json`
  - `data/real_v058_ab/reports/v058_compare_fullscale_best.md`

## v0.5.8.1 Dual-Gate (Sign + Continuous) Matrix (2026-02-18)

Goal:
- keep v0.5.8 training pipeline frozen (`run_v058_taskmix_acceptance.py`).
- keep v053 sign compare logic intact while adding a parallel continuous gate based on:
  - `delta_gap_delta_p90_rel`
  - `delta_gap_delta_p50_rel`
- use final dual-gate verdict for strict smoke promotion.

### Code Changes

- new dual-gate compare:
  - `scripts/compare_v060_ab_reports.py`
  - added fields:
    - `deltas.delta_gap_delta_p90_rel`
    - `deltas.delta_gap_delta_p50_rel`
    - `continuous_checks.p50_non_worse`
    - `verdict_sign`
    - `verdict_continuous`
    - `verdict_final`
    - `verdict_path`
- new dual-gate matrix summarizer:
  - `scripts/summarize_v060_matrix.py`
  - ranking:
    - `verdict_final_score desc`
    - `delta_gap_delta_p90_rel desc`
    - `delta_sign_consistency desc`
    - `delta_flip_count asc`
- new orchestrator:
  - `scripts/run_v0581_matrix_ab.py`
  - runs smoke baseline + 4 treatments, per-arm dual-gate compare, matrix summary, and conditional full-scale.

### Smoke Matrix Command (executed)

```bash
python3 scripts/run_v0581_matrix_ab.py
```

Defaults used:
- smoke raw: `data/raw/episodes_taskmix_v0581_ab_smoke.jsonl`
- smoke scale: `n_tasks=300`, `rollouts_per_task=3`, `max_pairs=240`, split `200/40`
- training: `gap_l2`, `consistency_lambda={0.0 baseline, 0.10 treatment}`, `stage3_only`
- counterfactual audit: `stepwise/xml_v1` vs `stepwise/marker_v2`
- dual-gate thresholds:
  - `p90_weak_threshold=0.10`
  - `p90_effective_threshold=0.20`
  - `p50_tolerance=0.02`

### Smoke Artifacts

- baseline report:
  - `data/real_v0581_ab/smoke_baseline/reports/v0581_taskmix_acceptance_smoke_baseline.json`
- treatment reports:
  - `data/real_v0581_ab/smoke_treat_f05_wr/reports/v0581_taskmix_acceptance_smoke_treat_f05_wr.json`
  - `data/real_v0581_ab/smoke_treat_f05_wor/reports/v0581_taskmix_acceptance_smoke_treat_f05_wor.json`
  - `data/real_v0581_ab/smoke_treat_f10_wr/reports/v0581_taskmix_acceptance_smoke_treat_f10_wr.json`
  - `data/real_v0581_ab/smoke_treat_f10_wor/reports/v0581_taskmix_acceptance_smoke_treat_f10_wor.json`
- per-arm dual-gate compare:
  - `data/real_v0581_ab/reports/v0581_compare_smoke_f05_wr.json`
  - `data/real_v0581_ab/reports/v0581_compare_smoke_f05_wor.json`
  - `data/real_v0581_ab/reports/v0581_compare_smoke_f10_wr.json`
  - `data/real_v0581_ab/reports/v0581_compare_smoke_f10_wor.json`
- matrix summary:
  - `data/real_v0581_ab/reports/v0581_smoke_matrix_summary.json`
  - `data/real_v0581_ab/reports/v0581_smoke_matrix_summary.md`
  - `data/real_v0581_ab/reports/v0581_smoke_gate_decision.json`

### Observed Smoke Results

Baseline:
- `passed=true`
- `sign_consistency_rate=1.0`
- `flip_count=0`
- `gap_delta_p50=0.000822756439447403`
- `gap_delta_p90=0.002008615434169769`

All 4 treatments:
- `verdict_sign=无效`
- `verdict_continuous=无效`
- `verdict_final=无效`
- `delta_sign_consistency=0.0`
- `delta_flip_count=0`
- `non_regression.all_ok=true`

Best arm and strict gate:
- `best_arm=f05_wr`
- `best_verdict_final=无效`
- `proceed_fullscale=false`
- reason: `strict_smoke_gate_not_met`

Interpretation:
- dual-gate compare runs correctly and adds continuous evidence, but under current semantic-equivalent counterfactual setup all arms remain below weak-effective thresholds.
- policy outcome remains `smoke-gated stop`, so no full-scale run is executed.

### Full-Scale (conditional)

Only when `data/real_v0581_ab/reports/v0581_smoke_gate_decision.json` has `proceed_fullscale=true`, runner will produce:
- `data/raw/episodes_taskmix_v0581_ab_fullscale.jsonl`
- `data/real_v0581_ab/full_baseline/`
- `data/real_v0581_ab/full_treat_best_<arm>/`
- `data/real_v0581_ab/reports/v0581_compare_fullscale_best.json`
- `data/real_v0581_ab/reports/v0581_compare_fullscale_best.md`

## v0.6.0 SWE Fuel Swap (Local JSONL + Hybrid Pair Source) (2026-02-18)

Goal:
- keep v0.5.x strict pipeline discipline (`build_pairs -> sanitize -> lint -> train/eval -> parity`) unchanged.
- replace fuel source with SWE-agent trajectories via local JSONL adapter.
- use hybrid pair source: `same_task_or_generated`.

### Code Changes

- new HF dataset preparation script:
  - `scripts/prepare_swe_trajectories_v060.py`
- new SWE adapter:
  - `ingest/adapters/swe_agent_v060.py`
- new SWE env judge:
  - `ingest/envs/swe_logs_env.py`
- env registry extension:
  - `ingest/envs/registry.py` adds `swe_logs`
- pair builder extensions:
  - `scripts/build_pairs_v030.py`
  - adds `--pair_source_mode {generated_only,same_task_or_generated}`
  - adds adapter `swe_agent_v060`
  - pair meta adds:
    - `pair_source`
    - `pos_steps_len`, `neg_steps_len`, `steps_len_delta_abs`
    - `pos_patch_len`, `neg_patch_len`, `patch_len_delta_abs`
    - `pos_tool_calls`, `neg_tool_calls`, `tool_calls_delta_abs`
- lint extensions:
  - `scripts/lint_data.py`
  - adds:
    - `steps_delta_abs_p50/p90`
    - `patch_delta_abs_p50/p90`
    - `tool_calls_delta_abs_p50/p90`
    - `--fail_on_shortcut_length --max_steps_delta_abs_p90`
- new runners:
  - `scripts/run_v060_swe_precheck.py` (Phase0, no training)
  - `scripts/run_v060_swe_acceptance.py` (Phase1, minimal train/eval)

### Data Preparation Command (HF -> Local JSONL)

```bash
python3 scripts/prepare_swe_trajectories_v060.py \
  --dataset_id nebius/SWE-agent-trajectories \
  --split train \
  --revision 68195a1450865274106246d0d0296a1d6807b88e \
  --out_path data/raw/episodes_swe_v060.jsonl \
  --stats_path data/raw/episodes_swe_v060.stats.json \
  --forbidden_markers_file configs/forbidden_markers_v060.json
```

Expected stats keys (`data/raw/episodes_swe_v060.stats.json`):
- `rows_in`, `rows_out`
- `dropped_missing_instruction`, `dropped_empty_steps`, `dropped_label_conflict`
- `success_counts`, `target_label_counts`, `success_source_counts`
- `trajectory_format_counts`
- `tasks_unique`, `task_size_stats`
- `rows_with_forbidden_marker_in_text`
- `rows_with_hard_leak_marker_in_text`
- `rows_with_soft_marker_in_text`
- `top_markers_hit`, `marker_example_instance_ids`

Label policy:
- `target` first
- fallback: `eval_logs` -> `exit_status`
- conflict default: keep sample and record conflict in `raw_meta`

### Phase0 Command (Precheck)

```bash
python3 scripts/run_v060_swe_precheck.py \
  --raw_path data/raw/episodes_swe_v060.jsonl \
  --adapter swe_agent_v060 \
  --env_judge swe_logs \
  --pair_source_mode same_task_or_generated \
  --work_dir data/real_v060_swe \
  --max_pairs 1200 \
  --target_train_size 1000 \
  --target_valid_size 200 \
  --forbidden_markers_file configs/forbidden_markers_v060.json
```

Phase0 Level-A gates:
- `pairs_trainable >= 1200`
- `judge_unknown_rate <= 0.05`
- `env_judge_consistency >= 0.95`
- `missing=0`, `leak=0`, `truncation_risk_last_k_steps=0`
- `steps_delta_abs_p90 <= 50`

### Phase1 Command (Minimal Training)

```bash
python3 scripts/run_v060_swe_acceptance.py \
  --raw_path data/raw/episodes_swe_v060.jsonl \
  --adapter swe_agent_v060 \
  --env_judge swe_logs \
  --pair_source_mode same_task_or_generated \
  --work_dir data/real_v060_swe \
  --results_dir results/v060_swe \
  --max_steps 2000
```

Device fallback behavior:
- default `--device auto --amp_dtype none`
- if `--device cuda` is requested while CUDA is unavailable, runner falls back to CPU unless `--strict_device` is set.

Evidence pack command:

```bash
python3 scripts/export_v060_evidence_pack.py \
  --episodes_path data/raw/episodes_swe_v060.jsonl \
  --stats_path data/raw/episodes_swe_v060.stats.json \
  --episodes_sha_path data/raw/episodes_swe_v060.jsonl.sha256 \
  --stats_sha_path data/raw/episodes_swe_v060.stats.json.sha256 \
  --env_snapshot_path data/real_v060_swe/reports/env_snapshot.json
```

Phase1+ stability sweep:

```bash
python3 scripts/run_v060_swe_phase1_plus.py \
  --raw_path data/raw/episodes_swe_v060.jsonl \
  --max_pairs 10000 \
  --seeds 42,43,44 \
  --steps_delta_values 50,30 \
  --work_dir data/real_v060_swe_phase1p \
  --results_dir results/v060_swe_phase1p
```

Phase1 Level-B gates:
- `pair_accuracy > 0.55`
- `gap_p50 > 0`
- parity diff `<= 0.03`

### Compliance

- data licensing note:
  - `docs/data_licenses.md`
- marker source-of-truth:
  - `configs/forbidden_markers_v060.json`
  - hard leak markers drive lint leak gate; soft markers are audit-only.
- policy:
  - keep exported artifacts to pair text + reports only
  - do not publish third-party repository snapshots

## v0.6.2 SWE Simulated Online BoN (Closed Loop) (2026-02-19)

### Scope

- Introduce simulated online environment with snapshot/restore and per-step action commit.
- Evaluate closed-loop policies on the same task set:
  - `baseline_no_planning`
  - `baseline_random_bon`
  - `rm_bon`
- Keep v0.6.1-style strict statistical/audit gate.

### Main Command (Smoke, CPU)

```bash
python3 scripts/run_v062_swe_bon_online.py \
  --raw_path data/raw/episodes_swe_v060.jsonl \
  --ckpt results/v061_swe_phase1p_r2/pairs_10000/s42_d30/rm.pt \
  --results_dir results/v062_swe_online_smoke \
  --n_tasks 10 \
  --seed 42 \
  --bon_n 4 \
  --rollout_horizon 1 \
  --max_env_steps 20 \
  --device cpu
```

Expected artifacts:
- `results/v062_swe_online_smoke/online_traces.jsonl`
- `results/v062_swe_online_smoke/online_run_meta.json`

### Evaluation + Summary

```bash
python3 scripts/eval_v062_online.py \
  --traces_path results/v062_swe_online_smoke/online_traces.jsonl \
  --output_json results/v062_swe_online_smoke/online_eval.json \
  --n_bootstrap 500 \
  --bootstrap_seed 42

python3 scripts/summarize_v062_online.py \
  --eval_json results/v062_swe_online_smoke/online_eval.json \
  --output_json results/v062_swe_online_smoke/online_summary.json \
  --output_md results/v062_swe_online_smoke/online_summary.md
```

Smoke reference (local run):
- `n_tasks = 10`
- `passed = false`
- `rm_minus_best_baseline = 0.0`

### Optional Matrix

```bash
python3 scripts/run_v062_online_matrix.py \
  --raw_path data/raw/episodes_swe_v060.jsonl \
  --ckpt results/v061_swe_phase1p_r2/pairs_10000/s42_d30/rm.pt \
  --results_dir results/v062_swe_online_matrix \
  --seeds 42,43,44 \
  --n_tasks 300 \
  --bon_n 4 \
  --rollout_horizon 1 \
  --max_env_steps 20 \
  --device cuda \
  --output_json results/v062_swe_online_matrix/matrix_raw.json
```

### Gate Rules (v0.6.2)

- `delta_threshold_ok`: `rm_minus_best_baseline >= 0.03`
- `delta_ci_lower_gt_zero`: `ci_low > 0`
- `audit_steps_corr_ok`: `|rm_score_steps_len_corr| < 0.20`
- `audit_token_corr_ok`: `|rm_score_token_proxy_corr| < 0.20`
- `passed = all(checks)`

## v0.6.3 SWE Real Executor Migration (SWE-bench Lite + Docker) (2026-02-19)

### Scope

- Freeze RM checkpoint and BoN planner interface.
- Replace v0.6.2 simulated env backend with real patch/test execution backend.
- Keep v0.6.2 planning/audit gates and add executor-health gates.

### Step A: Build Task Manifest (Intersection)

```bash
python3 scripts/build_v063_task_manifest.py \
  --raw_path data/raw/episodes_swe_v060.jsonl \
  --swe_lite_manifest_path data/raw/swebench_lite_manifest.jsonl \
  --output_path results/v063_swe_real/task_manifest_intersection.jsonl \
  --drop_report_path results/v063_swe_real/task_manifest_drop_report.json \
  --min_candidates 4 \
  --prefer_nonempty_patch true \
  --max_tasks 100
```

Expected artifacts:
- `results/v063_swe_real/task_manifest_intersection.jsonl`
- `results/v063_swe_real/task_manifest_drop_report.json`

### Step B/C: Real Backend Online Run

```bash
python3 scripts/run_v063_swe_bon_online.py \
  --raw_path data/raw/episodes_swe_v060.jsonl \
  --ckpt results/v061_swe_phase1p_r2/pairs_10000/s42_d30/rm.pt \
  --results_dir results/v063_swe_real \
  --env_backend swe_real \
  --swe_lite_manifest_path results/v063_swe_real/task_manifest_intersection.jsonl \
  --docker_image swebench/swebench-lite:latest \
  --workspace_root /tmp/v063_swe_real \
  --max_task_runtime_sec 600 \
  --max_patch_chars 50000 \
  --n_tasks 100 \
  --seed 42 \
  --bon_n 4 \
  --rollout_horizon 1 \
  --max_env_steps 1 \
  --device cuda
```

Expected artifacts:
- `results/v063_swe_real/online_traces.jsonl`
- `results/v063_swe_real/online_run_meta.json`

### Step D: Eval + Summary

```bash
python3 scripts/eval_v062_online.py \
  --traces_path results/v063_swe_real/online_traces.jsonl \
  --output_json results/v063_swe_real/online_eval.json \
  --n_bootstrap 2000 \
  --bootstrap_seed 42 \
  --success_delta_threshold 0.03 \
  --audit_abs_corr_threshold 0.20 \
  --min_runnable_task_rate 0.85 \
  --min_test_exec_success_rate 0.80 \
  --max_timeout_rate 0.10

python3 scripts/summarize_v062_online.py \
  --eval_json results/v063_swe_real/online_eval.json \
  --output_json results/v063_swe_real/online_summary.json \
  --output_md results/v063_swe_real/online_summary.md
```

### Gate Rules (v0.6.3)

Planning gates:
- `rm_minus_best_baseline >= 0.03`
- `delta_ci.ci_low > 0`
- `|rm_score_steps_len_corr| < 0.20`
- `|rm_score_token_proxy_corr| < 0.20`

Executor-health gates:
- `runnable_task_rate >= 0.85`
- `test_exec_success_rate >= 0.80`
- `timeout_rate <= 0.10`

Final verdict:
- `passed = all(planning_checks) and all(executor_health_checks)`

## v0.6.4 M1 Main Run + M2 Scaling Matrix (2026-02-19)

### M1: 100-task Main Run (Real Backend)

```bash
python3 scripts/run_v063_m1_main.py \
  --raw_path data/raw/episodes_swe_v060.jsonl \
  --ckpt results/v061_swe_phase1p_r2/pairs_10000/s42_d30/rm.pt \
  --swe_lite_manifest_path data/raw/swebench_lite_manifest.jsonl \
  --docker_image swebench/swebench-lite:latest \
  --workspace_root /tmp/v063_swe_real_main \
  --results_dir results/v063_swe_real_main \
  --n_tasks 100 \
  --seeds 42,43,44 \
  --bon_n 4 \
  --rollout_horizon 1 \
  --max_env_steps 1
```

Expected artifacts:
- `results/v063_swe_real_main/task_manifest_intersection.jsonl`
- `results/v063_swe_real_main/task_manifest_drop_report.json`
- `results/v063_swe_real_main/m1_main_report.json`
- `results/v063_swe_real_main/seed_42/online_eval.json` (and `seed_43`, `seed_44`)

### M2: Inference-time Scaling Matrix

```bash
python3 scripts/run_v064_scaling_matrix.py \
  --raw_path data/raw/episodes_swe_v060.jsonl \
  --ckpt results/v061_swe_phase1p_r2/pairs_10000/s42_d30/rm.pt \
  --swe_lite_manifest_path results/v063_swe_real_main/task_manifest_intersection.jsonl \
  --results_dir results/v064_scaling \
  --bon_n_grid 1,2,4,8 \
  --seeds 42,43,44 \
  --n_tasks 100 \
  --fixed_horizon 1 \
  --fixed_max_env_steps 1 \
  --output_json results/v064_scaling/matrix_raw.json

python3 scripts/summarize_v064_scaling.py \
  --matrix_json results/v064_scaling/matrix_raw.json \
  --output_json results/v064_scaling/matrix_summary.json \
  --output_md results/v064_scaling/matrix_summary.md \
  --curve_points_json results/v064_scaling/curve_points.json
```

Expected artifacts:
- `results/v064_scaling/matrix_raw.json`
- `results/v064_scaling/matrix_summary.json`
- `results/v064_scaling/matrix_summary.md`
- `results/v064_scaling/curve_points.json`

## v0.6.5 M3 Online Bootstrap (2 Rounds) (2026-02-19)

### Preflight (Before Starting Paid GPU Runtime)

```bash
python3 scripts/preflight_v065_4090.py \
  --ckpt results/v061_swe_phase1p_r2/pairs_10000/s42_d30/rm.pt \
  --swe_lite_manifest_path results/v063_swe_real_main/task_manifest_intersection.jsonl \
  --device cuda \
  --strict_git_clean true \
  --require_docker_image_present true \
  --output_json results/preflight/v065_4090_preflight.json

bash scripts/freeze_v065_run_state.sh
```

### Round Orchestration

```bash
python3 scripts/run_v065_bootstrap_rounds.py \
  --base_ckpt results/v061_swe_phase1p_r2/pairs_10000/s42_d30/rm.pt \
  --rounds 2 \
  --raw_path data/raw/episodes_swe_v060.jsonl \
  --swe_lite_manifest_path results/v063_swe_real_main/task_manifest_intersection.jsonl \
  --results_dir results/v065_bootstrap \
  --n_tasks 100 \
  --seeds 42,43,44 \
  --bon_n 4 \
  --rollout_horizon 1 \
  --max_env_steps 1
```

Expected artifacts:
- `results/v065_bootstrap/round_0/round_report.json`
- `results/v065_bootstrap/round_1/round_report.json`
- `results/v065_bootstrap/round_2/round_report.json`
- `results/v065_bootstrap/bootstrap_summary.json`

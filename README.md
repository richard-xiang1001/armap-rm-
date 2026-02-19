# ARMAP Reward Model (Text-only) — Minimal Repro

This repo provides a **minimal runnable** text-only Reward Model (RM) training pipeline inspired by **ARMAP: Scaling Autonomous Agents via Automatic Reward Modeling and Planning**.

v0.2.1 acceptance reference (AutoDL RTX 4090) has validated this end-to-end guarantee:
- real trajectories -> export/sanitize -> strict lint gate (`missing=0`, `leak=0`, `truncation=0`)
- lightweight bf16 GPU training -> eval (`pair_accuracy>0.55`, `gap_p50>0`)
- CPU/GPU parity check on the same checkpoint and validation split (`abs(diff)<=0.03`, observed `0.0`)
- acceptance artifact report: `data/real/reports/v021_acceptance.json` with `passed=true`
Version guidance: recommended reference tag is `v0.2.3`; `v0.2.1` is the first acceptance milestone, and `v0.2.2` is an occupied immutable tag.

v0.3.x adds ARMAP-style data and input-shape alignment:
- v0.3.0: generated executable negatives (`replay_corrupt -> swap_step -> truncate`) + arithmetic env judge + new replay/consistency gates
- v0.3.1: stepwise input formatting (`<OBS>/<ACT>/<RES>`) + selectable reward pooling (`mean/last/last_k_step`)
- v0.3.2 (optional scaffold): lightweight multimodal wrapper (`rm/vlm`) and `webshop_like` adapter

v0.4.x extends this to real evidence and protocolized judge:
- v0.4.0: real-log acceptance runner (`scripts/run_v040_real_acceptance.py`)
- v0.4.1: env judge protocol/registry (`ingest/envs/protocol.py`, `ingest/envs/registry.py`)
- v0.4.2: stronger hard negatives + bucket sampling + mismatch dumps + `run_ablation_v04x.py`

v0.5.0 adds a no-agent data fuel loop:
- task-mix env judge (`arithmetic/retrieval/constraint`) with deterministic success checks
- scalable episode factory (`scripts/generate_taskmix_episodes_v050.py`)
- pre-pair quality gate (`scripts/validate_taskmix_episodes_v050.py`)
- release acceptance runner (`scripts/run_v050_taskmix_acceptance.py`)
- ablation runner with mismatch-by-family stats (`scripts/run_ablation_v05x.py`)

v0.5.2 adds evidence hardening on top of v0.5.1:
- hard-negative curriculum acceptance runner (`scripts/run_v052_taskmix_acceptance.py`)
- train-time checkpoint init for stage-wise continuation (`python3 -m rm.train --init_ckpt ...`)
- counterfactual formatting audit (`scripts/audit_counterfactual_format_v052.py`)

v0.5.2 one-command full-scale (no-agent, release-grade):

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

v0.5.3 adds counterfactual flip analysis + format-aug training:
- format-aware augmentation builder (`scripts/build_format_aug_pairs_v053.py`)
- counterfactual flip analyzer (`scripts/analyze_counterfactual_flips_v053.py`)
- v0.5.3 acceptance runner (`scripts/run_v053_taskmix_acceptance.py`)
- `PairCollator` supports `input_format=auto` via `meta.format_variant`
- v0.5.3 A/B shortest-loop (`off+flat` vs `concat+auto`) commands and archived comparison are documented in `docs/reproducibility.md` ("v0.5.3 A/B Counterfactual Consistency Compare")
- v0.5.4 adds training-level consistency regularization (`sign_hinge`) and A/B compare (`consistency_lambda=0.0` vs `0.10`); commands and archived results are in `docs/reproducibility.md` ("v0.5.4 A/B Consistency Regularization Compare")
- v0.5.5 keeps the same A/B protocol but switches treatment to `gap_l2` and adds smoke-gated full-scale policy + training diagnostics (`consistency_violation_nonpos_rate`, `consistency_gap_abs_*`); details in `docs/reproducibility.md` ("v0.5.5 A/B gap_l2 + Smoke Gate")
- v0.5.6 keeps `gap_l2` and adds stage-3 flip hard-mining (`flip80`) under the same strict smoke gate; commands, gate decision, and archived artifacts are in `docs/reproducibility.md` ("v0.5.6 A/B gap_l2 + Flip Hard-Mining + Strict Smoke Gate")
- v0.5.7 adds a 2x2 hard-mining matrix (`flip_fraction={0.5,1.0}` x `{with_replacement,without_replacement}`) with mining-split source, per-arm compare, and matrix-level strict smoke gate; see `docs/reproducibility.md` ("v0.5.7 A/B Matrix + Strict Smoke Gate")
- v0.5.8 keeps v0.5.7 training frozen and switches counterfactual audit to semantic-equivalent stepwise styles (`xml_v1` vs `marker_v2`), adds `gap_delta_*` evidence, and re-runs the same 2x2 strict smoke gate matrix; see `docs/reproducibility.md` ("v0.5.8 Semantic-Equivalent Counterfactual Audit + Strict Smoke Gate")
- v0.5.8.1 keeps v0.5.8 training frozen and upgrades compare/gate to dual-gate (`sign + gap_delta_p90_rel`) via parallel scripts (`compare_v060`, `summarize_v060`), preserving historical compare compatibility; see `docs/reproducibility.md` ("v0.5.8.1 Dual-Gate (Sign + Continuous) Matrix")
- v0.6.0 keeps v0.5.x training/gate discipline and swaps fuel to SWE-agent trajectories (`swe_agent_v060` adapter + `swe_logs` judge), with hybrid pair sourcing (`same_task_or_generated`) and shortcut-length lint gate (`steps_delta_abs_p90`); license boundary notes are in `docs/data_licenses.md`
- v0.6.1 adds SWE planning hardening (`run_v061_swe_planning_eval.py` + `run_v061_swe_planning_matrix.py`) and records a multi-seed planning matrix result with evidence packaging; see `docs/releases/v0.6.1.md`
- v0.6.2 adds simulated online BoN closed-loop planning (`online/` + `run_v062_swe_bon_online.py`) with trace-level audit, bootstrap CI gating, and summary/matrix tooling; see `docs/releases/v0.6.2.md`
- v0.6.3 upgrades online env backend from simulated replay to real patch/test execution (`SWE-bench Lite + Docker`) while freezing RM/BoN; see `docs/releases/v0.6.3.md`
- v0.6.4 adds M1 main-run orchestration and M2 scaling matrix tooling (`run_v063_m1_main.py`, `run_v064_scaling_matrix.py`, `summarize_v064_scaling.py`); see `docs/releases/v0.6.4.md`
- v0.6.5 adds M3 online bootstrap scripts (`build_v065_bootstrap_pairs.py`, `run_v065_bootstrap_rounds.py`) for 2-round `collect->pair->train->re-plan`; see `docs/releases/v0.6.5.md`

v0.5.3 one-command full-scale (no-agent, release-grade):

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

## v0.3.0 Acceptance (ARMAP-style negatives + replay/judge gate)

Run one command for:
- `build_pairs_v030 -> sanitize -> lint(gate) -> train/eval -> CPU/GPU parity`
- additional gate metrics:
  - `neg_replay_success_rate`
  - `env_judge_consistency`

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

Report output:
- `data/real/reports/v030_acceptance.json`
- official 4090 evidence handoff: `docs/runbooks/v030_acceptance_remote.md`
- latest 4090 precheck evidence: `data/real_v030_precheck/reports/v030_acceptance.json` (`passed=true`)

## v0.3.1 Input Format + Pooling Switches

`rm.train` and `rm.eval` now support:
- `--input_format {flat,stepwise}`
- `--pooling {mean,last,last_k_step}`
- `--last_k_steps_pool`
- `--use_refined_instruction {true,false}`

Example:

```bash
python3 -m rm.train \
  --train_path data/real/train.jsonl \
  --valid_path data/real/valid.jsonl \
  --save_dir results/v031_stepwise \
  --max_len 512 \
  --batch_size 256 \
  --device cuda \
  --amp_dtype bf16 \
  --input_format stepwise \
  --pooling last_k_step \
  --last_k_steps_pool 3
```

## v0.3.x Ablation Runner

Single-variable ablations across:
- neg type: `truncate_last_k/replay_corrupt/swap_step`
- instruction: `raw/refined`
- format: `flat/stepwise`
- pooling: `mean/last/last_k_step`

```bash
python3 scripts/run_ablation_v03x.py \
  --raw_path data/raw/episodes_real.jsonl \
  --adapter generic_jsonl \
  --work_dir data/ablation/v03x \
  --results_dir results/ablation/v03x
```

Outputs:
- `results/ablation/v03x/v03x_ablation.csv`
- `results/ablation/v03x/v03x_ablation.md`

## v0.4.0 Real Evidence Acceptance

```bash
python3 scripts/run_v040_real_acceptance.py \
  --raw_path data/raw/episodes_real_v040.jsonl \
  --adapter real_logs_v1 \
  --env_judge real_logs \
  --work_dir data/real_v040 \
  --results_dir results/v040_real \
  --max_pairs 1200 \
  --target_train_size 1000 \
  --target_valid_size 200
```

Key v0.4 gates:
- `env_judge_consistency >= 0.95`
- `judge_unknown_rate <= 0.05`
- `pair_accuracy > 0.55`, `gap_p50 > 0`

Current reference status (2026-02-16):
- quality gates pass on available sample (`judge_unknown_rate=0.0`, `env_judge_consistency=1.0`)
- release-grade is blocked only by data scale (`pairs_trainable=31`, target `1200 -> 1000/200`)
- blocker summary: `data/real_v040/reports/precheck_pairs_summary.json`

## v0.4.x Ablation Runner

```bash
python3 scripts/run_ablation_v04x.py \
  --raw_path data/raw/episodes_real_v040.jsonl \
  --adapter real_logs_v1 \
  --env_judge real_logs \
  --work_dir data/ablation/v04x \
  --results_dir results/ablation/v04x
```

Outputs:
- `results/ablation/v04x/v04x_ablation.csv`
- `results/ablation/v04x/v04x_ablation.md`

## v0.5.0 TaskMix Acceptance (No-Agent Data Loop)

```bash
python3 scripts/run_v050_taskmix_acceptance.py \
  --raw_path data/raw/episodes_taskmix_v050.jsonl \
  --adapter taskmix_v050 \
  --env_judge taskmix \
  --work_dir data/real_v050 \
  --results_dir results/v050_real \
  --max_pairs 1200 \
  --target_train_size 1000 \
  --target_valid_size 200
```

Pipeline:
- `generate_taskmix_episodes_v050.py` (optional auto-generate)
- `validate_taskmix_episodes_v050.py` (episodes/family/diversity gate)
- `build_pairs_v030.py` (`taskmix_v050 + taskmix`)
- `sanitize -> lint -> train -> eval -> parity`

Main v0.5 defaults:
- data families sampled `1:1:1` (`arithmetic/retrieval/constraint`)
- release target unchanged: `1200 -> 1000/200`
- required done reasons: `goal_satisfied,wrong_answer,invalid_action,constraint_violation,timeout_step_cap`

## v0.5.x Ablation Runner

```bash
python3 scripts/run_ablation_v05x.py \
  --raw_path data/raw/episodes_taskmix_v050.jsonl \
  --adapter taskmix_v050 \
  --env_judge taskmix \
  --work_dir data/ablation/v05x \
  --results_dir results/ablation/v05x
```

Outputs:
- `results/ablation/v05x/v05x_ablation.csv`
- `results/ablation/v05x/v05x_ablation.md`
- includes mismatch distribution by family (`arithmetic/retrieval/constraint`)

## v0.6.0 SWE Precheck (Phase0)

Prepare raw episodes from Hugging Face first:

```bash
python3 scripts/prepare_swe_trajectories_v060.py \
  --dataset_id nebius/SWE-agent-trajectories \
  --split train \
  --out_path data/raw/episodes_swe_v060.jsonl \
  --stats_path data/raw/episodes_swe_v060.stats.json \
  --forbidden_markers_file configs/forbidden_markers_v060.json
```

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

Level-A gates:
- `pairs_trainable >= 1200`
- `judge_unknown_rate <= 0.05`
- `env_judge_consistency >= 0.95`
- `missing=0`, `leak=0`, `truncation_risk_last_k_steps=0`
- `steps_delta_abs_p90 <= 50`

## v0.6.0 SWE Minimal Acceptance (Phase1)

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

Notes:
- `scripts/run_v060_swe_acceptance.py` defaults to `--device auto --amp_dtype none`.
- If `--device cuda` is requested but CUDA is unavailable, runner falls back to CPU unless `--strict_device` is set.

Evidence pack:

```bash
python3 scripts/export_v060_evidence_pack.py \
  --episodes_path data/raw/episodes_swe_v060.jsonl \
  --stats_path data/raw/episodes_swe_v060.stats.json \
  --episodes_sha_path data/raw/episodes_swe_v060.jsonl.sha256 \
  --stats_sha_path data/raw/episodes_swe_v060.stats.json.sha256 \
  --env_snapshot_path data/real_v060_swe/reports/env_snapshot.json
```

Phase1+ stability sweep (10k pairs, multi-seed):

```bash
python3 scripts/run_v060_swe_phase1_plus.py \
  --raw_path data/raw/episodes_swe_v060.jsonl \
  --max_pairs 10000 \
  --seeds 42,43,44 \
  --steps_delta_values 50,30 \
  --work_dir data/real_v060_swe_phase1p \
  --results_dir results/v060_swe_phase1p
```

Level-B gates:
- `pair_accuracy > 0.55`
- `gap_p50 > 0`
- CPU/GPU parity diff `<= 0.03`

## v0.6.2 SWE Simulated Online BoN (Closed Loop)

Run online traces for 3 policies (`baseline_no_planning`, `baseline_random_bon`, `rm_bon`):

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

Evaluate and summarize:

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

Optional multi-seed matrix:

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

Acceptance gates:
- `rm_minus_best_baseline >= 0.03`
- `delta_ci.ci_low > 0`
- `|rm_score_steps_len_corr| < 0.20`
- `|rm_score_token_proxy_corr| < 0.20`

## v0.6.3 SWE Real Executor (SWE-bench Lite + Docker)

Build runnable task manifest from trajectories and SWE-lite metadata:

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

Run real backend closed-loop online traces:

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

Evaluate and summarize:

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

v0.6.3 acceptance gates:
- planning: `rm_minus_best_baseline >= 0.03` and `delta_ci.ci_low > 0`
- anti-shortcut audit: `|rm_score_steps_len_corr| < 0.20`, `|rm_score_token_proxy_corr| < 0.20`
- executor health: `runnable_task_rate >= 0.85`, `test_exec_success_rate >= 0.80`, `timeout_rate <= 0.10`
- final verdict: `passed = all(planning_checks) and all(executor_health_checks)`

## v0.6.4 M1 Main Run + M2 Scaling Matrix

M1 one-command main run (manifest -> online -> eval -> summarize):

```bash
python3 scripts/run_v063_m1_main.py \
  --raw_path data/raw/episodes_swe_v060.jsonl \
  --ckpt results/v061_swe_phase1p_r2/pairs_10000/s42_d30/rm.pt \
  --swe_lite_manifest_path data/raw/swebench_lite_manifest.jsonl \
  --results_dir results/v063_swe_real_main \
  --n_tasks 100 \
  --seeds 42,43,44
```

M2 scaling matrix (`bon_n={1,2,4,8}`):

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

## v0.6.5 M3 Online Bootstrap (2 Rounds)

Preflight before paid GPU run (recommended):

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

Primary outputs:
- `results/v063_swe_real_main/m1_main_report.json`
- `results/v064_scaling/matrix_raw.json`
- `results/v064_scaling/matrix_summary.json`
- `results/v065_bootstrap/bootstrap_summary.json`

## What is being learned?
The RM learns a scalar scoring function **R(x, h)** such that for the same instruction *x*,

- R(x, h⁺) > R(x, h⁻)

and is trained with the pairwise logistic objective.

See `docs/training_details.md` for a full engineering-level protocol.

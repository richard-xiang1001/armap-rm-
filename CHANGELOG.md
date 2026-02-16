# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog, and this project follows semantic versioning.

## [v0.4.2] - 2026-02-16

### Added

- Added v0.4.x ablation runner:
  - `scripts/run_ablation_v04x.py`
  - dimensions: `neg_type`, `refine_mode`, `input_format`, `pooling`
- Added mismatch export for eval debugging:
  - `rm/eval.py --dump_mismatches`
- Added bucket sampling support:
  - `rm/data.py` (bucket weights/sampler)
  - `rm/train.py --bucket_sampling`
- Release note: `docs/releases/v0.4.2.md`

### Changed

- Hard-negative construction strengthened:
  - `replay_corrupt`: prioritizes late-step and key-parameter perturbation
  - `swap_step`: adjacent-step constrained swaps for more plausible failures
- Published v0.4.2 ablation evidence paths:
  - `results/ablation/v04x/v04x_ablation.csv`
  - `results/ablation/v04x/v04x_ablation.md`
- Updated v0.4.2 evidence summary after rerun:
  - all 24/24 ablation runs now complete successfully
  - mismatch concentration identified in `swap_step` runs (8 non-empty mismatch files)

## [v0.4.1] - 2026-02-16

### Added

- Added env judge protocol:
  - `ingest/envs/protocol.py`
  - `JudgeResult` + `CompareResult` + `compare_judgments(...)`
- Added judge registry:
  - `ingest/envs/registry.py`
  - `get_env_judge(...)`
- Added real logs judge implementation:
  - `ingest/envs/real_logs_env.py`
- Release note: `docs/releases/v0.4.1.md`

### Changed

- Refactored negative generators to injected env judge (no hard-coded arithmetic judge):
  - `ingest/negatives/{replay_corrupt.py,swap_step.py,truncate.py,registry.py}`
- Extended pair meta with protocol fields:
  - `judge_unknown`, `judge_unknown_reason`, `judge_protocol_version`
- Added dual-judge migration evidence reports:
  - `data/v041_arithmetic/reports/v030_acceptance.json`
  - `data/v041_real_logs/reports/v030_acceptance.json`
- Refreshed protocol migration evidence:
  - both judge runs now satisfy strict unknown/consistency gates (`judge_unknown_rate=0.0`, `env_judge_consistency=1.0`)

## [v0.4.0] - 2026-02-16

### Added

- Added real logs adapter:
  - `ingest/adapters/real_logs_v1.py`
- Added v0.4.0 real evidence acceptance runner:
  - `scripts/run_v040_real_acceptance.py`
- Added real-log sample generator:
  - `scripts/generate_real_logs_v040_sample.py`
- Added release notes:
  - `docs/releases/v0.4.0.md`

### Changed

- Updated pair builder and acceptance flow:
  - `scripts/build_pairs_v030.py` now supports `--env_judge`
  - emits `pairs.all.jsonl` + `pairs.trainable.jsonl` workflow
- Extended lint gates:
  - `judge_unknown_rate`
  - `--fail_on_high_judge_unknown --max_judge_unknown_rate`
- Enhanced sanitize audit:
  - key-value masking support
  - `mask_type_counts`, `masked_field_ratio`, `masked_value_ratio`
- Updated reproducibility doc with v0.4.0/v0.4.2 commands:
  - `docs/reproducibility.md`
- Recorded release-blocker evidence and precheck evidence:
  - blocker: `data/real_v040/reports/{export_stats_v040.json,train_lint.json}`
  - precheck: `data/real_v040_precheck/reports/v040_real_acceptance.json`
- Added explicit scale-blocker decision artifact:
  - `data/real_v040/reports/precheck_pairs_summary.json` (`LT_300_DATA_SCALE_BLOCKER`)
- Updated release note/reproducibility narrative:
  - blocker is now documented as data scale only (quality gates green on current sample)

## [v0.3.2] - 2026-02-16

### Added

- Optional multimodal scaffolding:
  - `rm/vlm/wrapper.py` with freeze-backbone + scalar-head layout
  - `ingest/adapters/webshop_like.py` for Webshop-like multimodal episode logs
- Release note: `docs/releases/v0.3.2.md`

## [v0.3.1] - 2026-02-16

### Added

- Stepwise trajectory formatter:
  - `rm/formatters/stepwise.py`
- Pooling module:
  - `rm/pooling.py` with `mean`, `last`, `last_k_step`

### Changed

- Extended RM train/eval interfaces:
  - `--input_format {flat,stepwise}`
  - `--pooling {mean,last,last_k_step}`
  - `--last_k_steps_pool`
  - `--use_refined_instruction {true,false}`
- Updated checkpoint config to store v0.3.1 input/pooling settings.
- Release note: `docs/releases/v0.3.1.md`

## [v0.3.0] - 2026-02-16

### Added

- Added ARMAP-style pair-construction pipeline:
  - `scripts/build_pairs_v030.py`
  - `ingest/refine_instruction.py`
  - `ingest/negatives/{replay_corrupt.py,swap_step.py,truncate.py,registry.py}`
  - `ingest/envs/{base.py,arithmetic_env.py}`
- Added v0.3.0 acceptance runner:
  - `scripts/run_v030_acceptance.py`
- Added v0.3.x ablation runner:
  - `scripts/run_ablation_v03x.py`
- Added release note: `docs/releases/v0.3.0.md`

### Changed

- Extended pair `meta` schema (backward-compatible):
  - `construction_method`, `neg_type`, `replay_ok`, `replay_attempted`
  - `env_judge_pos`, `env_judge_neg`, `env_judge_consistent`
  - `instruction_refine_mode`, `instruction_refine_changed`
- Extended `scripts/lint_data.py` with replay/env-consistency metrics and gates:
  - `neg_replay_attempted`, `neg_replay_ok_count`, `neg_replay_success_rate`
  - `env_judge_consistency`
  - `--fail_on_low_replay_ok --min_replay_ok_rate`
  - `--fail_on_low_env_consistency --min_env_judge_consistency`
- Updated docs:
  - `README.md`
  - `docs/training_details.md`
  - `docs/reproducibility.md`
  - `docs/runbooks/v030_acceptance_remote.md` (remote 4090 acceptance handbook)
  - Recorded v0.3.0 4090 precheck evidence in:
    - `docs/reproducibility.md`
    - `docs/releases/v0.3.0.md`

## [v0.2.2] - 2026-02-16

### Changed

- Documentation and release alignment only (no training/data-pipeline behavior changes):
  - Added v0.2.1 acceptance-closure summary to `README.md`.
  - Recorded v0.2.1 reference-run and metric-interpretation notes in docs.
  - Published `docs/releases/v0.2.2.md` to keep release tag content aligned with latest docs.

## [v0.2.1] - 2026-02-16

### Added

- Added `scripts/run_v021_acceptance.py`:
  - runs end-to-end acceptance for `5500 -> 5000/500` split
  - validates strict gate metrics
  - runs GPU and CPU eval on the same checkpoint/valid set
  - computes parity diff and writes `v021_acceptance.json`

### Changed

- Enhanced `scripts/run_v020_pipeline.py` split controls:
  - new `--valid_size` argument to request exact validation size
  - ratio-based split now uses rounded size when `--valid_size` is not set
- Updated docs with v0.2.1 acceptance command and report path:
  - `README.md`
  - `docs/reproducibility.md`

## [v0.2.0] - 2026-02-16

### Added

- Added real-trajectory ingest pipeline:
  - `ingest/adapters/generic_jsonl.py`
  - `ingest/adapters/armap_style.py`
  - `ingest/export_pairs.py` (same-task success/failure pairing + tail-truncation fallback negatives)
  - `ingest/sanitize_traj.py` (leak masking + step marker normalization + audit report)
- Added one-command end-to-end pipeline: `scripts/run_v020_pipeline.py`
  - fixed order: `export_pairs -> sanitize -> lint gate -> train -> eval`
- Added minimal de-identified real-like sample generator: `scripts/generate_real_episode_sample.py`
- Added `docs/releases/v0.2.0.md` release note.

### Changed

- Extended `rm.train` CLI for GPU throughput and stability tuning:
  - `--device {auto,cpu,cuda}`
  - `--amp_dtype {none,bf16,fp16}` with bf16->fp16 fallback on unsupported GPUs
  - `--grad_accum_steps`
  - `--max_steps`
  - `--pin_memory {auto,true,false}`
  - `--prefetch_factor`
  - `--persistent_workers {auto,true,false}`
- Enhanced training logs (`metrics.jsonl`) with:
  - `samples_per_sec`
  - `tokens_per_sec`
  - `amp_dtype_used`
- Extended `scripts/lint_data.py` with strict truncation gate support:
  - `--fail_on_truncation_risk`
  - `--max_truncation_risk_last_k_steps`
  - `--report_path`

### Documentation

- Updated `README.md` with v0.2.0 real-trajectory workflow and 4090 smoke command.
- Updated `docs/training_details.md` with strict gate policy and ingest/sanitize workflow.
- Updated `docs/reproducibility.md` with v0.2.0 4090/real-data reporting template and parity criteria.

## [v0.1.0] - 2026-02-16

### Added

- Added `scripts/lint_data.py` for RM dataset checks:
  - required field completeness
  - empty trajectory ratio
  - token-length statistics
  - leakage keyword scan
  - truncation risk on `Final:` lines under target `max_len`
  - truncation risk on the tail `K` steps via configurable step markers
  - optional external leak-term dictionary via `--leak_terms_file`
- Added `docs/reproducibility.md` with a reusable run-report template and a reference run record.
- Added `docs/releases/v0.1.0.md` as a publish-ready release note.

### Changed

- Enhanced `python3 -m rm.eval` output while preserving compatibility:
  - existing fields kept: `pair_accuracy`, `avg_reward_gap`, `n`
  - new fields: `gap_p10`, `gap_p50`, `gap_p90`
- Updated docs and quickstart to include data linting before training.

### Fixed

- Documented and mitigated sandboxed macOS shared-memory worker failure (`torch_shm_manager ... Operation not permitted`) by using `--num_workers 0`.
- Documented truncation failure mode (`max_len=128` hiding `Final:` signal) and the stable workaround (`max_len=512` on synthetic data).

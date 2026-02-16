# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog, and this project follows semantic versioning.

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

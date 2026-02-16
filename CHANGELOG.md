# Changelog

All notable changes to this project are documented in this file.

The format is based on Keep a Changelog, and this project follows semantic versioning.

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

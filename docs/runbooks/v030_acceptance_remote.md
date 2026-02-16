# v0.3.0 Remote Acceptance Runbook (4090)

This runbook is for producing the **official v0.3.0 evidence pack** on a single RTX 4090 host.

## 1. Scope Freeze

- Target: `v0.3.0` acceptance only.
- Dataset: `data/raw/episodes_real.jsonl` (same lineage as v0.2.1).
- Fixed defaults:
  - `seed=42`
  - `max_pairs=5500`
  - `target_train_size=5000`
  - `target_valid_size=500`
  - `max_len=512`
  - `device=cuda`
  - `amp_dtype=bf16`

## 2. Preflight

Run in repo root:

```bash
pwd
git rev-parse --short HEAD
python3 -V
python3 - <<'PY'
import torch, platform
print({
  'os': platform.platform(),
  'torch': torch.__version__,
  'cuda_available': torch.cuda.is_available(),
  'device_count': torch.cuda.device_count(),
  'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
})
PY
nvidia-smi
ls -lh data/raw/episodes_real.jsonl
```

If `data/raw/episodes_real.jsonl` is missing, stop and sync dataset first.

## 3. Canonical Acceptance Command

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

## 4. Required Artifacts

Keep these files:

- `data/real/reports/v030_acceptance.json`
- `data/real/reports/train_lint.json`
- `data/real/reports/export_stats_v030.json`
- `results/v030_real/metrics.jsonl`
- `results/v030_real/rm.pt`

## 5. Hard Acceptance Criteria

All must pass:

- `pairs_out >= 5500`
- `rows_with_missing = 0`
- `leak_hit_count = 0`
- `truncation_risk_last_k_steps <= 0.05`
- `neg_replay_success_rate >= 0.8`
- `env_judge_consistency >= 0.95`
- `pair_accuracy > 0.55`
- `gap_p50 > 0`
- `abs(pair_acc_gpu - pair_acc_cpu) <= 0.03`
- `passed = true`

## 6. Failure Triage (Fixed Order)

1. `gate_replay_ok` failed:
- Inspect `export_stats_v030.json` fields:
  - `neg_type_counts`
  - `neg_attempted_counts`
- Confirm `replay_corrupt` coverage is not unexpectedly low.

2. `gate_env_consistency_ok` failed:
- Sample 100 pairs from `pairs.unsanitized.jsonl`.
- Audit `meta.env_judge_pos/env_judge_neg/env_judge_consistent` against `traj_pos/traj_neg` and `Final:` lines.
- Check whether sanitize masked key judge text unexpectedly.

3. `train_pair_acc_ok` or `eval_gap_p50_ok` failed:
- Rerun once with conservative settings only:
  - keep `max_len=512`
  - use `--input_format flat --pooling mean`

4. `parity_ok` failed:
- Re-run GPU/CPU eval on same checkpoint and same valid split with `num_workers=0`.

## 7. Evidence Publication Checklist

After successful run:

1. Fill `docs/reproducibility.md` section `v0.3.0 Reference Run`.
2. Fill `docs/releases/v0.3.0.md` section `Reference Run Evidence`.
3. Optionally add one-line status in `README.md` that v0.3.0 reference run passed.
4. Update `CHANGELOG.md` with evidence publication note.

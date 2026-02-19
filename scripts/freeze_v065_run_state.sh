#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${ROOT_DIR}/results/preflight/run_state_${STAMP}"
mkdir -p "${OUT_DIR}"

echo "[freeze] root=${ROOT_DIR}"
echo "[freeze] out=${OUT_DIR}"

(
  cd "${ROOT_DIR}"
  git rev-parse HEAD > "${OUT_DIR}/git_head.txt" || true
  git branch --show-current > "${OUT_DIR}/git_branch.txt" || true
  git status --short > "${OUT_DIR}/git_status_short.txt" || true
  git diff --stat > "${OUT_DIR}/git_diff_stat.txt" || true
  git log -1 --oneline > "${OUT_DIR}/git_log1.txt" || true
)

python3 --version > "${OUT_DIR}/python_version.txt" 2>&1 || true
python3 -m pip freeze > "${OUT_DIR}/pip_freeze.txt" 2>&1 || true

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "${OUT_DIR}/nvidia_smi.txt" 2>&1 || true
else
  echo "nvidia-smi not found" > "${OUT_DIR}/nvidia_smi.txt"
fi

if command -v docker >/dev/null 2>&1; then
  docker --version > "${OUT_DIR}/docker_version.txt" 2>&1 || true
  docker info > "${OUT_DIR}/docker_info.txt" 2>&1 || true
else
  echo "docker not found" > "${OUT_DIR}/docker_version.txt"
  echo "docker not found" > "${OUT_DIR}/docker_info.txt"
fi

cat > "${OUT_DIR}/run_notes.txt" <<'EOF'
Recommended order on paid GPU:
1) Run scripts/preflight_v065_4090.py with --device cuda and strict checks.
2) Run M1 smoke (seed=42, small n_tasks) and verify gates/log schema.
3) Run M1 full (seeds 42/43/44), then M2 scaling, then M3 bootstrap.
EOF

echo "{\"run_state_dir\":\"${OUT_DIR}\"}"

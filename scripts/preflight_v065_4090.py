#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def _run(cmd: List[str], cwd: Path) -> Dict[str, Any]:
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, check=False)
    return {
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "stdout": str(proc.stdout or "").strip(),
        "stderr": str(proc.stderr or "").strip(),
        "elapsed_sec": float(time.perf_counter() - t0),
    }


def _bool_text(v: str) -> bool:
    t = str(v or "").strip().lower()
    if t in {"1", "true", "yes", "y"}:
        return True
    if t in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Invalid bool text: {v}")


def _check_cli_help(repo_root: Path, script_rel: str) -> Dict[str, Any]:
    py = sys.executable
    res = _run([py, str(repo_root / script_rel), "--help"], cwd=repo_root)
    return {
        "script": script_rel,
        "ok": bool(res["returncode"] == 0),
        "returncode": int(res["returncode"]),
        "elapsed_sec": float(res["elapsed_sec"]),
        "stderr_tail": "\n".join(str(res["stderr"]).splitlines()[-5:]),
    }


def _torch_cuda_snapshot(repo_root: Path, device: str) -> Dict[str, Any]:
    py = sys.executable
    code = (
        "import json, torch\n"
        "out={\n"
        "  'torch_version': getattr(torch, '__version__', ''),\n"
        "  'cuda_available': bool(torch.cuda.is_available()),\n"
        "  'cuda_device_count': int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,\n"
        "  'cuda_device_name': str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else '',\n"
        "  'torch_cuda_version': str(getattr(torch.version, 'cuda', '') or ''),\n"
        "}\n"
        "print(json.dumps(out))\n"
    )
    res = _run([py, "-c", code], cwd=repo_root)
    payload: Dict[str, Any] = {}
    if res["returncode"] == 0 and res["stdout"]:
        try:
            payload = json.loads(res["stdout"].splitlines()[-1])
        except Exception:
            payload = {"parse_error": "torch_snapshot_json_parse_failed", "raw_stdout": res["stdout"]}
    ok = bool(res["returncode"] == 0)
    if device == "cuda":
        ok = ok and bool(payload.get("cuda_available", False))
    return {
        "ok": ok,
        "device_requested": device,
        "details": payload,
        "stderr_tail": "\n".join(str(res["stderr"]).splitlines()[-5:]),
    }


def _git_snapshot(repo_root: Path) -> Dict[str, Any]:
    branch = _run(["git", "branch", "--show-current"], cwd=repo_root)
    head = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    status = _run(["git", "status", "--short"], cwd=repo_root)
    return {
        "branch": str(branch["stdout"]).strip(),
        "head": str(head["stdout"]).strip(),
        "dirty": bool(str(status["stdout"]).strip()),
        "status_count": int(len([x for x in str(status["stdout"]).splitlines() if x.strip()])),
        "status_preview": "\n".join(str(status["stdout"]).splitlines()[:20]),
        "ok": bool(branch["returncode"] == 0 and head["returncode"] == 0 and status["returncode"] == 0),
    }


def _docker_snapshot(repo_root: Path, docker_image: str, require_image_present: bool) -> Dict[str, Any]:
    docker_path = shutil.which("docker")
    if not docker_path:
        return {
            "ok": False,
            "docker_found": False,
            "reason": "docker_not_found",
            "image_present": False,
        }

    ver = _run(["docker", "--version"], cwd=repo_root)
    inspect = _run(["docker", "image", "inspect", str(docker_image)], cwd=repo_root)
    image_present = bool(inspect["returncode"] == 0)
    ok = bool(ver["returncode"] == 0)
    if require_image_present:
        ok = ok and image_present

    return {
        "ok": ok,
        "docker_found": True,
        "docker_path": docker_path,
        "docker_version": str(ver["stdout"]).strip(),
        "image": str(docker_image),
        "image_present": image_present,
        "inspect_stderr_tail": "\n".join(str(inspect["stderr"]).splitlines()[-5:]),
    }


def _nvidia_smi_snapshot(repo_root: Path, device: str) -> Dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return {
            "ok": (device != "cuda"),
            "nvidia_smi_found": False,
            "reason": "nvidia_smi_not_found",
            "summary": "",
        }
    res = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], cwd=repo_root)
    return {
        "ok": bool(res["returncode"] == 0),
        "nvidia_smi_found": True,
        "summary": str(res["stdout"]).strip(),
        "stderr_tail": "\n".join(str(res["stderr"]).splitlines()[-5:]),
    }


def _path_check(path_str: str, must_exist: bool) -> Dict[str, Any]:
    p = Path(path_str)
    return {
        "path": str(p),
        "exists": bool(p.exists()),
        "is_file": bool(p.is_file()),
        "is_dir": bool(p.is_dir()),
        "ok": (bool(p.exists()) if must_exist else True),
    }


def _touch_writable(path: Path) -> Dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".preflight_write_probe"
    try:
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return {"path": str(path), "ok": True}
    except Exception as e:  # pragma: no cover
        return {"path": str(path), "ok": False, "error": str(e)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Cheap preflight checks before paid 4090 runs (v0.6.5).")
    ap.add_argument("--repo_root", type=str, default=".")
    ap.add_argument("--raw_path", type=str, default="data/raw/episodes_swe_v060.jsonl")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--swe_lite_manifest_path", type=str, required=True)
    ap.add_argument("--docker_image", type=str, default="swebench/swebench-lite:latest")
    ap.add_argument("--workspace_root", type=str, default="/tmp/v065_bootstrap")
    ap.add_argument("--results_dir", type=str, default="results/v065_bootstrap")
    ap.add_argument("--device", type=str, default="cuda", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--strict_git_clean", type=str, default="false", choices=["true", "false"])
    ap.add_argument("--require_docker_image_present", type=str, default="true", choices=["true", "false"])
    ap.add_argument("--output_json", type=str, default="results/preflight/v065_4090_preflight.json")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    strict_git_clean = _bool_text(args.strict_git_clean)
    require_image_present = _bool_text(args.require_docker_image_present)
    t0 = time.perf_counter()

    checks: Dict[str, Dict[str, Any]] = {}
    checks["git"] = _git_snapshot(repo_root)
    if strict_git_clean:
        checks["git"]["ok"] = bool(checks["git"]["ok"] and (not checks["git"]["dirty"]))
        checks["git"]["strict_git_clean"] = True
    else:
        checks["git"]["strict_git_clean"] = False

    checks["python"] = {"version": sys.version.split()[0], "ok": True}
    checks["nvidia_smi"] = _nvidia_smi_snapshot(repo_root, device=str(args.device))
    checks["torch_cuda"] = _torch_cuda_snapshot(repo_root, device=str(args.device))
    checks["docker"] = _docker_snapshot(repo_root, docker_image=str(args.docker_image), require_image_present=require_image_present)

    checks["paths"] = {
        "raw_path": _path_check(str(repo_root / args.raw_path), must_exist=True),
        "ckpt": _path_check(str(repo_root / args.ckpt), must_exist=True),
        "swe_lite_manifest_path": _path_check(str(repo_root / args.swe_lite_manifest_path), must_exist=True),
    }

    checks["writable"] = {
        "results_dir": _touch_writable(repo_root / args.results_dir),
        "workspace_root_parent": _touch_writable(Path(args.workspace_root).parent),
    }

    cli_scripts = [
        "scripts/run_v063_m1_main.py",
        "scripts/run_v064_scaling_matrix.py",
        "scripts/summarize_v064_scaling.py",
        "scripts/build_v065_bootstrap_pairs.py",
        "scripts/run_v065_bootstrap_rounds.py",
    ]
    checks["cli_help"] = {
        "items": [_check_cli_help(repo_root, x) for x in cli_scripts],
    }
    checks["cli_help"]["ok"] = all(bool(x.get("ok", False)) for x in checks["cli_help"]["items"])

    path_ok = all(bool(v.get("ok", False)) for v in checks["paths"].values())
    writable_ok = all(bool(v.get("ok", False)) for v in checks["writable"].values())
    passed = all(
        [
            bool(checks["git"]["ok"]),
            bool(checks["python"]["ok"]),
            bool(checks["nvidia_smi"]["ok"]),
            bool(checks["torch_cuda"]["ok"]),
            bool(checks["docker"]["ok"]),
            bool(path_ok),
            bool(writable_ok),
            bool(checks["cli_help"]["ok"]),
        ]
    )

    report = {
        "version": "v0.6.5-preflight-4090",
        "timestamp_unix": int(time.time()),
        "repo_root": str(repo_root),
        "device": str(args.device),
        "strict_git_clean": strict_git_clean,
        "require_docker_image_present": require_image_present,
        "checks": checks,
        "passed": bool(passed),
        "wall_time_sec": float(time.perf_counter() - t0),
        "next_step": (
            "safe_to_start_paid_run" if passed else "fix_failed_checks_before_starting_paid_run"
        ),
    }

    output_path = (repo_root / args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "output_json": str(output_path),
                "passed": bool(passed),
                "next_step": report["next_step"],
            },
            ensure_ascii=False,
        )
    )
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


def _safe_name(text: str) -> str:
    keep = []
    for ch in str(text or ""):
        if ch.isalnum() or ch in {"-", "_", "."}:
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("_")
    return out or "task"


def _repo_url(repo: str) -> str:
    text = str(repo or "").strip()
    if text.startswith("http://") or text.startswith("https://"):
        return text
    if text.endswith(".git") and "/" in text:
        return f"https://github.com/{text}"
    return f"https://github.com/{text}.git"


def _tail(text: str, max_lines: int = 20) -> str:
    lines = str(text or "").splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[-max_lines:])


@dataclass
class TaskRuntime:
    instance_id: str
    repo: str
    base_commit: str
    test_cmd: str
    base_dir: Path
    local_repo_path: str


@dataclass
class ExecResult:
    patch_apply_ok: bool
    tests_passed: bool
    exit_code: int | None
    stderr_tail: str
    elapsed_sec: float
    exec_error_type: str
    stdout_tail: str


class SWEBenchLiteDockerExecutor:
    def __init__(
        self,
        manifest: Dict[str, Dict],
        docker_image: str,
        workspace_root: str,
        max_task_runtime_sec: int = 600,
        keep_failed_workdirs: bool = False,
    ):
        self.manifest = {str(k): dict(v) for k, v in dict(manifest or {}).items()}
        self.docker_image = str(docker_image)
        self.workspace_root = Path(workspace_root)
        self.max_task_runtime_sec = max(int(max_task_runtime_sec), 1)
        self.keep_failed_workdirs = bool(keep_failed_workdirs)
        self.base_root = self.workspace_root / "base"
        self.run_root = self.workspace_root / "runs"
        self.base_root.mkdir(parents=True, exist_ok=True)
        self.run_root.mkdir(parents=True, exist_ok=True)

    def _get_spec(self, instance_id: str) -> Dict:
        spec = self.manifest.get(str(instance_id), None)
        if not spec:
            raise KeyError(f"Instance not found in manifest: {instance_id}")
        return dict(spec)

    def prepare_task(self, instance_id: str) -> TaskRuntime:
        spec = self._get_spec(instance_id)
        repo = str(spec.get("repo") or "").strip()
        base_commit = str(spec.get("base_commit") or "").strip()
        test_cmd = str(spec.get("test_cmd") or "").strip()
        local_repo_path = str(spec.get("local_repo_path") or "").strip()
        if not test_cmd:
            raise RuntimeError(f"Manifest missing test_cmd: {instance_id}")

        base_dir = self.base_root / _safe_name(instance_id)
        marker = base_dir / ".prepared_v063"
        if marker.exists() and (base_dir / ".git").exists():
            return TaskRuntime(
                instance_id=str(instance_id),
                repo=repo,
                base_commit=base_commit,
                test_cmd=test_cmd,
                base_dir=base_dir,
                local_repo_path=local_repo_path,
            )

        if base_dir.exists():
            shutil.rmtree(base_dir)

        if local_repo_path:
            src = Path(local_repo_path)
            if not src.exists():
                raise RuntimeError(f"local_repo_path not found for {instance_id}: {local_repo_path}")
            shutil.copytree(src, base_dir)
        else:
            if not repo:
                raise RuntimeError(f"Manifest missing repo/local_repo_path: {instance_id}")
            base_dir.parent.mkdir(parents=True, exist_ok=True)
            clone_cmd = ["git", "clone", "--depth", "1", _repo_url(repo), str(base_dir)]
            subprocess.run(clone_cmd, check=True, text=True)

        if base_commit and not local_repo_path:
            subprocess.run(["git", "-C", str(base_dir), "fetch", "--depth", "1", "origin", base_commit], check=False, text=True)
        if base_commit:
            subprocess.run(["git", "-C", str(base_dir), "checkout", base_commit], check=True, text=True)

        marker.write_text("prepared\n", encoding="utf-8")
        return TaskRuntime(
            instance_id=str(instance_id),
            repo=repo,
            base_commit=base_commit,
            test_cmd=test_cmd,
            base_dir=base_dir,
            local_repo_path=local_repo_path,
        )

    def run_patch_eval(self, task_runtime: TaskRuntime, patch_text: str) -> ExecResult:
        patch = str(patch_text or "")
        if not patch.strip():
            return ExecResult(
                patch_apply_ok=False,
                tests_passed=False,
                exit_code=3,
                stderr_tail="empty patch",
                elapsed_sec=0.0,
                exec_error_type="empty_patch",
                stdout_tail="",
            )

        run_dir = self.run_root / f"{_safe_name(task_runtime.instance_id)}_{uuid.uuid4().hex[:10]}"
        patch_file = Path(tempfile.mkstemp(prefix="v063_patch_", suffix=".diff")[1])
        start = time.perf_counter()
        stdout = ""
        stderr = ""
        exit_code: int | None = None
        error_type = "none"

        try:
            shutil.copytree(task_runtime.base_dir, run_dir)
            patch_file.write_text(patch, encoding="utf-8")

            script = (
                "set -e; "
                "cd /workspace/repo; "
                "git apply --whitespace=nowarn /tmp/patch.diff || exit 101; "
                f"({task_runtime.test_cmd}) || exit 102; "
                "exit 0"
            )
            cmd = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{run_dir}:/workspace/repo",
                "-v",
                f"{patch_file}:/tmp/patch.diff:ro",
                str(self.docker_image),
                "bash",
                "-lc",
                script,
            ]

            proc = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                timeout=float(self.max_task_runtime_sec),
                check=False,
            )
            stdout = str(proc.stdout or "")
            stderr = str(proc.stderr or "")
            exit_code = int(proc.returncode)
            if exit_code == 0:
                error_type = "none"
            elif exit_code == 101:
                error_type = "patch_apply_failed"
            elif exit_code == 102:
                error_type = "tests_failed"
            else:
                error_type = "runtime_error"
        except FileNotFoundError as e:
            stderr = str(e)
            error_type = "docker_missing"
            exit_code = None
        except subprocess.TimeoutExpired as e:
            stdout = str(e.stdout or "")
            stderr = str(e.stderr or "")
            error_type = "timeout"
            exit_code = 124
        except Exception as e:  # pragma: no cover
            stderr = str(e)
            error_type = "runtime_error"
            exit_code = None
        finally:
            elapsed = float(time.perf_counter() - start)
            if patch_file.exists():
                patch_file.unlink(missing_ok=True)
            if run_dir.exists() and not (self.keep_failed_workdirs and error_type not in {"none", "tests_failed"}):
                shutil.rmtree(run_dir, ignore_errors=True)

        patch_apply_ok = bool(exit_code in {0, 102})
        tests_passed = bool(exit_code == 0)
        return ExecResult(
            patch_apply_ok=patch_apply_ok,
            tests_passed=tests_passed,
            exit_code=exit_code,
            stderr_tail=_tail(stderr),
            elapsed_sec=elapsed,
            exec_error_type=error_type,
            stdout_tail=_tail(stdout),
        )

    def reset_task(self, task_runtime: TaskRuntime) -> None:
        # Stateless runner: each `run_patch_eval` copies from prepared base dir.
        _ = task_runtime

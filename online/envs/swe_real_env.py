from __future__ import annotations

import json
import random
from typing import Any, Dict, List

from ingest.adapters.swe_agent_v060 import iter_episodes
from online.envs.base import OnlineEnv
from online.executors.swebench_lite import SWEBenchLiteDockerExecutor


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _load_manifest(path: str) -> Dict[str, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if str(path).endswith(".jsonl"):
        rows = _read_jsonl(path)
    else:
        payload = _read_json(path)
        if isinstance(payload, list):
            rows = [dict(x) for x in payload if isinstance(x, dict)]
        elif isinstance(payload, dict):
            if isinstance(payload.get("instances"), list):
                rows = [dict(x) for x in payload.get("instances", []) if isinstance(x, dict)]
            else:
                rows = [dict(payload)]

    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        instance_id = str(row.get("instance_id") or row.get("task_id") or row.get("id") or "").strip()
        if not instance_id:
            continue
        out[instance_id] = dict(row)
    return out


class SWERealOnlineEnv(OnlineEnv):
    def __init__(
        self,
        raw_path: str,
        manifest_path: str,
        docker_image: str,
        workspace_root: str,
        max_task_runtime_sec: int = 600,
        max_patch_chars: int = 50000,
        keep_failed_workdirs: bool = False,
    ):
        self.raw_path = str(raw_path)
        self.manifest_path = str(manifest_path)
        self.max_patch_chars = max(int(max_patch_chars), 1)

        manifest = _load_manifest(self.manifest_path)
        if not manifest:
            raise RuntimeError(f"Empty or invalid manifest: {self.manifest_path}")

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for ep in iter_episodes(self.raw_path):
            raw_meta = dict(ep.get("raw_meta") or {})
            instance_id = str(raw_meta.get("instance_id") or ep.get("task_id") or "").strip()
            if not instance_id:
                continue
            if instance_id not in manifest:
                continue
            grouped.setdefault(instance_id, []).append(ep)

        self._tasks: Dict[str, List[Dict[str, Any]]] = {}
        for instance_id, rows in grouped.items():
            candidates: List[Dict[str, Any]] = []
            for idx, row in enumerate(rows):
                raw_meta = dict(row.get("raw_meta") or {})
                patch_text = str(raw_meta.get("generated_patch") or "").strip()
                candidates.append(
                    {
                        "candidate_id": f"{instance_id}__cand_{idx:03d}",
                        "instance_id": instance_id,
                        "instruction_refined": str(row.get("instruction_refined") or row.get("instruction_raw") or "").strip(),
                        "steps": [str(x) for x in row.get("steps", []) if str(x).strip()],
                        "patch_text": patch_text,
                        "raw_meta": raw_meta,
                    }
                )
            if candidates:
                self._tasks[instance_id] = candidates

        if not self._tasks:
            raise RuntimeError("No runnable task intersection between raw episodes and manifest.")

        self._manifest = manifest
        self._executor = SWEBenchLiteDockerExecutor(
            manifest=self._manifest,
            docker_image=str(docker_image),
            workspace_root=str(workspace_root),
            max_task_runtime_sec=int(max_task_runtime_sec),
            keep_failed_workdirs=bool(keep_failed_workdirs),
        )

        self._task_id = ""
        self._candidates: Dict[str, Dict[str, Any]] = {}
        self._history: List[Dict[str, Any]] = []
        self._done = False
        self._success = False
        self._seed = 0

    def list_task_ids(self) -> List[str]:
        return sorted(self._tasks.keys())

    def reset(self, task_id: str, seed: int) -> Dict:
        key = str(task_id or "").strip()
        if key not in self._tasks:
            raise KeyError(f"Unknown task_id/instance_id: {key}")

        rows = list(self._tasks[key])
        rng = random.Random(int(seed))
        rng.shuffle(rows)

        self._task_id = key
        self._seed = int(seed)
        self._candidates = {str(x["candidate_id"]): dict(x) for x in rows}
        self._history = []
        self._done = False
        self._success = False
        return self._state()

    def _state(self) -> Dict[str, Any]:
        return {
            "task_id": self._task_id,
            "instance_id": self._task_id,
            "done": bool(self._done),
            "success": bool(self._success),
            "history_len": len(self._history),
            "candidate_count": len(self._candidates),
        }

    def get_task_context(self) -> Dict[str, Any]:
        templates: List[Dict[str, Any]] = []
        for candidate_id in sorted(self._candidates.keys()):
            cand = self._candidates[candidate_id]
            first_step = ""
            if cand.get("steps"):
                first_step = str(cand["steps"][0]).strip()
            templates.append(
                {
                    "candidate_id": candidate_id,
                    "next_step": first_step,
                    "remaining_steps": 1,
                    "candidate_success": None,
                    "instance_id": self._task_id,
                    "patch_text": str(cand.get("patch_text") or ""),
                }
            )

        instruction = ""
        if templates:
            c0 = self._candidates[templates[0]["candidate_id"]]
            instruction = str(c0.get("instruction_refined") or "").strip()

        return {
            "task_id": self._task_id,
            "instance_id": self._task_id,
            "instruction_refined": instruction,
            "candidate_templates": templates,
            "manifest": dict(self._manifest.get(self._task_id) or {}),
        }

    def step(self, action: Dict) -> tuple[Dict, float | None, bool, Dict]:
        if self._done:
            raise RuntimeError("Environment already done; call reset() first.")

        candidate_id = str((action or {}).get("candidate_id") or "").strip()
        if candidate_id not in self._candidates:
            raise KeyError(f"Unknown candidate_id: {candidate_id}")

        cand = self._candidates[candidate_id]
        patch_text = str((action or {}).get("patch_text") or cand.get("patch_text") or "")
        if len(patch_text) > self.max_patch_chars:
            patch_text = patch_text[: self.max_patch_chars]

        committed_text = str((action or {}).get("action_text") or "").strip()
        if not committed_text:
            committed_text = str(cand.get("steps", [""])[0] if cand.get("steps") else "").strip()

        exec_error_type = "none"
        patch_apply_ok = False
        tests_passed = False
        exit_code = None
        stderr_tail = ""
        elapsed_sec = 0.0

        try:
            runtime = self._executor.prepare_task(self._task_id)
            result = self._executor.run_patch_eval(task_runtime=runtime, patch_text=patch_text)
            self._executor.reset_task(runtime)
            patch_apply_ok = bool(result.patch_apply_ok)
            tests_passed = bool(result.tests_passed)
            exit_code = result.exit_code
            stderr_tail = str(result.stderr_tail or "")
            elapsed_sec = float(result.elapsed_sec)
            exec_error_type = str(result.exec_error_type or "none")
        except Exception as e:
            exec_error_type = "prepare_failed"
            stderr_tail = str(e)

        self._done = True
        self._success = bool(tests_passed)
        self._history.append(
            {
                "candidate_id": candidate_id,
                "instance_id": self._task_id,
                "text": committed_text,
                "patch_chars": len(patch_text),
                "patch_apply_ok": patch_apply_ok,
                "tests_passed": tests_passed,
                "exit_code": exit_code,
                "exec_error_type": exec_error_type,
            }
        )

        info = {
            "task_id": self._task_id,
            "instance_id": self._task_id,
            "candidate_id": candidate_id,
            "committed_text": committed_text,
            "committed_steps_len": 1,
            "patch_chars": len(patch_text),
            "patch_apply_ok": bool(patch_apply_ok),
            "tests_passed": bool(tests_passed),
            "exit_code": exit_code,
            "stderr_tail": stderr_tail,
            "elapsed_sec": float(elapsed_sec),
            "exec_error_type": exec_error_type,
            "candidate_done": True,
            "candidate_success": bool(tests_passed),
        }
        reward = 1.0 if tests_passed else 0.0
        return self._state(), reward, True, info

    def is_done(self) -> bool:
        return bool(self._done)

    def get_observation_text(self) -> str:
        lines: List[str] = []
        ctx = self.get_task_context()
        instruction = str(ctx.get("instruction_refined") or "").strip()
        if instruction:
            lines.append(f"Task: {instruction}")
        for idx, item in enumerate(self._history, start=1):
            lines.append(f"Step {idx}: {str(item.get('text') or '').strip()}")
        return "\n".join(lines).strip()

    def snapshot(self) -> bytes:
        payload = {
            "task_id": self._task_id,
            "seed": self._seed,
            "history": self._history,
            "done": self._done,
            "success": self._success,
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")

    def restore(self, blob: bytes) -> None:
        payload = json.loads(bytes(blob).decode("utf-8"))
        self._task_id = str(payload.get("task_id") or "")
        self._seed = int(payload.get("seed", 0))
        self._history = [dict(x) for x in list(payload.get("history") or [])]
        self._done = bool(payload.get("done", False))
        self._success = bool(payload.get("success", False))

    def render_history(self) -> List[Dict]:
        return [dict(x) for x in self._history]

    def get_terminal_success(self) -> bool:
        return bool(self._success)

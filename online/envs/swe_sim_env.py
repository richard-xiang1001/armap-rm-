from __future__ import annotations

import json
import random
from typing import Any, Dict, List

from ingest.adapters.swe_agent_v060 import iter_episodes
from online.envs.base import OnlineEnv


def _episode_to_candidate(episode: Dict[str, Any], index: int) -> Dict[str, Any]:
    steps = [str(x) for x in episode.get("steps", []) if str(x).strip()]
    raw_meta = dict(episode.get("raw_meta") or {})
    instance_id = str(raw_meta.get("instance_id") or episode.get("task_id") or "").strip()
    patch_text = str(raw_meta.get("generated_patch") or "").strip()
    return {
        "candidate_id": f"{episode.get('task_id', 'task')}__cand_{index:03d}",
        "episode_id": episode.get("episode_id"),
        "run_id": episode.get("run_id"),
        "instruction_refined": str(episode.get("instruction_refined") or episode.get("instruction_raw") or "").strip(),
        "steps": steps,
        "success": bool(episode.get("success") is True),
        "raw_meta": raw_meta,
        "instance_id": instance_id,
        "patch_text": patch_text,
    }


class SWESimOnlineEnv(OnlineEnv):
    def __init__(self, raw_path: str):
        self.raw_path = str(raw_path)
        self._tasks: Dict[str, List[Dict[str, Any]]] = {}
        for ep in iter_episodes(self.raw_path):
            task_id = str(ep.get("task_id") or "").strip()
            if not task_id:
                continue
            self._tasks.setdefault(task_id, []).append(ep)
        if not self._tasks:
            raise RuntimeError(f"No normalized SWE episodes found: {self.raw_path}")

        self._task_id: str = ""
        self._candidates: Dict[str, Dict[str, Any]] = {}
        self._pointers: Dict[str, int] = {}
        self._history: List[Dict[str, Any]] = []
        self._done = False
        self._success = False
        self._seed = 0

    def list_task_ids(self) -> List[str]:
        return sorted(self._tasks.keys())

    def reset(self, task_id: str, seed: int) -> Dict:
        task_key = str(task_id or "").strip()
        if task_key not in self._tasks:
            raise KeyError(f"Unknown task_id: {task_key}")

        rows = list(self._tasks[task_key])
        rng = random.Random(int(seed))
        rng.shuffle(rows)

        candidates: Dict[str, Dict[str, Any]] = {}
        for idx, row in enumerate(rows):
            cand = _episode_to_candidate(row, index=idx)
            candidates[cand["candidate_id"]] = cand

        self._task_id = task_key
        self._seed = int(seed)
        self._candidates = candidates
        self._pointers = {cid: 0 for cid in candidates.keys()}
        self._history = []
        self._done = False
        self._success = False
        return self._state()

    def _state(self) -> Dict[str, Any]:
        return {
            "task_id": self._task_id,
            "done": bool(self._done),
            "success": bool(self._success),
            "history_len": len(self._history),
            "candidate_count": len(self._candidates),
        }

    def get_task_context(self) -> Dict[str, Any]:
        templates: List[Dict[str, Any]] = []
        for candidate_id in sorted(self._candidates.keys()):
            cand = self._candidates[candidate_id]
            ptr = int(self._pointers.get(candidate_id, 0))
            next_step = ""
            if ptr < len(cand["steps"]):
                next_step = str(cand["steps"][ptr]).strip()
            templates.append(
                {
                    "candidate_id": candidate_id,
                    "next_step": next_step,
                    "remaining_steps": max(len(cand["steps"]) - ptr, 0),
                    "candidate_success": bool(cand["success"]),
                    "instance_id": str(cand.get("instance_id") or self._task_id),
                    "patch_text": str(cand.get("patch_text") or ""),
                }
            )

        instruction = ""
        if templates:
            first_id = templates[0]["candidate_id"]
            instruction = str(self._candidates[first_id].get("instruction_refined") or "").strip()

        return {
            "task_id": self._task_id,
            "instruction_refined": instruction,
            "candidate_templates": templates,
        }

    def step(self, action: Dict) -> tuple[Dict, float | None, bool, Dict]:
        if self._done:
            raise RuntimeError("Environment already done; call reset() for next task.")

        candidate_id = str((action or {}).get("candidate_id") or "").strip()
        if candidate_id not in self._candidates:
            raise KeyError(f"Unknown candidate_id: {candidate_id}")

        delta_steps = int((action or {}).get("delta_steps", 1))
        delta_steps = max(delta_steps, 1)

        cand = self._candidates[candidate_id]
        ptr = int(self._pointers.get(candidate_id, 0))
        nxt = min(ptr + delta_steps, len(cand["steps"]))
        committed = cand["steps"][ptr:nxt]

        for idx, text in enumerate(committed, start=1):
            self._history.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_step_index": ptr + idx,
                    "text": str(text),
                }
            )

        self._pointers[candidate_id] = nxt
        candidate_done = bool(nxt >= len(cand["steps"]))
        if candidate_done:
            self._done = True
            self._success = bool(cand["success"])

        reward = 1.0 if (candidate_done and cand["success"]) else 0.0 if candidate_done else None
        info = {
            "task_id": self._task_id,
            "instance_id": str(cand.get("instance_id") or self._task_id),
            "candidate_id": candidate_id,
            "delta_steps": delta_steps,
            "committed_steps": [str(x) for x in committed],
            "committed_text": "\n".join(str(x) for x in committed if str(x).strip()),
            "committed_steps_len": len(committed),
            "patch_chars": len(str(cand.get("patch_text") or "")),
            "patch_apply_ok": None,
            "tests_passed": None,
            "exit_code": None,
            "stderr_tail": "",
            "elapsed_sec": 0.0,
            "exec_error_type": "simulated",
            "candidate_ptr": int(nxt),
            "candidate_total_steps": len(cand["steps"]),
            "candidate_done": candidate_done,
            "candidate_success": bool(cand["success"]) if candidate_done else None,
        }
        return self._state(), reward, bool(self._done), info

    def is_done(self) -> bool:
        return bool(self._done)

    def get_observation_text(self) -> str:
        ctx = self.get_task_context()
        instruction = str(ctx.get("instruction_refined") or "").strip()
        lines: List[str] = []
        if instruction:
            lines.append(f"Task: {instruction}")
        for idx, item in enumerate(self._history, start=1):
            lines.append(f"Step {idx}: {str(item.get('text') or '').strip()}")
        return "\n".join(lines).strip()

    def snapshot(self) -> bytes:
        payload = {
            "task_id": self._task_id,
            "seed": self._seed,
            "pointers": self._pointers,
            "history": self._history,
            "done": self._done,
            "success": self._success,
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")

    def restore(self, blob: bytes) -> None:
        payload = json.loads(bytes(blob).decode("utf-8"))
        self._task_id = str(payload["task_id"])
        self._seed = int(payload.get("seed", 0))
        self._pointers = {str(k): int(v) for k, v in dict(payload.get("pointers") or {}).items()}
        self._history = [dict(x) for x in list(payload.get("history") or [])]
        self._done = bool(payload.get("done", False))
        self._success = bool(payload.get("success", False))

    def render_history(self) -> List[Dict]:
        return [dict(x) for x in self._history]

    def get_terminal_success(self) -> bool:
        return bool(self._success)

from __future__ import annotations

import random
from typing import Dict, List


def _perturb(text: str, variant: str) -> str:
    raw = " ".join(str(text or "").split())
    if not raw:
        return ""
    if variant == "identity":
        return raw
    if variant == "trim":
        return raw[: max(8, min(len(raw), 96))].rstrip()
    if variant == "focus":
        return f"Focus on this next step: {raw}"
    if variant == "annotate":
        return f"Candidate action -> {raw}"
    return raw


def propose_actions(obs_text: str, history: list[dict], n: int, seed: int, task_ctx: dict) -> List[Dict]:
    del obs_text
    del history

    count = max(int(n), 1)
    ctx = dict(task_ctx or {})
    templates = [dict(x) for x in list(ctx.get("candidate_templates") or []) if str(x.get("candidate_id") or "").strip()]
    if not templates:
        return []

    rng = random.Random(int(seed))
    templates_sorted = sorted(templates, key=lambda x: str(x.get("candidate_id", "")))
    rng.shuffle(templates_sorted)

    variants = ["identity", "trim", "focus", "annotate"]
    actions: List[Dict] = []
    cursor = 0
    while len(actions) < count and templates_sorted:
        tpl = templates_sorted[cursor % len(templates_sorted)]
        cursor += 1
        variant = variants[len(actions) % len(variants)]
        template_id = str(tpl.get("candidate_id") or "")
        next_step = str(tpl.get("next_step") or "").strip()
        patch_text = str(tpl.get("patch_text") or "").strip()
        instance_id = str(tpl.get("instance_id") or ctx.get("task_id") or "").strip()
        if not next_step:
            next_step = f"continue with {template_id}"
        action_text = _perturb(next_step, variant=variant)
        actions.append(
            {
                "candidate_id": template_id,
                "delta_steps": 1,
                "action_text": action_text,
                "patch_text": patch_text,
                "instance_id": instance_id,
                "source": "template_perturb",
                "meta": {
                    "template_id": template_id,
                    "variant": variant,
                    "remaining_steps": int(tpl.get("remaining_steps", 0)),
                    "noise_seed": int(seed),
                    "rank": len(actions) + 1,
                },
            }
        )

    return actions

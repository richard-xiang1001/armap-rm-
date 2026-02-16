from __future__ import annotations

import random
from typing import Dict, List

from ingest.negatives import replay_corrupt, swap_step, truncate


NEG_ORDER = ("replay_corrupt", "swap_step", "truncate_last_k")


def _run_generator(name: str, episode: Dict, rng: random.Random, tail_drop_steps: int) -> Dict | None:
    if name == "replay_corrupt":
        return replay_corrupt.generate(episode=episode, rng=rng)
    if name == "swap_step":
        return swap_step.generate(episode=episode, rng=rng)
    if name == "truncate_last_k":
        return truncate.generate(episode=episode, rng=rng, tail_drop_steps=tail_drop_steps)
    raise ValueError(f"Unknown negative generator: {name}")


def generate_negative_with_fallback(
    episode: Dict,
    rng: random.Random,
    tail_drop_steps: int = 2,
    max_attempts: int = 2,
    forced_neg_type: str = "",
) -> Dict | None:
    """Try replay_corrupt -> swap_step -> truncate_last_k until replayable failure.

    max_attempts controls how many generated candidates we accept for each positive trajectory.
    In v0.3.0 default usage we still emit one pair per positive sample.
    """

    attempted: List[str] = []
    candidates: List[str]
    if forced_neg_type:
        candidates = [forced_neg_type]
    else:
        candidates = list(NEG_ORDER)

    tries = 0
    for neg_name in candidates:
        if tries >= max_attempts:
            break
        tries += 1
        attempted.append(neg_name)
        generated = _run_generator(name=neg_name, episode=episode, rng=rng, tail_drop_steps=tail_drop_steps)
        if generated is None:
            continue

        generated["attempted_neg_types"] = attempted
        generated["replay_attempted"] = "replay_corrupt" in attempted

        # If we requested a specific type, return it directly.
        if forced_neg_type:
            return generated

        # For fallback mode, prefer replayable failures.
        if generated.get("replay_ok"):
            return generated

    # Last fallback: force truncate and return whatever we can get.
    if not forced_neg_type:
        generated = _run_generator(name="truncate_last_k", episode=episode, rng=rng, tail_drop_steps=tail_drop_steps)
        if generated is not None:
            attempted.append("truncate_last_k")
            generated["attempted_neg_types"] = attempted
            generated["replay_attempted"] = "replay_corrupt" in attempted
        return generated

    return None

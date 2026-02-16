from __future__ import annotations

from ingest.envs.arithmetic_env import ArithmeticEnvJudge
from ingest.envs.protocol import EnvJudge
from ingest.envs.real_logs_env import RealLogsEnvJudge


def get_env_judge(name: str) -> EnvJudge:
    normalized = str(name or "").strip().lower()
    if normalized in {"", "arithmetic"}:
        return ArithmeticEnvJudge()
    if normalized in {"real_logs", "real_logs_v1"}:
        return RealLogsEnvJudge()
    raise ValueError(f"Unsupported env judge: {name}")


def supported_env_judges() -> list[str]:
    return ["arithmetic", "real_logs"]

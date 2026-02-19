from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class OnlineEnv(ABC):
    @abstractmethod
    def reset(self, task_id: str, seed: int) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Dict) -> Tuple[Dict, float | None, bool, Dict]:
        raise NotImplementedError

    @abstractmethod
    def is_done(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_observation_text(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def snapshot(self) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def restore(self, blob: bytes) -> None:
        raise NotImplementedError

    @abstractmethod
    def render_history(self) -> List[Dict]:
        raise NotImplementedError

    @abstractmethod
    def get_task_context(self) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def get_terminal_success(self) -> bool:
        raise NotImplementedError

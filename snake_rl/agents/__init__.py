"""Players: non-learning baselines, tabular Q-learning, DQN.

Every player implements the :class:`Policy` protocol used by
``snake_rl.evaluate``: ``act(obs, env) -> action``. Baselines read the full
environment state through ``env``; learned agents use only ``obs``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from snake_rl.env.core import SnakeEnv


@runtime_checkable
class Policy(Protocol):
    name: str

    def act(self, obs: np.ndarray, env: SnakeEnv) -> int: ...

    def reset(self) -> None:
        """Called at the start of every evaluation episode."""
        ...

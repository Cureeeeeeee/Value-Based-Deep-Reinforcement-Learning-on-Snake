"""Snake environments: scalar reference (``core``) and vectorised (``vec_env``).

Importing this package registers ``Snake-v0`` with Gymnasium so that
``gymnasium.make("Snake-v0", cfg=EnvConfig(...))`` works.
"""

from __future__ import annotations

import gymnasium as gym

from .core import EnvConfig, SnakeEnv
from .obs import ObsConfig
from .rewards import RewardConfig

__all__ = ["EnvConfig", "ObsConfig", "RewardConfig", "SnakeEnv"]

if "Snake-v0" not in gym.registry:
    gym.register(id="Snake-v0", entry_point="snake_rl.env.core:SnakeEnv")

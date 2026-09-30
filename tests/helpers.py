"""Hand-built board positions shared by the environment, observation and
reward tests. Cells are listed tail -> head; the heading is implied by the
last two cells. Coordinates assume W >= 8 unless stated.
"""

from __future__ import annotations

import numpy as np

from snake_rl.env import EnvConfig, ObsConfig, RewardConfig, SnakeEnv
from snake_rl.env.common import STRAIGHT, TURN_LEFT, TURN_RIGHT

FAR_FOOD = (0, 0)


def make_env(W: int = 10, **kwargs) -> SnakeEnv:
    env = SnakeEnv(EnvConfig(W=W, **kwargs))
    env.reset(seed=0)
    return env


def random_episode(env: SnakeEnv, seed: int, max_steps: int = 10_000):
    """Play uniformly random actions until the episode ends.

    Yields ``(action, obs, reward, terminated, truncated, info)`` per step.
    """
    rng = np.random.default_rng(seed)
    for _ in range(max_steps):
        action = int(rng.integers(env.action_space.n))
        result = env.step(action)
        yield (action, *result)
        if result[2] or result[3]:
            return
    raise AssertionError("episode did not end")


# ----------------------------------------------------------------- positions
#: Straight snake in the middle heading right; head at (5, 5).
STRAIGHT_RIGHT = [(5, 3), (5, 4), (5, 5)]

#: Length 5 hook. Head (5, 4) heading LEFT; turning right (-> up) enters
#: (4, 4), which has age 2 -> self collision.
HOOK_SELF_COLLISION = [(3, 4), (4, 4), (4, 5), (5, 5), (5, 4)]
HOOK_COLLIDING_ACTION = TURN_RIGHT

#: Length 4 in a 2x2 block. Head (5, 4) heading LEFT; turning right (-> up)
#: enters the tail cell (4, 4), which vacates this step -> legal.
BLOCK_TAIL_CHASE = [(4, 4), (4, 5), (5, 5), (5, 4)]
BLOCK_TAIL_CHASE_ACTION = TURN_RIGHT

#: Same 2x2 cell set and same head as BLOCK_TAIL_CHASE but traversed the
#: other way round: heading DOWN. Binary planes identical, heading differs.
BLOCK_OTHER_HEADING = [(5, 5), (4, 5), (4, 4), (5, 4)]

#: Two length-6 snakes with identical cell set, head (5, 3) and heading LEFT
#: but different tails ((3, 4) vs (4, 5)). Only body age tells them apart,
#: and after one move different cells vacate. This is the Markov argument
#: for the age plane (PLAN.md section 0.2).
COIL_TAIL_A = [(3, 4), (3, 5), (4, 5), (4, 4), (5, 4), (5, 3)]
COIL_TAIL_B = [(4, 5), (3, 5), (3, 4), (4, 4), (5, 4), (5, 3)]


def serpentine_almost_full(W: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    """Snake covering all but one cell of a WxW board, plus the food on that
    cell, with the head adjacent to it. Eating wins the game."""
    cells: list[tuple[int, int]] = []
    for r in range(W):
        cols = range(W) if r % 2 == 0 else range(W - 1, -1, -1)
        cells.extend((r, c) for c in cols)
    food = cells[-1]
    return cells[:-1], food


__all__ = [
    "BLOCK_OTHER_HEADING",
    "BLOCK_TAIL_CHASE",
    "BLOCK_TAIL_CHASE_ACTION",
    "COIL_TAIL_A",
    "COIL_TAIL_B",
    "FAR_FOOD",
    "HOOK_COLLIDING_ACTION",
    "HOOK_SELF_COLLISION",
    "STRAIGHT",
    "STRAIGHT_RIGHT",
    "TURN_LEFT",
    "TURN_RIGHT",
    "EnvConfig",
    "ObsConfig",
    "RewardConfig",
    "SnakeEnv",
    "make_env",
    "random_episode",
    "serpentine_almost_full",
]

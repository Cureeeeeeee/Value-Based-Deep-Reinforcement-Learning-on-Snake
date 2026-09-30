"""Non-learning reference players (PLAN.md section 3.8).

They fix the coordinate system in which the learned agents are judged:

* :class:`RandomPolicy` - uniform over the action space (floor).
* :class:`BFSGreedyPolicy` - shortest path to the food by breadth-first
  search on the unweighted grid (identical to A* here; called BFS to avoid
  the heuristic question). Time-aware: a body cell of age ``k`` is passable
  if the head arrives after at least ``k`` steps, exactly the rule ``step``
  applies. It ignores self-trapping. No path -> first safe action in the
  order straight / left / right; nothing safe -> straight (and die).
* :class:`HamiltonianPolicy` - follows a fixed Hamiltonian cycle (even ``W``
  only: serpentine over columns ``1..W-1``, return along column 0). It never
  collides and, because ``H >= W^2``, never starves, so it eats every food
  and wins: the planning upper bound in score and the lower bound in speed.
  No shortcuts.

Policies see the whole environment (``act(obs, env)``); learned agents use
only ``obs``.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from snake_rl.env.common import (
    DELTAS,
    RELATIVE_TURN,
    STRAIGHT,
    heading_to_action,
)
from snake_rl.env.core import SnakeEnv
from snake_rl.env.obs import cell_is_danger

BASELINE_NAMES = ("random", "bfs", "hamiltonian")


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int | None = 0) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, env: SnakeEnv) -> int:
        return int(self._rng.integers(env.action_space.n))


# --------------------------------------------------------------------- BFS
def bfs_first_heading(body_age: np.ndarray, head: tuple[int, int], food: tuple[int, int]) -> int | None:
    """Heading of the first step of a shortest head -> food path, or ``None``.

    A cell reached at BFS depth ``d`` is passable if ``body_age <= d``: the
    tail vacates one cell per step and no food is eaten before the goal, so
    the occupancy at arrival is exactly the current age minus ``d``.
    """
    W = body_age.shape[0]
    dist = np.full((W, W), -1, dtype=np.int32)
    first = np.full((W, W), -1, dtype=np.int32)
    dist[head] = 0
    queue: deque[tuple[int, int]] = deque([head])
    while queue:
        r, c = queue.popleft()
        d = int(dist[r, c])
        if (r, c) == food:
            return int(first[r, c])
        for h in range(4):
            nr, nc = r + int(DELTAS[h, 0]), c + int(DELTAS[h, 1])
            if not (0 <= nr < W and 0 <= nc < W) or dist[nr, nc] != -1:
                continue
            if body_age[nr, nc] > d + 1:
                continue
            dist[nr, nc] = d + 1
            first[nr, nc] = h if d == 0 else first[r, c]
            queue.append((nr, nc))
    return None


def _is_safe(env: SnakeEnv, heading: int) -> bool:
    head = np.array([env.head])
    return not bool(cell_is_danger(env.body_age[None], head + DELTAS[heading])[0])


class BFSGreedyPolicy:
    name = "bfs"

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, env: SnakeEnv) -> int:
        desired = bfs_first_heading(env.body_age, env.head, env.food)
        if desired is None:
            for turn in RELATIVE_TURN:  # straight, left, right
                h = int((env.heading + turn) % 4)
                if _is_safe(env, h):
                    desired = h
                    break
            else:
                return STRAIGHT if env.cfg.action_mode == "relative" else int(env.heading)
        return heading_to_action(env.heading, desired, env.cfg.action_mode)


# ------------------------------------------------------------- Hamiltonian
def hamiltonian_cycle(W: int) -> np.ndarray:
    """``(W*W, 2)`` cells of a fixed Hamiltonian cycle on an even ``W`` grid.

    Column 0 is the return path (bottom -> top); rows sweep columns
    ``1..W-1`` alternately right / left starting at ``(0, 1)``. Consecutive
    entries are adjacent and the last entry ``(1, 0)`` is adjacent to the
    first ``(0, 0)``.
    """
    if W < 4 or W % 2:
        raise ValueError("a Hamiltonian cycle of this construction needs even W >= 4")
    cells: list[tuple[int, int]] = [(0, 0)]
    for r in range(W):
        cols = range(1, W) if r % 2 == 0 else range(W - 1, 0, -1)
        cells.extend((r, c) for c in cols)
    cells.extend((r, 0) for r in range(W - 1, 0, -1))
    return np.array(cells, dtype=np.int64)


class HamiltonianPolicy:
    """Follow the cycle in whichever direction does not run into the neck."""

    name = "hamiltonian"

    def __init__(self) -> None:
        self._W: int | None = None
        self._cycle: np.ndarray | None = None
        self._index: np.ndarray | None = None

    def reset(self) -> None:
        pass

    def _prepare(self, W: int) -> None:
        if self._W != W:
            self._cycle = hamiltonian_cycle(W)
            self._index = np.empty((W, W), dtype=np.int64)
            self._index[self._cycle[:, 0], self._cycle[:, 1]] = np.arange(W * W)
            self._W = W

    def act(self, obs: np.ndarray, env: SnakeEnv) -> int:
        self._prepare(env.W)
        assert self._cycle is not None and self._index is not None
        n = env.W * env.W
        head = env.head
        i = int(self._index[head])
        neck = (head[0] - int(DELTAS[env.heading, 0]), head[1] - int(DELTAS[env.heading, 1]))
        target = tuple(int(v) for v in self._cycle[(i + 1) % n])
        if target == neck:
            target = tuple(int(v) for v in self._cycle[(i - 1) % n])
        delta = (target[0] - head[0], target[1] - head[1])
        desired = next(h for h in range(4) if tuple(DELTAS[h].tolist()) == delta)
        return heading_to_action(env.heading, desired, env.cfg.action_mode)


def make_policy(name: str, seed: int | None = 0):
    if name == "random":
        return RandomPolicy(seed)
    if name == "bfs":
        return BFSGreedyPolicy()
    if name == "hamiltonian":
        return HamiltonianPolicy()
    raise ValueError(f"unknown baseline {name!r}; choose from {BASELINE_NAMES}")

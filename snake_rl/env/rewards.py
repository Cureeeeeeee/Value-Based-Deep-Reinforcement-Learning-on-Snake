"""Reward variants R0-R3 shared by the scalar and vectorised environments
(PLAN.md section 3.5, the second experimental axis).

===  =====================================================================
R0   +1 food, -1 death (wall / self), 0 otherwise. Starvation truncation
     is *not* a death and gets no -1; the DQN target must bootstrap it.
R1   R0 - step_penalty on every step.
R2   R0 + gamma * Phi(s') - Phi(s), Phi(s) = -manhattan(head, food) /
     (2 (W - 1)), Phi(terminal) = 0. Potential-based shaping (Ng et al.
     1999): optimal policy unchanged. After eating, Phi(s') uses the new
     food; still a valid potential. ``gamma`` must equal the agent's
     discount for the guarantee to hold.
R3   R0 + closer if the move reduced the distance to the food that was on
     the board at the start of the step, else + farther. Naive distance
     shaping. Default is the asymmetric version (closer +0.01, farther 0)
     so that oscillating is a net gain and reward hacking is possible;
     ``farther = -0.01`` gives the symmetric version written in Part B,
     which equals potential shaping with gamma = 1 and cannot be hacked.
===  =====================================================================

The Manhattan distance to a fixed target changes by exactly +-1 per move,
so the R3 shaping term is never zero on a non-eating step.

Inputs are batched arrays (or scalars); output is float32.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

RewardVariant = Literal["R0", "R1", "R2", "R3"]


@dataclass(frozen=True)
class RewardConfig:
    variant: RewardVariant = "R0"
    food: float = 1.0
    death: float = -1.0
    #: R1 per-step penalty (subtracted).
    step_penalty: float = 0.01
    #: R2 discount; must match the agent's gamma.
    gamma: float = 0.99
    #: R3 bonus for moving closer / farther.
    closer: float = 0.01
    farther: float = 0.0


def potential(dist, W: int) -> np.ndarray:
    """Phi(s) = -d / (2 (W - 1)) in [-1, 0]; 2 (W - 1) is the largest distance."""
    return -np.asarray(dist, dtype=np.float32) / np.float32(2 * (W - 1))


def compute_reward(
    cfg: RewardConfig,
    W: int,
    *,
    ate,
    died,
    terminal,
    d_prev,
    d_moved,
    d_next,
) -> np.ndarray:
    """Reward for one transition (batched).

    Parameters
    ----------
    ate       : bool, the head reached the food.
    died      : bool, terminated by wall or self collision (gets ``death``).
    terminal  : bool, ``terminated`` for any cause (wall / self / win);
                Phi(s') = 0. Truncation is *not* terminal here.
    d_prev    : Manhattan(head, food) before the move.
    d_moved   : Manhattan(new_head, food-before-move); always d_prev +- 1.
    d_next    : Manhattan(new_head, food-after-move); differs from d_moved
                only after eating. Ignored where ``terminal``.
    """
    ate = np.asarray(ate, dtype=bool)
    died = np.asarray(died, dtype=bool)
    terminal = np.asarray(terminal, dtype=bool)
    d_prev = np.asarray(d_prev, dtype=np.int64)
    d_moved = np.asarray(d_moved, dtype=np.int64)
    d_next = np.asarray(d_next, dtype=np.int64)

    zero = np.float32(0.0)
    r = np.where(ate, np.float32(cfg.food), zero) + np.where(died, np.float32(cfg.death), zero)
    if cfg.variant == "R0":
        return r.astype(np.float32)
    if cfg.variant == "R1":
        return (r - np.float32(cfg.step_penalty)).astype(np.float32)
    if cfg.variant == "R2":
        phi_next = np.where(terminal, zero, potential(d_next, W))
        return (r + np.float32(cfg.gamma) * phi_next - potential(d_prev, W)).astype(np.float32)
    if cfg.variant == "R3":
        shaping = np.where(d_moved < d_prev, np.float32(cfg.closer), np.float32(cfg.farther))
        return (r + shaping).astype(np.float32)
    raise ValueError(f"unknown reward variant {cfg.variant!r}")

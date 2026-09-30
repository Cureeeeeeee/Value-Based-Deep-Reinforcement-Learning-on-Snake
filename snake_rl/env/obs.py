"""Observation encoders shared by the scalar and vectorised environments.

Two representations (the first experimental axis, PLAN.md section 3.4):

* **A** - 11 binary features: danger straight / left / right (3), heading
  one-hot (4), food up / down / left / right of the head in absolute
  coordinates (4). This is the community-standard hand-crafted Snake
  representation and is cited as related work. It is *not* Markov: the body
  layout is discarded.
* **B** - multi-plane grid ``(C, W, W)`` float32:
  ``[head, body, food, body_age / length, heading x 4 (broadcast)]`` = 8
  planes. Config switches drop the age plane and/or the heading planes; the
  3-plane version ``[head, body, food]`` is the non-Markov ablation. The age
  plane is the one that makes the state Markov (it recovers both the neck,
  hence the heading, and the tail, hence which cell vacates next); the heading
  planes are redundant given the age plane and exist only to save the CNN a
  step.

All encoders are batched: inputs carry a leading ``N`` axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .common import DELTAS, head_positions

N_FEATURES_A = 11
#: Feature indices of observation A (kept next to the encoder for the GUI).
A_DANGER_STRAIGHT, A_DANGER_LEFT, A_DANGER_RIGHT = 0, 1, 2
A_HEADING = slice(3, 7)  # one-hot, heading order up/right/down/left
A_FOOD_UP, A_FOOD_DOWN, A_FOOD_LEFT, A_FOOD_RIGHT = 7, 8, 9, 10


@dataclass(frozen=True)
class ObsConfig:
    kind: Literal["A", "B"] = "A"
    #: B only: include ``body_age / length`` plane (Markov fix).
    age_plane: bool = True
    #: B only: include 4 broadcast heading planes (redundant given age plane).
    heading_planes: bool = True

    @property
    def n_planes(self) -> int:
        return 3 + int(self.age_plane) + 4 * int(self.heading_planes)

    def shape(self, W: int) -> tuple[int, ...]:
        if self.kind == "A":
            return (N_FEATURES_A,)
        if self.kind == "B":
            return (self.n_planes, W, W)
        raise ValueError(f"unknown observation kind {self.kind!r}")


def cell_is_danger(body_age: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """True where moving into ``cells`` ``(N, 2)`` would end the episode.

    A cell is dangerous if it is off-grid or holds body with ``age > 1``.
    The tail cell (age 1) vacates during the move (PLAN.md section 3.3, step 4)
    unless the snake eats, and food is never on the body, so age 1 is always
    safe. This matches ``step`` exactly; ``tests/test_obs.py`` checks it.
    """
    N, W, _ = body_age.shape
    r, c = cells[:, 0], cells[:, 1]
    in_bounds = (r >= 0) & (r < W) & (c >= 0) & (c < W)
    val = body_age[np.arange(N), np.clip(r, 0, W - 1), np.clip(c, 0, W - 1)]
    return ~in_bounds | (val > 1)


def encode_A(body_age: np.ndarray, heading: np.ndarray, food: np.ndarray) -> np.ndarray:
    """Observation A for a batch. ``body_age (N,W,W)``, ``heading (N,)``, ``food (N,2)``."""
    N = body_age.shape[0]
    heading = np.asarray(heading, dtype=np.int64)
    food = np.asarray(food, dtype=np.int64)
    head = head_positions(body_age)
    out = np.zeros((N, N_FEATURES_A), dtype=np.float32)
    for k, turn in enumerate((0, -1, 1)):  # straight, left, right
        h = (heading + turn) % 4
        out[:, k] = cell_is_danger(body_age, head + DELTAS[h])
    out[np.arange(N), 3 + heading] = 1.0
    out[:, A_FOOD_UP] = food[:, 0] < head[:, 0]
    out[:, A_FOOD_DOWN] = food[:, 0] > head[:, 0]
    out[:, A_FOOD_LEFT] = food[:, 1] < head[:, 1]
    out[:, A_FOOD_RIGHT] = food[:, 1] > head[:, 1]
    return out


def encode_B(
    body_age: np.ndarray,
    heading: np.ndarray,
    food: np.ndarray,
    length: np.ndarray,
    cfg: ObsConfig,
) -> np.ndarray:
    """Observation B ``(N, C, W, W)`` float32 for a batch."""
    N, W, _ = body_age.shape
    heading = np.asarray(heading, dtype=np.int64)
    food = np.asarray(food, dtype=np.int64)
    length = np.asarray(length, dtype=np.float32)
    idx = np.arange(N)
    head = head_positions(body_age)

    planes: list[np.ndarray] = []
    head_plane = np.zeros((N, 1, W, W), dtype=np.float32)
    head_plane[idx, 0, head[:, 0], head[:, 1]] = 1.0
    planes.append(head_plane)
    planes.append((body_age > 0).astype(np.float32)[:, None])
    food_plane = np.zeros((N, 1, W, W), dtype=np.float32)
    food_plane[idx, 0, food[:, 0], food[:, 1]] = 1.0
    planes.append(food_plane)
    if cfg.age_plane:
        planes.append((body_age.astype(np.float32) / length[:, None, None])[:, None])
    if cfg.heading_planes:
        one_hot = np.zeros((N, 4), dtype=np.float32)
        one_hot[idx, heading] = 1.0
        planes.append(np.broadcast_to(one_hot[:, :, None, None], (N, 4, W, W)))
    return np.concatenate(planes, axis=1)


def encode(
    cfg: ObsConfig,
    body_age: np.ndarray,
    heading: np.ndarray,
    food: np.ndarray,
    length: np.ndarray,
) -> np.ndarray:
    """Dispatch on ``cfg.kind``. All inputs batched."""
    if cfg.kind == "A":
        return encode_A(body_age, heading, food)
    if cfg.kind == "B":
        return encode_B(body_age, heading, food, length, cfg)
    raise ValueError(f"unknown observation kind {cfg.kind!r}")

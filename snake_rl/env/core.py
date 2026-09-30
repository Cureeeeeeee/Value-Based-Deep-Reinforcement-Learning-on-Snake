"""Scalar reference Snake environment (Gymnasium API).

Readability first: this is the implementation used by the tests, the GUI and
the oral defence. The vectorised training engine (``vec_env.py``) must agree
with it step by step (``tests/test_vec_env.py``). Rules that both share live
in ``common.py``; observations in ``obs.py``; rewards in ``rewards.py``.

Step order (PLAN.md section 3.3; the order *is* the semantics):

1. heading' from the action; ``new_head = head + delta``.
2. Off-grid  -> ``terminated``, cause ``wall``.
3. ``new_head == food`` -> eat: body_age is **not** decremented (tail stays),
   ``length += 1``, ``body_age[new_head] = length``, ``score += 1``,
   ``steps_since_food = 0``, food re-spawns uniformly on an empty cell. If the
   board is full (``length == W*W``) -> ``terminated``, cause ``win``.
4. Otherwise decrement every positive cell (the tail vacates), then
   ``body_age[new_head] > 0`` -> ``terminated``, cause ``self``; else place
   the head and ``steps_since_food += 1``. Moving into the vacating tail cell
   is legal.
5. Alive and ``steps_since_food >= H`` -> ``truncated``, cause ``starve``.
6. Return ``(obs, reward, terminated, truncated, info)`` with
   ``info = {cause, score, length, steps}``.

``steps_since_food`` is not observable, which is exactly why a starvation
truncation must be bootstrapped by the learner rather than treated as a
terminal state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .common import (
    AGE_DTYPE,
    CAUSE_NAMES,
    CAUSE_NONE,
    CAUSE_SELF,
    CAUSE_STARVE,
    CAUSE_WALL,
    CAUSE_WIN,
    DELTAS,
    RIGHT,
    TERMINAL_CAUSES,
    ActionMode,
    body_age_from_cells,
    cells_from_body_age,
    default_starve_limit,
    infer_heading,
    initial_cells,
    manhattan,
    n_actions,
    next_heading,
    sample_food,
)
from .obs import ObsConfig, encode
from .rewards import RewardConfig, compute_reward


@dataclass(frozen=True)
class EnvConfig:
    W: int = 10
    action_mode: ActionMode = "relative"
    #: Starvation limit H; ``None`` -> ``max(100, W*W)``.
    starve_limit: int | None = None
    obs: ObsConfig = field(default_factory=ObsConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    @property
    def H(self) -> int:
        return self.starve_limit if self.starve_limit is not None else default_starve_limit(self.W)


class SnakeEnv(gym.Env):
    """Single Snake board. See module docstring for the rules."""

    metadata = {"render_modes": ["ansi"], "render_fps": 10}

    def __init__(self, cfg: EnvConfig | None = None, render_mode: str | None = None) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else EnvConfig()
        if self.cfg.W < 4:
            raise ValueError("W must be at least 4")
        self.W = self.cfg.W
        self.H = self.cfg.H
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(n_actions(self.cfg.action_mode))
        self.observation_space = spaces.Box(0.0, 1.0, self.cfg.obs.shape(self.W), dtype=np.float32)

        self._rng: np.random.Generator | None = None
        self.body_age: np.ndarray = np.zeros((self.W, self.W), dtype=AGE_DTYPE)
        self.heading: int = RIGHT
        self.food: tuple[int, int] = (0, 0)
        self.length: int = 0
        self.score: int = 0
        self.steps_since_food: int = 0
        self.steps: int = 0
        self._done: bool = True

    # ------------------------------------------------------------------ API
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None or self._rng is None:
            self._rng = np.random.default_rng(seed)
        cells = initial_cells(self.W)
        self.body_age = body_age_from_cells(cells, self.W)
        self.heading = RIGHT
        self.length = len(cells)
        self.score = 0
        self.steps_since_food = 0
        self.steps = 0
        self.food = sample_food(self._rng, self.body_age)
        self._done = False
        return self._obs(), self._info(CAUSE_NONE)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._done:
            raise RuntimeError("step() called on a finished episode; call reset() first")
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action} for {self.action_space}")
        W = self.W

        # 1. heading and target cell
        heading = int(next_heading(self.heading, action, self.cfg.action_mode))
        head = self.head
        dr, dc = DELTAS[heading]
        new_head = (head[0] + int(dr), head[1] + int(dc))
        food_prev = self.food
        d_prev = int(manhattan(head, food_prev))
        d_moved = int(manhattan(new_head, food_prev))
        self.heading = heading
        self.steps += 1

        ate = died = False
        cause = CAUSE_NONE
        if not (0 <= new_head[0] < W and 0 <= new_head[1] < W):
            # 2. wall
            cause, died = CAUSE_WALL, True
        elif new_head == food_prev:
            # 3. eat: tail does not move
            ate = True
            self.length += 1
            self.body_age[new_head] = self.length
            self.score += 1
            self.steps_since_food = 0
            if self.length == W * W:
                cause = CAUSE_WIN
            else:
                self.food = sample_food(self._rng, self.body_age)
        else:
            # 4. move: vacate the tail first, then test the target cell
            self.body_age[self.body_age > 0] -= 1
            if self.body_age[new_head] > 0:
                cause, died = CAUSE_SELF, True
            else:
                self.body_age[new_head] = self.length
                self.steps_since_food += 1
                # 5. starvation truncation
                if self.steps_since_food >= self.H:
                    cause = CAUSE_STARVE

        terminated = cause in TERMINAL_CAUSES
        truncated = cause == CAUSE_STARVE
        self._done = terminated or truncated
        d_next = int(manhattan(new_head, self.food))
        reward = float(
            compute_reward(
                self.cfg.reward,
                W,
                ate=ate,
                died=died,
                terminal=terminated,
                d_prev=d_prev,
                d_moved=d_moved,
                d_next=d_next,
            )
        )
        return self._obs(), reward, terminated, truncated, self._info(cause)

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        return self.to_ansi()

    # ------------------------------------------------------------ inspection
    @property
    def head(self) -> tuple[int, int]:
        r, c = np.unravel_index(int(np.argmax(self.body_age)), self.body_age.shape)
        return int(r), int(c)

    @property
    def rng(self) -> np.random.Generator:
        if self._rng is None:
            raise RuntimeError("call reset() first")
        return self._rng

    def snake_cells(self) -> list[tuple[int, int]]:
        """Cells tail -> head."""
        return cells_from_body_age(self.body_age)

    def to_ansi(self) -> str:
        chars = np.full((self.W, self.W), ".", dtype="<U1")
        chars[self.body_age > 0] = "o"
        chars[self.head] = "H"
        chars[self.food] = "*"
        return "\n".join("".join(row) for row in chars)

    def set_state(
        self,
        cells: Sequence[tuple[int, int]],
        food: tuple[int, int],
        heading: int | None = None,
        steps_since_food: int = 0,
    ) -> np.ndarray:
        """Overwrite the board (tests, GUI replay). ``cells`` are tail -> head.

        ``heading`` defaults to the neck -> head direction. Score and step
        counters are reset. Returns the observation.
        """
        if self._rng is None:
            self._rng = np.random.default_rng()
        body_age = body_age_from_cells(cells, self.W)
        for (r0, c0), (r1, c1) in zip(cells[:-1], cells[1:]):
            if abs(r0 - r1) + abs(c0 - c1) != 1:
                raise ValueError(f"cells {(r0, c0)} and {(r1, c1)} are not adjacent")
        if body_age[food] != 0:
            raise ValueError("food on the snake")
        self.body_age = body_age
        self.heading = infer_heading(body_age) if heading is None else int(heading)
        self.food = (int(food[0]), int(food[1]))
        self.length = len(cells)
        self.score = 0
        self.steps = 0
        self.steps_since_food = int(steps_since_food)
        self._done = False
        return self._obs()

    # -------------------------------------------------------------- helpers
    def _obs(self) -> np.ndarray:
        return encode(
            self.cfg.obs,
            self.body_age[None],
            np.array([self.heading]),
            np.array([self.food]),
            np.array([self.length]),
        )[0]

    def _info(self, cause: int) -> dict[str, Any]:
        return {
            "cause": CAUSE_NAMES[cause],
            "score": self.score,
            "length": self.length,
            "steps": self.steps,
        }

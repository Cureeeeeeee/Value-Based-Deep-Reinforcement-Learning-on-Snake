"""Vectorised Snake environment (the training engine).

Throughput first: ``N`` boards advance together as one ``(N, W, W)`` batch of
``body_age`` grids and every rule is a masked array operation. The rules are
those of the scalar reference (``core.py``, PLAN.md section 3.3); observations
(``obs.py``) and rewards (``rewards.py``) are the shared implementations, so
they are written exactly once.

Equivalence guarantee (``tests/test_vec_env.py`` is the regression gate):
sub-env ``i`` behaves bitwise like a scalar ``SnakeEnv(cfg)`` that was
``reset(seed=seed + i)``, is fed the same actions, and is ``reset()`` without
a seed immediately after each of its episodes ends.

Step order per sub-env (the masks ``wall`` / ``ate`` / ``move`` are disjoint):

1. heading' from the action; ``new_head = head + delta``; ``steps += 1``.
2. ``wall``: off-grid -> ``terminated``. Grid and ``steps_since_food`` are
   untouched.
3. ``ate``: ``new_head == food`` -> body_age is **not** decremented,
   ``length += 1``, head placed with the new length, ``score += 1``,
   ``steps_since_food = 0``. Board full -> ``terminated`` (``win``), else a
   new food is drawn.
4. ``move``: decrement every positive cell (the tail vacates), then test the
   target cell: still occupied -> ``terminated`` (``self``) and the head is
   not placed; otherwise place the head and ``steps_since_food += 1``.
5. Moved, alive and ``steps_since_food >= H`` -> ``truncated`` (``starve``).
6. Auto-reset: finished sub-envs restart at once. ``step`` returns the first
   observation of the new episode for them and keeps the true successor of
   the transition in ``info["final_obs"]``.

``truncated`` is not ``terminated``: a starvation-truncated transition must be
bootstrapped from its real successor, which is why replay buffers store
``info["final_obs"]`` (not the post-reset observation) as ``next_obs``.

RNG (PLAN.md section 3.6): one ``np.random.Generator`` per sub-env, seeded
``seed + i``. The only consumption is one ``random(W * W)`` draw per food
placement (protocol of ``common.sample_food``): once per (auto-)reset and once
after eating unless the board is full. Two environments whose base seeds differ
by less than ``n_envs`` share generators (base ``s + 1`` is base ``s`` shifted
by one sub-env); runs that must be independent space their base seeds by at
least ``n_envs``.

State arrays (``body_age``, ``heading``, ``food``, ``length``, ``score``,
``steps_since_food``, ``steps``) are public and updated in place; callers
treat them as read-only and copy what they want to keep. Arrays *returned* by
``reset`` / ``step`` / ``observe`` are never modified afterwards by the env
and never share memory with each other or with the state arrays.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .common import (
    AGE_DTYPE,
    CAUSE_NONE,
    CAUSE_SELF,
    CAUSE_STARVE,
    CAUSE_WALL,
    CAUSE_WIN,
    DELTAS,
    RIGHT,
    body_age_from_cells,
    head_positions,
    infer_heading,
    initial_cells,
    manhattan,
    n_actions,
    next_heading,
)
from .core import EnvConfig
from .obs import encode
from .rewards import compute_reward


class VecSnakeEnv:
    """``n_envs`` independent Snake boards stepped as one batch, with auto-reset.

    See the module docstring for the rules. ``debug=True`` runs
    :meth:`check_invariants` after every ``reset`` and ``step``.
    """

    def __init__(
        self,
        cfg: EnvConfig | None = None,
        n_envs: int = 128,
        seed: int = 0,
        debug: bool = False,
    ) -> None:
        self.cfg = cfg if cfg is not None else EnvConfig()
        if self.cfg.W < 4:
            raise ValueError("W must be at least 4")
        if n_envs < 1:
            raise ValueError("n_envs must be at least 1")
        self.n_envs = int(n_envs)
        self.W = self.cfg.W
        self.H = self.cfg.H
        self.n_actions = n_actions(self.cfg.action_mode)
        self.obs_shape = self.cfg.obs.shape(self.W)
        #: Base seed of the generators in use; sub-env ``i`` owns ``seed + i``.
        self.seed = int(seed)
        self.debug = bool(debug)

        N, W = self.n_envs, self.W
        self.body_age: np.ndarray = np.zeros((N, W, W), dtype=AGE_DTYPE)
        self.heading: np.ndarray = np.full(N, RIGHT, dtype=np.int64)
        self.food: np.ndarray = np.zeros((N, 2), dtype=np.int64)
        self.length: np.ndarray = np.zeros(N, dtype=np.int64)
        self.score: np.ndarray = np.zeros(N, dtype=np.int64)
        self.steps_since_food: np.ndarray = np.zeros(N, dtype=np.int64)
        self.steps: np.ndarray = np.zeros(N, dtype=np.int64)

        self._rngs = [np.random.default_rng(self.seed + i) for i in range(N)]
        #: Cached head cell, always equal to ``head_positions(body_age)``.
        self._head: np.ndarray = np.zeros((N, 2), dtype=np.int64)
        self._rows = np.arange(N)
        cells = initial_cells(W)
        self._start_grid = body_age_from_cells(cells, W)
        self._start_head = np.array(cells[-1], dtype=np.int64)
        self._start_length = len(cells)
        self._started = False

    # ------------------------------------------------------------------ API
    def reset(self, seed: int | None = None) -> np.ndarray:
        """Restart every board and return the observation ``(N, *obs_shape)``
        only (no info dict, unlike ``SnakeEnv.reset``).

        ``seed`` re-creates the generators as ``default_rng(seed + i)``;
        ``None`` keeps them, so the random streams continue.
        """
        if seed is not None:
            seed = int(seed)
            # Build first: an invalid seed must leave ``seed`` and the streams as they were.
            self._rngs = [np.random.default_rng(seed + i) for i in range(self.n_envs)]
            self.seed = seed
        self._restart(self._rows)
        self._started = True
        if self.debug:
            self.check_invariants()
        return self.observe()

    def step(
        self, actions: np.ndarray | Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Advance every board by one action.

        Returns ``(obs, reward, terminated, truncated, info)``; all arrays have
        a leading ``N`` axis. ``obs`` is the observation to act on next (the
        first one of the new episode where the episode ended). ``info`` holds
        ``final_obs`` (true successor of this transition, before auto-reset),
        ``done`` (``terminated | truncated``), ``cause`` (``CAUSE_*`` ids) and
        ``score`` / ``length`` / ``steps`` as they stood before auto-reset.
        ``obs`` and ``info["final_obs"]`` are always two separate arrays, equal
        on the rows that did not finish.
        """
        self._require_reset()
        actions = self._check_actions(actions)
        W, rows, grid = self.W, self._rows, self.body_age

        # 1. heading and target cell
        heading = next_heading(self.heading, actions, self.cfg.action_mode)
        head = self._head
        new_head = head + DELTAS[heading]
        d_prev = manhattan(head, self.food)
        d_moved = manhattan(new_head, self.food)
        self.heading[:] = heading
        self.steps += 1

        # 2. wall / 3. eat masks. Food is always on the grid, so ``ate``
        # excludes ``wall``; everything else is an ordinary move.
        r, c = new_head[:, 0], new_head[:, 1]
        wall = (r < 0) | (r >= W) | (c < 0) | (c >= W)
        ate = (r == self.food[:, 0]) & (c == self.food[:, 1])
        move = ~(wall | ate)

        # 4. move: vacate the tail first, then test the target cell (so the
        # vacating tail cell is legal). Coordinates are clipped only for the
        # wall rows, whose lookup is masked out by ``move``.
        np.subtract(grid, 1, out=grid, where=move[:, None, None] & (grid > 0))
        occupied = grid[rows, np.clip(r, 0, W - 1), np.clip(c, 0, W - 1)] > 0
        hit = move & occupied
        ok = move & ~hit

        # Head placement (eat: with the new length; self collision: not placed).
        self.length += ate
        placed = np.flatnonzero(ate | ok)
        grid[placed, r[placed], c[placed]] = self.length[placed]
        head[placed] = new_head[placed]

        self.score += ate
        self.steps_since_food[ate] = 0
        self.steps_since_food += ok

        # 5. episode ends. The starvation check runs on plain moves only.
        win = ate & (self.length == W * W)
        starve = ok & (self.steps_since_food >= self.H)
        died = wall | hit
        terminated = died | win
        truncated = starve
        done = terminated | truncated
        cause = np.full(self.n_envs, CAUSE_NONE, dtype=np.int64)
        cause[wall] = CAUSE_WALL
        cause[hit] = CAUSE_SELF
        cause[starve] = CAUSE_STARVE
        cause[win] = CAUSE_WIN

        # New food for the eaters, unless the board is full.
        eaters = np.flatnonzero(ate & ~win)
        if eaters.size:
            self._spawn_food(eaters)
        d_next = manhattan(new_head, self.food)
        reward = compute_reward(
            self.cfg.reward,
            W,
            ate=ate,
            died=died,
            terminal=terminated,
            d_prev=d_prev,
            d_moved=d_moved,
            d_next=d_next,
        )

        # True successor of every transition, snapshotted before auto-reset.
        final_obs = self.observe()
        info = {
            "final_obs": final_obs,
            "done": done,
            "cause": cause,
            "score": self.score.copy(),
            "length": self.length.copy(),
            "steps": self.steps.copy(),
        }

        # 6. auto-reset: finished boards start their next episode now. ``obs``
        # is always its own array, so a caller editing it in place can never
        # corrupt the ``final_obs`` that goes into the replay buffer.
        obs = final_obs.copy()
        finished = np.flatnonzero(done)
        if finished.size:
            self._restart(finished)
            obs[finished] = encode(
                self.cfg.obs,
                grid[finished],
                self.heading[finished],
                self.food[finished],
                self.length[finished],
            )
        if self.debug:
            self.check_invariants()
        return obs, reward, terminated, truncated, info

    def observe(self) -> np.ndarray:
        """Fresh observation ``(N, *obs_shape)`` float32 of the current state."""
        self._require_reset()
        return encode(self.cfg.obs, self.body_age, self.heading, self.food, self.length)

    # ------------------------------------------------------------ inspection
    @property
    def head(self) -> np.ndarray:
        """``(N, 2)`` head cells ``(r, c)`` (a copy)."""
        return self._head.copy()

    def set_state(
        self,
        i: int,
        cells: Sequence[tuple[int, int]],
        food: tuple[int, int],
        heading: int | None = None,
        steps_since_food: int = 0,
    ) -> None:
        """Overwrite board ``i`` (tests, GUI replay). ``cells`` are tail -> head.

        Like ``SnakeEnv.set_state`` (``heading`` defaults to the neck -> head
        direction; score and step counters of board ``i`` are reset) but
        returns ``None``, requires ``reset()`` first and rejects off-grid
        cells. The other boards are untouched; a rejected call changes nothing.
        """
        self._require_reset()
        i = int(i)
        if not 0 <= i < self.n_envs:
            raise IndexError(f"sub-env index {i} out of range for n_envs = {self.n_envs}")
        W = self.W
        cells = [(int(r), int(c)) for r, c in cells]
        if not cells:
            raise ValueError("cells must not be empty")
        for r, c in cells:
            if not (0 <= r < W and 0 <= c < W):
                raise ValueError(f"cell {(r, c)} is off the grid")
        body_age = body_age_from_cells(cells, W)
        for (r0, c0), (r1, c1) in zip(cells[:-1], cells[1:]):
            if abs(r0 - r1) + abs(c0 - c1) != 1:
                raise ValueError(f"cells {(r0, c0)} and {(r1, c1)} are not adjacent")
        food = (int(food[0]), int(food[1]))
        if not (0 <= food[0] < W and 0 <= food[1] < W):
            raise ValueError(f"food {food} is off the grid")
        if body_age[food] != 0:
            raise ValueError("food on the snake")
        heading = infer_heading(body_age) if heading is None else int(heading)
        if not 0 <= heading <= 3:
            raise ValueError(f"invalid heading {heading}")
        steps_since_food = int(steps_since_food)  # may raise: convert before any write

        self.body_age[i] = body_age
        self.heading[i] = heading
        self.food[i] = food
        self.length[i] = len(cells)
        self.score[i] = 0
        self.steps[i] = 0
        self.steps_since_food[i] = steps_since_food
        self._head[i] = cells[-1]

    def check_invariants(self) -> None:
        """Raise ``AssertionError`` naming a broken sub-env (the lowest index
        that fails the first violated check).

        Checked for every board: the positive cells of ``body_age`` are exactly
        the ages ``1..length``, each once; consecutive ages are 4-adjacent;
        the food is on an empty in-bounds cell; ``heading`` is in ``0..3``;
        ``0 <= steps_since_food < H``; counters are non-negative; the cached
        head agrees with ``body_age``.
        """
        self._require_reset()
        N, W, rows = self.n_envs, self.W, self._rows
        n_cells = W * W
        flat = self.body_age.reshape(N, n_cells)
        length = self.length

        self._assert_all((length >= 1) & (length <= n_cells), "length outside 1..W*W")
        # Sorted in descending order a row must read length, ..., 2, 1, 0, ..., 0.
        expected = np.maximum(length[:, None] - np.arange(n_cells), 0)
        self._assert_all(
            (np.sort(flat, axis=1)[:, ::-1] == expected).all(axis=1),
            "body_age is not the ages 1..length, each exactly once",
        )
        # cell_of[n, k] = flat index of the cell with age k (column 0 is unused).
        cell_of = np.zeros((N, n_cells + 1), dtype=np.int64)
        cell_of[rows[:, None], flat] = np.arange(n_cells)
        r, c = np.divmod(cell_of[:, 1:], W)
        gap = np.abs(np.diff(r, axis=1)) + np.abs(np.diff(c, axis=1))  # ages k, k + 1
        linked = np.arange(2, n_cells + 1) <= length[:, None]  # age k + 1 exists
        self._assert_all(
            ~(linked & (gap != 1)).any(axis=1),
            "cells with consecutive ages are not adjacent",
        )

        food = self.food
        self._assert_all(((food >= 0) & (food < W)).all(axis=1), "food is off the grid")
        self._assert_all(self.body_age[rows, food[:, 0], food[:, 1]] == 0, "food is on the snake")
        self._assert_all((self.heading >= 0) & (self.heading <= 3), "heading outside 0..3")
        self._assert_all(
            (self.steps_since_food >= 0) & (self.steps_since_food < self.H),
            "steps_since_food outside 0..H-1",
        )
        self._assert_all((self.score >= 0) & (self.steps >= 0), "negative score or step counter")
        self._assert_all(
            (self._head == head_positions(self.body_age)).all(axis=1),
            "cached head disagrees with body_age",
        )

    # -------------------------------------------------------------- helpers
    def _require_reset(self) -> None:
        if not self._started:
            raise RuntimeError("call reset() first")

    def _check_actions(self, actions: np.ndarray | Sequence[int]) -> np.ndarray:
        actions = np.asarray(actions)
        if actions.shape != (self.n_envs,):
            raise ValueError(f"actions must have shape ({self.n_envs},), got {actions.shape}")
        if not np.issubdtype(actions.dtype, np.integer):
            raise ValueError(f"actions must be integers, got dtype {actions.dtype}")
        if actions.min() < 0 or actions.max() >= self.n_actions:
            raise ValueError(f"actions must lie in [0, {self.n_actions})")
        return actions.astype(np.int64, copy=False)

    def _restart(self, rows: np.ndarray) -> None:
        """Put boards ``rows`` into the initial state and draw their food."""
        self.body_age[rows] = self._start_grid
        self.heading[rows] = RIGHT
        self._head[rows] = self._start_head
        self.length[rows] = self._start_length
        self.score[rows] = 0
        self.steps_since_food[rows] = 0
        self.steps[rows] = 0
        self._spawn_food(rows)

    def _spawn_food(self, rows: np.ndarray) -> None:
        """Uniformly random empty cell for each board in ``rows``.

        Batched ``common.sample_food``: one key per cell from the board's own
        generator, occupied cells -> +inf, argmin. The caller guarantees an
        empty cell on every board in ``rows``.
        """
        keys = np.empty((rows.size, self.W * self.W))
        rngs = self._rngs
        for k, i in enumerate(rows.tolist()):  # one Generator per sub-env
            rngs[i].random(out=keys[k])
        keys[self.body_age[rows].reshape(rows.size, -1) > 0] = np.inf
        self.food[rows] = np.stack(np.divmod(keys.argmin(axis=1), self.W), axis=1)

    def _assert_all(self, ok: np.ndarray, what: str) -> None:
        if not ok.all():
            i = int(np.flatnonzero(~ok)[0])
            raise AssertionError(f"sub-env {i}: {what}")

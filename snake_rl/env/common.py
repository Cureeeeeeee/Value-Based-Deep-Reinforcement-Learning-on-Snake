"""Rules and primitives shared by the scalar environment (``core.py``) and the
vectorised environment (``vec_env.py``).

Everything that defines *what the game is* lives here or in ``obs.py`` /
``rewards.py`` so that it is written exactly once. Batched helpers take arrays
with a leading ``N`` axis; the scalar environment calls them with ``N = 1``.

Coordinate system: grid cell ``(r, c)``; ``r`` grows downwards, ``c`` grows to
the right. Headings are clockwise: 0 = up, 1 = right, 2 = down, 3 = left.

Internal state (PLAN.md section 3.1): ``body_age[W, W]`` with 0 = empty and
``k > 0`` = this cell stays occupied for ``k`` more steps. Tail = 1,
neck = length - 1, head = length. One array gives body, head, tail, heading
(neck = head neighbour with the largest value) and the age plane of
observation B.
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

# --- headings -----------------------------------------------------------------
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
HEADING_NAMES = ("up", "right", "down", "left")
#: (dr, dc) for each heading, indexed by heading id.
DELTAS = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)], dtype=np.int64)

# --- actions ------------------------------------------------------------------
STRAIGHT, TURN_LEFT, TURN_RIGHT = 0, 1, 2
#: Heading change for each relative action: straight 0, left -1, right +1.
RELATIVE_TURN = np.array([0, -1, 1], dtype=np.int64)
ActionMode = Literal["relative", "absolute"]

# --- episode end causes ---------------------------------------------------------
CAUSE_NONE, CAUSE_WALL, CAUSE_SELF, CAUSE_STARVE, CAUSE_WIN = range(5)
CAUSE_NAMES = ("none", "wall", "self", "starve", "win")
#: Causes that set ``terminated`` (next-state value is zero). ``starve`` sets
#: ``truncated`` instead and must be bootstrapped.
TERMINAL_CAUSES = (CAUSE_WALL, CAUSE_SELF, CAUSE_WIN)

#: dtype of ``body_age`` grids.
AGE_DTYPE = np.int32


def n_actions(action_mode: ActionMode) -> int:
    if action_mode == "relative":
        return 3
    if action_mode == "absolute":
        return 4
    raise ValueError(f"unknown action_mode {action_mode!r}")


def default_starve_limit(W: int) -> int:
    """Starvation limit H = max(100, W^2).

    On 10x10 a Hamiltonian player needs at most 99 steps between foods, so
    100 never kills it; on 16x16 the cycle has 256 cells, hence W^2.
    """
    return max(100, W * W)


def next_heading(heading, action, action_mode: ActionMode):
    """New heading after ``action``. Works element-wise on arrays.

    Relative mode: 0 straight, 1 turn left, 2 turn right.
    Absolute mode: action is the desired heading; the 180-degree reversal is
    treated as "straight" instead of an instant death (PLAN.md section 3.2).
    """
    heading = np.asarray(heading, dtype=np.int64)
    action = np.asarray(action, dtype=np.int64)
    if action_mode == "relative":
        return (heading + RELATIVE_TURN[action]) % 4
    if action_mode == "absolute":
        reverse = (heading + 2) % 4
        return np.where(action == reverse, heading, action)
    raise ValueError(f"unknown action_mode {action_mode!r}")


def heading_to_action(current: int, desired: int, action_mode: ActionMode) -> int:
    """Action that turns ``current`` heading into ``desired`` (used by planners).

    Relative mode cannot express the 180-degree reversal; asking for it is a
    caller bug and raises.
    """
    if action_mode == "absolute":
        return int(desired)
    if action_mode != "relative":
        raise ValueError(f"unknown action_mode {action_mode!r}")
    diff = (int(desired) - int(current)) % 4
    if diff == 0:
        return STRAIGHT
    if diff == 3:
        return TURN_LEFT
    if diff == 1:
        return TURN_RIGHT
    raise ValueError(f"heading {desired} is the reversal of {current}; not expressible as a relative action")


def initial_cells(W: int) -> list[tuple[int, int]]:
    """Initial snake, tail -> head: length 3, horizontally centred, heading right."""
    r, c = W // 2, W // 2
    return [(r, c - 2), (r, c - 1), (r, c)]


def body_age_from_cells(cells: Sequence[tuple[int, int]], W: int) -> np.ndarray:
    """Build a ``body_age`` grid from cells listed tail -> head."""
    body_age = np.zeros((W, W), dtype=AGE_DTYPE)
    for age, (r, c) in enumerate(cells, start=1):
        if body_age[r, c] != 0:
            raise ValueError(f"cell {(r, c)} listed twice")
        body_age[r, c] = age
    return body_age


def cells_from_body_age(body_age: np.ndarray) -> list[tuple[int, int]]:
    """Snake cells ordered tail -> head (ascending age)."""
    rows, cols = np.nonzero(body_age)
    order = np.argsort(body_age[rows, cols], kind="stable")
    return [(int(r), int(c)) for r, c in zip(rows[order], cols[order])]


def infer_heading(body_age: np.ndarray) -> int:
    """Heading implied by neck -> head. Requires length >= 2."""
    cells = cells_from_body_age(body_age)
    if len(cells) < 2:
        raise ValueError("need at least two cells to infer a heading")
    (nr, nc), (hr, hc) = cells[-2], cells[-1]
    delta = (hr - nr, hc - nc)
    for h, (dr, dc) in enumerate(DELTAS.tolist()):
        if (dr, dc) == delta:
            return h
    raise ValueError(f"neck {cells[-2]} and head {cells[-1]} are not adjacent")


def head_positions(body_age: np.ndarray) -> np.ndarray:
    """``(N, 2)`` head coordinates of a batch ``(N, W, W)`` of body_age grids."""
    N, W, _ = body_age.shape
    flat = body_age.reshape(N, -1).argmax(axis=1)
    return np.stack(np.divmod(flat, W), axis=1)


def manhattan(a, b) -> np.ndarray:
    """Manhattan distance between coordinate arrays whose last axis is (r, c)."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    return np.abs(a - b).sum(axis=-1)


def sample_food(rng: np.random.Generator, body_age: np.ndarray) -> tuple[int, int]:
    """Uniformly random empty cell.

    Protocol (PLAN.md section 3.6): draw one random key per cell, set occupied
    cells to +inf, take the argmin. The vectorised environment uses the same
    draw with the same shape so that the scalar/vectorised equivalence test
    holds for the same seed. The caller must guarantee at least one empty cell.
    """
    keys = rng.random(body_age.shape)
    keys[body_age > 0] = np.inf
    flat = int(np.argmin(keys))
    r, c = divmod(flat, body_age.shape[1])
    return int(r), int(c)

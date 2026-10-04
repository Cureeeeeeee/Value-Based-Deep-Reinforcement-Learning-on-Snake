"""Scalar environment: rules of PLAN.md section 3.1-3.3 and 3.6."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

from snake_rl.env.common import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    body_age_from_cells,
    cells_from_body_age,
    default_starve_limit,
    infer_heading,
    next_heading,
    sample_food,
)
from tests.helpers import (
    BLOCK_TAIL_CHASE,
    BLOCK_TAIL_CHASE_ACTION,
    FAR_FOOD,
    HOOK_COLLIDING_ACTION,
    HOOK_SELF_COLLISION,
    STRAIGHT,
    STRAIGHT_RIGHT,
    TURN_LEFT,
    TURN_RIGHT,
    EnvConfig,
    ObsConfig,
    SnakeEnv,
    make_env,
    random_episode,
    serpentine_almost_full,
)


# ------------------------------------------------------------- basic shape
def test_initial_state():
    env = make_env(10)
    obs, info = env.reset(seed=0)
    assert env.length == 3 and env.score == 0 and env.steps == 0
    assert env.heading == RIGHT
    assert env.head == (5, 5)
    assert env.snake_cells() == [(5, 3), (5, 4), (5, 5)]
    assert env.body_age[5, 3] == 1 and env.body_age[5, 4] == 2 and env.body_age[5, 5] == 3
    assert env.body_age[env.food] == 0
    assert info == {"cause": "none", "score": 0, "length": 3, "steps": 0}
    assert obs.shape == (11,) and obs.dtype == np.float32


@pytest.mark.parametrize(
    "kwargs, obs_shape, n_actions",
    [
        ({}, (11,), 3),
        ({"action_mode": "absolute"}, (11,), 4),
        ({"obs": ObsConfig(kind="B")}, (8, 10, 10), 3),
        ({"obs": ObsConfig(kind="B", age_plane=False, heading_planes=False)}, (3, 10, 10), 3),
    ],
)
def test_spaces(kwargs, obs_shape, n_actions):
    env = SnakeEnv(EnvConfig(W=10, **kwargs))
    assert env.action_space == spaces.Discrete(n_actions)
    assert env.observation_space.shape == obs_shape
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)


@pytest.mark.parametrize("kind", ["A", "B"])
def test_gymnasium_check_env(kind):
    check_env(SnakeEnv(EnvConfig(W=8, obs=ObsConfig(kind=kind))), skip_render_check=True)


def test_W_too_small_rejected():
    with pytest.raises(ValueError):
        SnakeEnv(EnvConfig(W=3))


# ------------------------------------------------------------- collisions
def test_wall_collision():
    env = make_env(10)
    env.set_state(STRAIGHT_RIGHT, food=FAR_FOOD)  # head (5, 5) heading right
    for _ in range(4):  # cols 6..9 are inside the board
        _, r, term, trunc, info = env.step(STRAIGHT)
        assert not term and not trunc and r == 0.0 and info["cause"] == "none"
    _, r, term, trunc, info = env.step(STRAIGHT)
    assert term and not trunc
    assert info["cause"] == "wall" and r == -1.0
    assert info["steps"] == 5
    with pytest.raises(RuntimeError):
        env.step(STRAIGHT)


def test_self_collision_into_age2_cell():
    env = make_env(10)
    env.set_state(HOOK_SELF_COLLISION, food=FAR_FOOD)
    assert env.heading == LEFT
    _, r, term, trunc, info = env.step(HOOK_COLLIDING_ACTION)
    assert term and not trunc and info["cause"] == "self" and r == -1.0


def test_tail_chase_is_legal():
    """Moving into the cell the tail vacates this step is allowed."""
    env = make_env(10)
    env.set_state(BLOCK_TAIL_CHASE, food=FAR_FOOD)
    assert env.heading == LEFT
    _, r, term, trunc, info = env.step(BLOCK_TAIL_CHASE_ACTION)
    assert not term and not trunc and info["cause"] == "none" and r == 0.0
    assert env.head == (4, 4)
    assert env.snake_cells() == [(4, 5), (5, 5), (5, 4), (4, 4)]


# ------------------------------------------------------------- eating
def test_growth_on_eating():
    env = make_env(10)
    env.set_state(STRAIGHT_RIGHT, food=(5, 6))
    _, r, term, trunc, info = env.step(STRAIGHT)
    assert not term and not trunc and r == 1.0
    assert env.length == 4 and env.score == 1 and info["score"] == 1 and info["length"] == 4
    assert env.snake_cells() == [(5, 3), (5, 4), (5, 5), (5, 6)]  # tail did not move
    assert env.steps_since_food == 0
    assert env.food != (5, 6) and env.body_age[env.food] == 0


def test_eating_resets_starvation_counter():
    H = 20
    env = make_env(10, starve_limit=H)
    env.set_state(STRAIGHT_RIGHT, food=(5, 6), steps_since_food=H - 1)
    _, _, term, trunc, _ = env.step(STRAIGHT)
    assert not term and not trunc and env.steps_since_food == 0


def test_food_never_on_snake_random_play():
    for seed in range(30):
        env = make_env(8)
        env.reset(seed=seed)
        assert env.body_age[env.food] == 0
        for _ in random_episode(env, seed):
            assert env.body_age[env.food] == 0


def test_win_on_full_board():
    W = 4
    env = make_env(W)
    cells, food = serpentine_almost_full(W)
    env.set_state(cells, food=food)
    _, r, term, trunc, info = env.step(STRAIGHT)
    assert term and not trunc
    assert info["cause"] == "win" and r == 1.0
    assert env.length == W * W and info["score"] == 1
    assert (env.body_age > 0).all()


# ------------------------------------------------------------- starvation
def test_starvation_truncates_at_exactly_H():
    H = 25
    env = make_env(8, starve_limit=H)
    env.set_state([(4, 2), (4, 3), (4, 4)], food=FAR_FOOD)
    # Always turning left cycles through a 2x2 block and never eats.
    for t in range(1, H + 1):
        _, r, term, trunc, info = env.step(TURN_LEFT)
        assert not term
        if t < H:
            assert not trunc and info["cause"] == "none"
        else:
            assert trunc and info["cause"] == "starve" and r == 0.0
    assert env.steps_since_food == H


@pytest.mark.parametrize("W, H", [(8, 100), (10, 100), (16, 256)])
def test_default_starve_limit(W, H):
    assert default_starve_limit(W) == H
    assert SnakeEnv(EnvConfig(W=W)).H == H
    assert SnakeEnv(EnvConfig(W=W, starve_limit=7)).H == 7


# ------------------------------------------------------------- actions
def test_relative_turns():
    env = make_env(10)
    env.set_state(STRAIGHT_RIGHT, food=FAR_FOOD)
    env.step(TURN_LEFT)
    assert env.heading == UP and env.head == (4, 5)
    env.step(TURN_RIGHT)
    assert env.heading == RIGHT and env.head == (4, 6)
    env.step(TURN_RIGHT)
    assert env.heading == DOWN and env.head == (5, 6)
    env.step(STRAIGHT)
    assert env.heading == DOWN and env.head == (6, 6)


def test_absolute_mode_reverse_is_straight():
    env = make_env(10, action_mode="absolute")
    env.set_state(STRAIGHT_RIGHT, food=FAR_FOOD)  # heading right
    _, _, term, _, _ = env.step(LEFT)  # 180-degree reversal
    assert not term and env.heading == RIGHT and env.head == (5, 6)
    env.step(UP)
    assert env.heading == UP and env.head == (4, 6)
    env.step(DOWN)  # reversal again
    assert env.heading == UP and env.head == (3, 6)


def test_next_heading_vectorised():
    heading = np.array([UP, RIGHT, DOWN, LEFT])
    assert next_heading(heading, np.array([0, 0, 0, 0]), "relative").tolist() == [UP, RIGHT, DOWN, LEFT]
    assert next_heading(heading, np.array([1, 1, 1, 1]), "relative").tolist() == [LEFT, UP, RIGHT, DOWN]
    assert next_heading(heading, np.array([2, 2, 2, 2]), "relative").tolist() == [RIGHT, DOWN, LEFT, UP]
    reverse = np.array([DOWN, LEFT, UP, RIGHT])
    assert next_heading(heading, reverse, "absolute").tolist() == heading.tolist()
    # heading RIGHT + action LEFT is the reversal -> stays RIGHT
    assert next_heading(heading, np.array([LEFT, LEFT, LEFT, LEFT]), "absolute").tolist() == [
        LEFT, RIGHT, LEFT, LEFT]


def test_invalid_action_rejected():
    env = make_env(10)
    with pytest.raises(ValueError):
        env.step(3)


# ------------------------------------------------------------- determinism
def test_seed_reproducibility():
    a, b = make_env(10), make_env(10)
    oa, _ = a.reset(seed=42)
    ob, _ = b.reset(seed=42)
    assert np.array_equal(oa, ob) and a.food == b.food
    rng = np.random.default_rng(1)
    done = False
    while not done:
        act = int(rng.integers(3))
        ra, rb = a.step(act), b.step(act)
        assert np.array_equal(ra[0], rb[0]) and ra[1:] == rb[1:]
        assert a.food == b.food
        done = ra[2] or ra[3]


def test_different_seeds_differ():
    foods = set()
    for s in range(10):
        env = SnakeEnv(EnvConfig(W=10))
        env.reset(seed=s)
        foods.add(env.food)
    assert len(foods) > 1


def test_reset_without_seed_keeps_rng_stream():
    env = SnakeEnv(EnvConfig(W=10))
    env.reset(seed=7)
    first = [env.reset()[0].tobytes() for _ in range(5)]
    env.reset(seed=7)
    second = [env.reset()[0].tobytes() for _ in range(5)]
    assert first == second


# ------------------------------------------------------------- food sampling
def test_sample_food_uniform_over_empty_cells():
    W = 10
    rng = np.random.default_rng(0)
    body_age = np.zeros((W, W), dtype=np.int32)
    body_age[:5] = 1  # top half occupied -> 50 empty cells
    n = 50_000
    counts = np.zeros((W, W), dtype=np.int64)
    for _ in range(n):
        counts[sample_food(rng, body_age)] += 1
    assert counts[body_age > 0].sum() == 0
    expected = n / 50
    empty_counts = counts[body_age == 0]
    assert (empty_counts > 0.75 * expected).all() and (empty_counts < 1.25 * expected).all()


# ------------------------------------------------------------- helpers
def test_body_age_helpers_roundtrip():
    W = 8
    cells = HOOK_SELF_COLLISION
    body_age = body_age_from_cells(cells, W)
    assert cells_from_body_age(body_age) == cells
    assert body_age.max() == len(cells) and (body_age > 0).sum() == len(cells)
    assert infer_heading(body_age) == LEFT
    assert infer_heading(body_age_from_cells(STRAIGHT_RIGHT, W)) == RIGHT
    with pytest.raises(ValueError):
        body_age_from_cells([(0, 0), (0, 0)], W)


def test_set_state_validation():
    env = make_env(10)
    with pytest.raises(ValueError):
        env.set_state([(0, 0), (0, 2)], food=(5, 5))  # not adjacent
    # A gap that infer_heading cannot see: only the pairwise-adjacency check rejects these.
    with pytest.raises(ValueError):
        env.set_state([(0, 0), (0, 2), (0, 3)], food=(5, 5))
    with pytest.raises(ValueError):
        env.set_state([(0, 0), (0, 2)], food=(5, 5), heading=RIGHT)
    with pytest.raises(ValueError):
        env.set_state(STRAIGHT_RIGHT, food=(5, 4))  # food on snake


def test_render_ansi():
    env = SnakeEnv(EnvConfig(W=8), render_mode="ansi")
    env.reset(seed=0)
    text = env.render()
    rows = text.split("\n")
    assert len(rows) == 8 and all(len(r) == 8 for r in rows)
    assert text.count("H") == 1 and text.count("*") == 1 and text.count("o") == 2


def test_steps_counter_in_info():
    env = make_env(8)
    env.reset(seed=0)
    n = 0
    for _, _, _, _, _, info in random_episode(env, 3):
        n += 1
        assert info["steps"] == n

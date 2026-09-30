"""Observation encoders A and B (PLAN.md section 3.4) and the Markov checks."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from snake_rl.env.common import DOWN, LEFT, RIGHT, UP, body_age_from_cells
from snake_rl.env.obs import (
    A_DANGER_LEFT,
    A_DANGER_RIGHT,
    A_DANGER_STRAIGHT,
    A_FOOD_DOWN,
    A_FOOD_LEFT,
    A_FOOD_RIGHT,
    A_FOOD_UP,
    A_HEADING,
    encode_A,
    encode_B,
)
from tests.helpers import (
    BLOCK_OTHER_HEADING,
    BLOCK_TAIL_CHASE,
    COIL_TAIL_A,
    COIL_TAIL_B,
    FAR_FOOD,
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
)

B_FULL = ObsConfig(kind="B")
B_NO_AGE = ObsConfig(kind="B", age_plane=False, heading_planes=True)
B_MINIMAL = ObsConfig(kind="B", age_plane=False, heading_planes=False)


# ------------------------------------------------------------- observation A
def test_A_is_binary_float32():
    env = make_env(10)
    env.reset(seed=0)
    for _, obs, *_ in random_episode(env, 0):
        assert obs.dtype == np.float32 and obs.shape == (11,)
        assert set(np.unique(obs)).issubset({0.0, 1.0})


def _assert_danger_bits_match_step(env: SnakeEnv, obs: np.ndarray) -> None:
    """Danger bit k must equal 'relative action k ends the episode by wall/self'."""
    for action, bit in ((STRAIGHT, A_DANGER_STRAIGHT), (TURN_LEFT, A_DANGER_LEFT), (TURN_RIGHT, A_DANGER_RIGHT)):
        probe = copy.deepcopy(env)
        _, _, term, _, info = probe.step(action)
        expected = term and info["cause"] in ("wall", "self")
        assert bool(obs[bit]) == expected, (action, info)


def test_A_danger_bits_match_step_rule_random_play():
    for seed in range(15):
        env = make_env(8)
        obs, _ = env.reset(seed=seed)
        _assert_danger_bits_match_step(env, obs)
        for _, obs, _, term, trunc, _ in random_episode(env, seed):
            if term or trunc:
                break
            _assert_danger_bits_match_step(env, obs)


def test_A_danger_bits_hand_built_positions():
    env = make_env(10)
    # Hook: turning right hits an age-2 cell, straight and left are free.
    obs = env.set_state(HOOK_SELF_COLLISION, food=FAR_FOOD)
    assert obs[[A_DANGER_STRAIGHT, A_DANGER_LEFT, A_DANGER_RIGHT]].tolist() == [0.0, 0.0, 1.0]
    _assert_danger_bits_match_step(env, obs)
    # 2x2 block: turning right enters the vacating tail -> not dangerous.
    obs = env.set_state(BLOCK_TAIL_CHASE, food=FAR_FOOD)
    assert obs[[A_DANGER_STRAIGHT, A_DANGER_LEFT, A_DANGER_RIGHT]].tolist() == [0.0, 0.0, 0.0]
    _assert_danger_bits_match_step(env, obs)
    # Wall ahead: head at (5, 9) heading right.
    obs = env.set_state([(5, 7), (5, 8), (5, 9)], food=FAR_FOOD)
    assert obs[[A_DANGER_STRAIGHT, A_DANGER_LEFT, A_DANGER_RIGHT]].tolist() == [1.0, 0.0, 0.0]
    _assert_danger_bits_match_step(env, obs)
    # Corner: head at (0, 9) heading up -> straight and right are walls.
    obs = env.set_state([(2, 9), (1, 9), (0, 9)], food=FAR_FOOD)
    assert obs[[A_DANGER_STRAIGHT, A_DANGER_LEFT, A_DANGER_RIGHT]].tolist() == [1.0, 0.0, 1.0]
    _assert_danger_bits_match_step(env, obs)


@pytest.mark.parametrize(
    "food, bits",
    [
        ((2, 5), (1, 0, 0, 0)),  # up
        ((8, 5), (0, 1, 0, 0)),  # down
        ((5, 1), (0, 0, 1, 0)),  # left
        ((5, 9), (0, 0, 0, 1)),  # right
        ((7, 8), (0, 1, 0, 1)),  # down-right
        ((0, 0), (1, 0, 1, 0)),  # up-left
    ],
)
def test_A_food_direction_bits(food, bits):
    env = make_env(10)
    obs = env.set_state(STRAIGHT_RIGHT, food=food)  # head (5, 5)
    assert obs[[A_FOOD_UP, A_FOOD_DOWN, A_FOOD_LEFT, A_FOOD_RIGHT]].tolist() == list(map(float, bits))
    assert obs[[A_FOOD_UP, A_FOOD_DOWN, A_FOOD_LEFT, A_FOOD_RIGHT]].sum() <= 2


@pytest.mark.parametrize(
    "cells, heading",
    [
        ([(5, 3), (5, 4), (5, 5)], RIGHT),
        ([(5, 7), (5, 6), (5, 5)], LEFT),
        ([(7, 5), (6, 5), (5, 5)], UP),
        ([(3, 5), (4, 5), (5, 5)], DOWN),
    ],
)
def test_A_heading_one_hot(cells, heading):
    env = make_env(10)
    obs = env.set_state(cells, food=FAR_FOOD)
    expected = np.zeros(4, dtype=np.float32)
    expected[heading] = 1.0
    assert obs[A_HEADING].tolist() == expected.tolist()


def test_encode_A_batched_equals_per_env():
    envs = [make_env(8) for _ in range(6)]
    for s, e in enumerate(envs):
        e.reset(seed=s)
    for e, s in zip(envs, range(6)):
        for _ in range(s):
            e.step(TURN_LEFT)
    single = np.stack([e._obs() for e in envs])
    batched = encode_A(
        np.stack([e.body_age for e in envs]),
        np.array([e.heading for e in envs]),
        np.array([e.food for e in envs]),
    )
    assert np.array_equal(single, batched)


# ------------------------------------------------------------- observation B
@pytest.mark.parametrize(
    "cfg, C",
    [(B_FULL, 8), (B_NO_AGE, 7), (ObsConfig(kind="B", age_plane=True, heading_planes=False), 4), (B_MINIMAL, 3)],
)
def test_B_shapes(cfg, C):
    env = SnakeEnv(EnvConfig(W=10, obs=cfg))
    obs, _ = env.reset(seed=0)
    assert obs.shape == (C, 10, 10) and obs.dtype == np.float32
    assert cfg.n_planes == C and env.observation_space.shape == (C, 10, 10)


def test_B_plane_contents():
    env = SnakeEnv(EnvConfig(W=10, obs=B_FULL))
    env.reset(seed=0)
    obs = env.set_state(HOOK_SELF_COLLISION, food=(0, 0))
    head, body, food, age = obs[0], obs[1], obs[2], obs[3]
    headings = obs[4:8]
    assert head.sum() == 1 and head[5, 4] == 1.0
    assert np.array_equal(body, (env.body_age > 0).astype(np.float32))
    assert food.sum() == 1 and food[0, 0] == 1.0
    assert np.allclose(age, env.body_age / 5.0)
    assert age[5, 4] == 1.0 and age[3, 4] == pytest.approx(0.2)
    assert age[env.body_age == 0].sum() == 0
    for h in range(4):
        expected = 1.0 if h == LEFT else 0.0
        assert (headings[h] == expected).all()


def test_encode_B_batched_equals_per_env():
    envs = [SnakeEnv(EnvConfig(W=8, obs=B_FULL)) for _ in range(5)]
    for s, e in enumerate(envs):
        e.reset(seed=s)
        for _ in range(s):
            e.step(TURN_RIGHT)
    single = np.stack([e._obs() for e in envs])
    batched = encode_B(
        np.stack([e.body_age for e in envs]),
        np.array([e.heading for e in envs]),
        np.array([e.food for e in envs]),
        np.array([e.length for e in envs]),
        B_FULL,
    )
    assert np.array_equal(single, batched)


def test_B_values_in_unit_interval_random_play():
    env = SnakeEnv(EnvConfig(W=8, obs=B_FULL))
    env.reset(seed=1)
    for _, obs, *_ in random_episode(env, 1):
        assert obs.min() >= 0.0 and obs.max() <= 1.0


# ------------------------------------------------------------- Markov checks
def _obs_for(cfg: ObsConfig, cells, food=FAR_FOOD, W: int = 10) -> np.ndarray:
    env = SnakeEnv(EnvConfig(W=W, obs=cfg))
    env.reset(seed=0)
    return env.set_state(cells, food=food)


def test_same_grid_different_heading_gives_different_obs():
    """Part B test: same cell set and food, different heading -> different obs
    for A and for B with heading information; identical for the 3-plane B."""
    for cfg in (ObsConfig(kind="A"), B_FULL, B_NO_AGE):
        a = _obs_for(cfg, BLOCK_TAIL_CHASE)  # heading LEFT
        b = _obs_for(cfg, BLOCK_OTHER_HEADING)  # heading DOWN
        assert not np.array_equal(a, b), cfg
    a = _obs_for(B_MINIMAL, BLOCK_TAIL_CHASE)
    b = _obs_for(B_MINIMAL, BLOCK_OTHER_HEADING)
    assert np.array_equal(a, b)


def test_age_plane_disambiguates_tail_but_heading_planes_do_not():
    """Two coiled snakes with the same cells, head and heading but different
    tails. Heading planes cannot tell them apart; the age plane can; and the
    successor states differ, so without the age plane the state is not Markov."""
    a_no_age = _obs_for(B_NO_AGE, COIL_TAIL_A)
    b_no_age = _obs_for(B_NO_AGE, COIL_TAIL_B)
    assert np.array_equal(a_no_age, b_no_age)
    assert np.array_equal(_obs_for(ObsConfig(kind="A"), COIL_TAIL_A), _obs_for(ObsConfig(kind="A"), COIL_TAIL_B))

    a_full = _obs_for(B_FULL, COIL_TAIL_A)
    b_full = _obs_for(B_FULL, COIL_TAIL_B)
    assert not np.array_equal(a_full, b_full)

    # Same action, different successor grids (different tail cell vacates).
    envs = []
    for cells in (COIL_TAIL_A, COIL_TAIL_B):
        env = SnakeEnv(EnvConfig(W=10, obs=B_MINIMAL))
        env.reset(seed=0)
        env.set_state(cells, food=FAR_FOOD)
        env.step(STRAIGHT)
        envs.append(env)
    assert envs[0].head == envs[1].head == (5, 2)
    assert not np.array_equal(envs[0].body_age > 0, envs[1].body_age > 0)
    assert (3, 4) not in envs[0].snake_cells() and (4, 5) not in envs[1].snake_cells()


def test_body_age_from_cells_matches_encoder_input():
    body_age = body_age_from_cells(COIL_TAIL_A, 10)
    assert body_age[5, 3] == 6 and body_age[3, 4] == 1

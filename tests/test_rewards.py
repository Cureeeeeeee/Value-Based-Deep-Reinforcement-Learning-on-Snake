"""Reward variants R0-R3 (PLAN.md section 3.5)."""

from __future__ import annotations

import numpy as np
import pytest

from snake_rl.env.common import manhattan
from snake_rl.env.rewards import compute_reward, potential
from tests.helpers import (
    BLOCK_TAIL_CHASE,
    FAR_FOOD,
    HOOK_COLLIDING_ACTION,
    HOOK_SELF_COLLISION,
    STRAIGHT,
    STRAIGHT_RIGHT,
    TURN_LEFT,
    EnvConfig,
    RewardConfig,
    SnakeEnv,
    make_env,
    serpentine_almost_full,
)

GAMMA = 0.99
VARIANTS = ("R0", "R1", "R2", "R3")


def env_with(variant: str, W: int = 10, **reward_kwargs) -> SnakeEnv:
    env = SnakeEnv(EnvConfig(W=W, reward=RewardConfig(variant=variant, gamma=GAMMA, **reward_kwargs)))
    env.reset(seed=0)
    return env


# ------------------------------------------------------------- event rewards
def _event_rewards(variant: str, **kw) -> dict[str, float]:
    """One reward per event type under ``variant``: ordinary / eat / wall / self / starve / win."""
    out = {}
    env = env_with(variant, **kw)
    env.set_state(STRAIGHT_RIGHT, food=FAR_FOOD)
    out["ordinary"] = env.step(STRAIGHT)[1]

    env = env_with(variant, **kw)
    env.set_state(STRAIGHT_RIGHT, food=(5, 6))
    out["eat"] = env.step(STRAIGHT)[1]

    env = env_with(variant, **kw)
    env.set_state([(5, 7), (5, 8), (5, 9)], food=FAR_FOOD)
    _, out["wall"], term, _, info = env.step(STRAIGHT)
    assert term and info["cause"] == "wall"

    env = env_with(variant, **kw)
    env.set_state(HOOK_SELF_COLLISION, food=FAR_FOOD)
    _, out["self"], term, _, info = env.step(HOOK_COLLIDING_ACTION)
    assert term and info["cause"] == "self"

    env = SnakeEnv(EnvConfig(W=10, starve_limit=5, reward=RewardConfig(variant=variant, gamma=GAMMA, **kw)))
    env.reset(seed=0)
    env.set_state(BLOCK_TAIL_CHASE, food=FAR_FOOD, steps_since_food=4)
    _, out["starve"], term, trunc, info = env.step(TURN_LEFT)
    assert trunc and not term and info["cause"] == "starve"

    env = SnakeEnv(EnvConfig(W=4, reward=RewardConfig(variant=variant, gamma=GAMMA, **kw)))
    env.reset(seed=0)
    cells, food = serpentine_almost_full(4)
    env.set_state(cells, food=food)
    _, out["win"], term, _, info = env.step(STRAIGHT)
    assert term and info["cause"] == "win"
    return out


def test_R0_events():
    r = _event_rewards("R0")
    assert r == {"ordinary": 0.0, "eat": 1.0, "wall": -1.0, "self": -1.0, "starve": 0.0, "win": 1.0}


def test_R1_events():
    r = _event_rewards("R1")
    assert r["ordinary"] == pytest.approx(-0.01)
    assert r["eat"] == pytest.approx(0.99)
    assert r["wall"] == pytest.approx(-1.01) and r["self"] == pytest.approx(-1.01)
    assert r["starve"] == pytest.approx(-0.01)
    assert r["win"] == pytest.approx(0.99)


@pytest.mark.parametrize("variant", VARIANTS)
def test_starvation_is_not_a_death(variant):
    """Truncation carries no death penalty in any variant: it is not a terminal state."""
    r = _event_rewards(variant)
    # Shaping terms are bounded by ~0.07 per step in every variant; a death penalty would show as -1.
    assert r["starve"] > -0.1
    if variant != "R2":  # R2 death reward also carries -Phi(s) (telescoping), tested separately
        assert r["wall"] <= -0.9 and r["self"] <= -0.9


@pytest.mark.parametrize("variant", VARIANTS)
def test_win_is_rewarded_like_eating(variant):
    r = _event_rewards(variant)
    assert r["win"] >= 0.9


# ------------------------------------------------------------- R2
def test_potential_normalisation():
    W = 10
    assert potential(0, W) == 0.0
    assert potential(2 * (W - 1), W) == -1.0
    assert potential(np.array([1, 9]), W).tolist() == pytest.approx([-1 / 18, -0.5])


def test_R2_single_step_values():
    W = 10
    env = env_with("R2")
    # ordinary: head (5,5) -> (5,6); food (0,0): d 10 -> 11
    env.set_state(STRAIGHT_RIGHT, food=FAR_FOOD)
    r = env.step(STRAIGHT)[1]
    assert r == pytest.approx(GAMMA * potential(11, W) - potential(10, W), abs=1e-6)
    # eat: d_prev 1, Phi(s') uses the new food
    env = env_with("R2")
    env.set_state(STRAIGHT_RIGHT, food=(5, 6))
    r = env.step(STRAIGHT)[1]
    d_new = manhattan((5, 6), env.food)
    assert r == pytest.approx(1.0 + GAMMA * potential(d_new, W) - potential(1, W), abs=1e-6)
    # death: Phi(terminal) = 0
    env = env_with("R2")
    env.set_state([(5, 7), (5, 8), (5, 9)], food=FAR_FOOD)  # d_prev = 14
    r = env.step(STRAIGHT)[1]
    assert r == pytest.approx(-1.0 - potential(14, W), abs=1e-6)


def _run_lockstep(seed: int, W: int, starve_limit: int | None):
    """Same seed and actions under R2 and R0; returns per-step F_t, gamma^T Phi(s_T), Phi(s_0)."""
    cfg = dict(W=W, starve_limit=starve_limit)
    e2 = SnakeEnv(EnvConfig(**cfg, reward=RewardConfig(variant="R2", gamma=GAMMA)))
    e0 = SnakeEnv(EnvConfig(**cfg, reward=RewardConfig(variant="R0")))
    e2.reset(seed=seed)
    e0.reset(seed=seed)
    phi_0 = float(potential(manhattan(e2.head, e2.food), W))
    rng = np.random.default_rng(seed)
    F = []
    while True:
        a = int(rng.integers(3))
        _, r2, term, trunc, _ = e2.step(a)
        _, r0, term0, trunc0, _ = e0.step(a)
        assert (term, trunc) == (term0, trunc0)
        F.append(r2 - r0)
        if term or trunc:
            break
    T = len(F)
    phi_T = 0.0 if term else float(potential(manhattan(e2.head, e2.food), W))
    return np.array(F), GAMMA**T * phi_T, phi_0, term


def test_R2_discounted_telescoping_invariant():
    """sum_t gamma^t F_t == gamma^T Phi(s_T) - Phi(s_0), for terminated (Phi(s_T)=0)
    and truncated episodes alike."""
    seen_terminated = seen_truncated = False
    for seed in range(40):
        starve = 15 if seed % 2 else None
        F, tail, phi_0, terminated = _run_lockstep(seed, W=8, starve_limit=starve)
        discounted = float(np.sum(GAMMA ** np.arange(len(F)) * F))
        assert discounted == pytest.approx(tail - phi_0, abs=1e-4)
        seen_terminated |= terminated
        seen_truncated |= not terminated
    assert seen_terminated and seen_truncated


# ------------------------------------------------------------- R3
def test_R3_asymmetric_default():
    env = env_with("R3")
    env.set_state(STRAIGHT_RIGHT, food=(5, 9))  # moving right gets closer
    assert env.step(STRAIGHT)[1] == pytest.approx(0.01)
    env = env_with("R3")
    env.set_state(STRAIGHT_RIGHT, food=(5, 0))  # moving right gets farther
    assert env.step(STRAIGHT)[1] == pytest.approx(0.0)
    assert RewardConfig().farther == 0.0 and RewardConfig().closer == 0.01


def test_R3_symmetric_option():
    env = env_with("R3", farther=-0.01)
    env.set_state(STRAIGHT_RIGHT, food=(5, 0))
    assert env.step(STRAIGHT)[1] == pytest.approx(-0.01)


def test_R3_eat_and_death_include_shaping():
    r = _event_rewards("R3")
    assert r["eat"] == pytest.approx(1.01)  # reaching the food is a 'closer' move
    assert r["ordinary"] == pytest.approx(0.0)  # STRAIGHT_RIGHT moving away from (0, 0)
    assert r["wall"] == pytest.approx(-1.0)  # head (5,9)->(5,10), farther from (0,0)


def test_R3_oscillation_is_net_positive():
    """The hackable property: two moves toward and away from the food net +0.01."""
    env = env_with("R3")
    env.set_state([(4, 4), (4, 5), (5, 5), (5, 4)], food=(9, 9))  # head (5,4) heading LEFT
    total = 0.0
    for a in (TURN_LEFT, TURN_LEFT, TURN_LEFT, TURN_LEFT):  # cycle the 2x2 block
        total += env.step(a)[1]
    assert total == pytest.approx(0.02)  # two closer, two farther, farther = 0


def test_R3_nonzero_every_non_eating_step():
    env = env_with("R3", farther=-0.01)
    for _ in range(50):
        _, r, term, trunc, info = env.step(TURN_LEFT)
        if term or trunc:
            break
        if info["score"] == 0:
            assert abs(r) == pytest.approx(0.01)


# ------------------------------------------------------------- batched API
def test_compute_reward_batched_matches_scalar():
    cfg = RewardConfig(variant="R2", gamma=GAMMA)
    W = 10
    rng = np.random.default_rng(0)
    N = 64
    ate = rng.random(N) < 0.2
    died = (~ate) & (rng.random(N) < 0.2)
    terminal = died | ((~ate) & (rng.random(N) < 0.05))
    d_prev = rng.integers(1, 18, N)
    d_moved = d_prev + rng.choice([-1, 1], N)
    d_next = np.where(ate, rng.integers(1, 18, N), d_moved)
    batched = compute_reward(cfg, W, ate=ate, died=died, terminal=terminal, d_prev=d_prev, d_moved=d_moved, d_next=d_next)
    assert batched.dtype == np.float32 and batched.shape == (N,)
    for i in range(N):
        single = compute_reward(
            cfg, W, ate=ate[i], died=died[i], terminal=terminal[i], d_prev=d_prev[i], d_moved=d_moved[i], d_next=d_next[i]
        )
        assert float(single) == pytest.approx(float(batched[i]))


def test_unknown_variant_rejected():
    with pytest.raises(ValueError):
        compute_reward(RewardConfig(variant="R9"), 10, ate=False, died=False, terminal=False, d_prev=1, d_moved=2, d_next=2)

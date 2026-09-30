"""Non-learning baselines (PLAN.md section 3.8) and the evaluation protocol."""

from __future__ import annotations

import json

import numpy as np
import pytest

from snake_rl.agents import Policy
from snake_rl.agents.baselines import (
    BFSGreedyPolicy,
    HamiltonianPolicy,
    RandomPolicy,
    bfs_first_heading,
    hamiltonian_cycle,
    make_policy,
)
from snake_rl.env.common import DOWN, LEFT, RIGHT, UP, body_age_from_cells, heading_to_action
from snake_rl.env.obs import A_DANGER_LEFT, A_DANGER_RIGHT, A_DANGER_STRAIGHT
from snake_rl.evaluate import EvalResult, env_config_from_dict, evaluate, markdown_table
from tests.helpers import FAR_FOOD, STRAIGHT, STRAIGHT_RIGHT, TURN_LEFT, TURN_RIGHT, EnvConfig, SnakeEnv, make_env

DANGER = [A_DANGER_STRAIGHT, A_DANGER_LEFT, A_DANGER_RIGHT]


# ------------------------------------------------------------- helpers
def test_heading_to_action_relative():
    assert heading_to_action(RIGHT, RIGHT, "relative") == STRAIGHT
    assert heading_to_action(RIGHT, UP, "relative") == TURN_LEFT
    assert heading_to_action(RIGHT, DOWN, "relative") == TURN_RIGHT
    assert heading_to_action(UP, LEFT, "relative") == TURN_LEFT
    assert heading_to_action(LEFT, UP, "relative") == TURN_RIGHT
    with pytest.raises(ValueError):
        heading_to_action(RIGHT, LEFT, "relative")


def test_heading_to_action_absolute():
    for h in range(4):
        assert heading_to_action(RIGHT, h, "absolute") == h


def test_make_policy_and_protocol():
    for name in ("random", "bfs", "hamiltonian"):
        pol = make_policy(name)
        assert isinstance(pol, Policy) and pol.name == name
    with pytest.raises(ValueError):
        make_policy("astar")


# ------------------------------------------------------------- random
def test_random_policy_valid_and_reproducible():
    env = make_env(8)
    a, b = RandomPolicy(seed=5), RandomPolicy(seed=5)
    obs, _ = env.reset(seed=0)
    acts_a = [a.act(obs, env) for _ in range(50)]
    acts_b = [b.act(obs, env) for _ in range(50)]
    assert acts_a == acts_b and set(acts_a) <= {0, 1, 2}
    env4 = make_env(8, action_mode="absolute")
    pol4 = RandomPolicy(seed=1)
    assert set(pol4.act(obs, env4) for _ in range(200)) == {0, 1, 2, 3}


# ------------------------------------------------------------- BFS
def test_bfs_food_straight_ahead():
    env = make_env(10)
    obs = env.set_state(STRAIGHT_RIGHT, food=(5, 8))
    assert bfs_first_heading(env.body_age, env.head, env.food) == RIGHT
    assert BFSGreedyPolicy().act(obs, env) == STRAIGHT


def test_bfs_food_behind_goes_around():
    env = make_env(10)
    obs = env.set_state(STRAIGHT_RIGHT, food=(5, 0))  # behind the head, neck blocks the row
    assert BFSGreedyPolicy().act(obs, env) in (TURN_LEFT, TURN_RIGHT)


def test_bfs_shortest_path_length_respects_body():
    """A vertical wall of body forces a detour; BFS must still find the food."""
    env = make_env(10)
    # Snake tail (0,5) ... head (8,5): a wall across column 5 from row 0 to 8, heading DOWN.
    cells = [(r, 5) for r in range(9)]
    obs = env.set_state(cells, food=(4, 9))
    a = BFSGreedyPolicy().act(obs, env)
    assert a == TURN_LEFT  # heading DOWN, left turn -> RIGHT (east), toward column 9
    _, _, term, _, _ = env.step(a)
    assert not term


def test_bfs_uses_vacating_tail_to_escape_pocket():
    """Head boxed in; the only exit is the tail cell, then the cells behind it
    as they vacate one per step. Static occupancy would find no path."""
    W = 8
    env = make_env(W)
    cells = [(3, 1), (2, 1), (2, 0), (1, 0), (1, 1), (1, 2), (2, 2), (3, 2), (4, 2), (4, 1), (4, 0), (3, 0)]
    obs = env.set_state(cells, food=(0, 7))
    assert env.heading == UP and env.head == (3, 0)
    assert bfs_first_heading(env.body_age, env.head, env.food) == RIGHT
    static = env.body_age.copy()
    static[static > 0] = 10_000  # never vacates
    assert bfs_first_heading(static, env.head, env.food) is None
    pol = BFSGreedyPolicy()
    for _ in range(12):  # follows its own tail out and reaches the food
        obs, _, term, trunc, info = env.step(pol.act(obs, env))
        assert not term and not trunc
        if info["score"] == 1:
            break
    assert info["score"] == 1


#: Head (2,2) boxed into the 3-cell pocket {(2,2), (1,2), (2,1)} by mid-body
#: cells whose ages (5..13) exceed any reachable depth (2); the tail (age 1)
#: is on the far side. Food (0,0) is unreachable even time-aware.
POCKET = [(3, 3), (3, 2), (3, 1), (3, 0), (2, 0), (1, 0), (1, 1), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (2, 2)]


def test_bfs_fallback_when_no_path():
    """No path to the food: first safe action in the order straight / left / right."""
    env = make_env(8)
    obs = env.set_state(POCKET, food=(0, 0))  # heading LEFT; straight -> (2,1) free
    assert env.heading == LEFT
    assert bfs_first_heading(env.body_age, env.head, env.food) is None
    assert BFSGreedyPolicy().act(obs, env) == STRAIGHT
    # Same board with the heading overridden to RIGHT: straight is the neck, left (UP) is free.
    obs = env.set_state(POCKET, food=(0, 0), heading=RIGHT)
    assert BFSGreedyPolicy().act(obs, env) == TURN_LEFT
    # Heading DOWN: straight (3,2) and left (RIGHT -> (2,3)) blocked, right (LEFT -> (2,1)) free.
    obs = env.set_state(POCKET, food=(0, 0), heading=DOWN)
    assert BFSGreedyPolicy().act(obs, env) == TURN_RIGHT


def test_bfs_dies_straight_when_fully_trapped():
    """Nothing safe at all: goes straight (and the episode ends by self collision)."""
    env = make_env(8)
    cells = [(3, 3), (3, 2), (3, 1), (3, 0), (2, 0), (1, 0), (1, 1), (0, 1), (0, 2), (0, 3), (1, 3), (1, 2), (2, 2), (2, 1)]
    obs = env.set_state(cells, food=(0, 0))  # head (2,1) heading LEFT: (2,0) age 5, (1,1) age 7, (3,1) age 3
    assert BFSGreedyPolicy().act(obs, env) == STRAIGHT
    _, _, term, _, info = env.step(STRAIGHT)
    assert term and info["cause"] == "self"


def test_bfs_never_takes_fatal_action_when_a_safe_one_exists():
    pol = BFSGreedyPolicy()
    n_steps = 0
    for seed in range(5):
        env = make_env(8)
        obs, _ = env.reset(seed=seed)
        while True:
            a = pol.act(obs, env)
            danger = obs[DANGER]
            if danger.sum() < 3:
                assert danger[a] == 0.0, (seed, a, danger)
            obs, _, term, trunc, _ = env.step(a)
            n_steps += 1
            if term or trunc:
                break
    assert n_steps > 100


def test_bfs_absolute_mode():
    env = make_env(10, action_mode="absolute")
    obs = env.set_state(STRAIGHT_RIGHT, food=(2, 5))
    assert BFSGreedyPolicy().act(obs, env) == UP


# ------------------------------------------------------------- Hamiltonian
@pytest.mark.parametrize("W", [4, 6, 8, 10, 16])
def test_hamiltonian_cycle_is_a_cycle(W):
    cyc = hamiltonian_cycle(W)
    assert cyc.shape == (W * W, 2)
    assert len({tuple(c) for c in cyc.tolist()}) == W * W
    steps = np.abs(np.diff(np.vstack([cyc, cyc[:1]]), axis=0)).sum(axis=1)
    assert (steps == 1).all()


@pytest.mark.parametrize("W", [3, 5, 7])
def test_hamiltonian_cycle_rejects_odd_W(W):
    with pytest.raises(ValueError):
        hamiltonian_cycle(W)


@pytest.mark.parametrize("W", [4, 6, 8])
def test_hamiltonian_always_wins(W):
    res = evaluate(HamiltonianPolicy(), EnvConfig(W=W), n_episodes=2, seed=7)
    assert [e.cause for e in res.episodes] == ["win", "win"]
    assert all(e.score == W * W - 3 for e in res.episodes)
    assert all(e.length == W * W for e in res.episodes)


def test_hamiltonian_wins_on_10x10_once():
    res = evaluate(HamiltonianPolicy(), EnvConfig(W=10), n_episodes=1, seed=0)
    assert res.episodes[0].cause == "win" and res.episodes[0].score == 97


def test_hamiltonian_absolute_mode():
    res = evaluate(HamiltonianPolicy(), EnvConfig(W=6, action_mode="absolute"), n_episodes=1, seed=3)
    assert res.episodes[0].cause == "win"


def test_hamiltonian_handles_initial_heading_against_cycle():
    """On W = 10 row 5 is swept leftwards but the snake starts heading right;
    the policy must go the other way round rather than reverse into its neck."""
    env = make_env(10)
    obs, _ = env.reset(seed=0)
    pol = HamiltonianPolicy()
    for _ in range(300):
        obs, _, term, trunc, _ = env.step(pol.act(obs, env))
        assert not term and not trunc


# ------------------------------------------------------------- evaluate
def test_evaluate_is_reproducible_and_seeded():
    cfg = EnvConfig(W=8)
    a = evaluate(RandomPolicy(seed=0), cfg, n_episodes=10, seed=123)
    b = evaluate(RandomPolicy(seed=0), cfg, n_episodes=10, seed=123)
    assert [e.seed for e in a.episodes] == list(range(123, 133))
    assert [(e.score, e.steps, e.cause) for e in a.episodes] == [(e.score, e.steps, e.cause) for e in b.episodes]
    s = a.summary()
    assert s["n_episodes"] == 10 and set(s["cause_fractions"]) == {"wall", "self", "starve", "win"}
    assert abs(sum(s["cause_fractions"].values()) - 1.0) < 1e-9
    assert a.ci95 >= 0.0


def test_evaluate_json_roundtrip(tmp_path):
    cfg = EnvConfig(W=8)
    res = evaluate(BFSGreedyPolicy(), cfg, n_episodes=3, seed=1)
    path = res.save(tmp_path / "bfs.json")
    loaded = EvalResult.load(path)
    assert loaded.to_dict() == res.to_dict()
    assert env_config_from_dict(loaded.env) == cfg
    raw = json.loads(path.read_text())
    assert raw["summary"]["n_episodes"] == 3 and "git" in raw


def test_bfs_clearly_beats_random_on_8x8():
    cfg = EnvConfig(W=8)
    rnd = evaluate(RandomPolicy(seed=0), cfg, n_episodes=20, seed=500)
    bfs = evaluate(BFSGreedyPolicy(), cfg, n_episodes=20, seed=500)
    assert bfs.mean_score > rnd.mean_score + 5
    table = markdown_table([rnd, bfs])
    assert table.count("\n") == 3 and "| random |" in table and "| bfs |" in table

"""Tabular Q-learning on observation A (PLAN.md milestone M4): the oracle.

Everything here is derived from the M4 specification, from the textbook
algorithm (Sutton & Barto 2018, eq. 6.8) and from the environments - never
from the implementation under test (``snake_rl/agents/tabular_q.py`` and
``snake_rl/agents/schedules.py``).

Contract under test.

* ``state_index(obs) = sum_k obs[..., k] * 2**k`` over the 11 binary features
  of observation A: row of the ``(2048, n_actions)`` Q table. The bit order is
  part of the contract (saved tables are read back by row).
* ``TabularQAgent.update(obs, actions, rewards, next_obs, terminated)`` is one
  synchronous batch of Q-learning updates. Every target reads the table as it
  was before the call, ``target = r + gamma * (0 if terminated else
  max_a' q[s', a'])``; a cell that occurs several times in the batch moves by
  ``alpha`` times the MEAN of its TD errors while ``visits`` counts every
  occurrence; the return value is the mean ``|td|`` over the transitions.
  With one transition per call this is exactly the textbook rule.
* truncated != terminated: the caller passes the env ``terminated`` flag only
  and takes ``next_obs`` from ``info["final_obs"]``, so a starvation
  truncation bootstraps from its true successor.
* Greedy actions break ties towards the lowest index (``np.argmax``), the same
  in ``act`` and ``act_batch``. ``act_epsilon`` has a fixed RNG protocol:
  ``explore = rng.random(N) < epsilon``, then ``random_actions =
  rng.integers(0, n_actions, size=N)``, both drawn on every call.
* ``linear_schedule(step, start, end, decay_steps) = start + (end - start) *
  min(1, step / decay_steps)``; ``end`` when ``decay_steps <= 0``.

Exactness. Wherever the spec fixes the arithmetic the comparison has no
tolerance: batches of one transition (the textbook rule, evaluated as
``q + alpha * ((r + gamma * max) - q)``) and hand-built batches whose numbers
are dyadic (every intermediate value is exact in binary, whatever the order of
the operations). Tolerances appear in two places only, where the spec gives a
real-number formula without an order of evaluation: real batches with repeated
cells against a cell-by-cell reference (1e-10 on the table, 1e-12 on the
returned mean; the TD errors of one cell may be summed in any order) and
``linear_schedule`` at non-dyadic values (1e-12).

Layout: state index, schedule, construction, the update rule on hand-built
transitions, the update rule on real batches, textbook equivalence (the
central test), action selection, learning sanity (the M4 acceptance
criterion), save / load, determinism.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import FrozenInstanceError, dataclass, fields
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

from snake_rl.agents import Policy
from snake_rl.agents.baselines import BFSGreedyPolicy, RandomPolicy
from snake_rl.agents.schedules import linear_schedule
from snake_rl.agents.tabular_q import N_STATES_A, TabularQAgent, TabularQConfig, state_index
from snake_rl.env.common import LEFT, RIGHT, UP
from snake_rl.env.obs import (
    A_DANGER_LEFT,
    A_DANGER_RIGHT,
    A_DANGER_STRAIGHT,
    A_FOOD_DOWN,
    A_FOOD_LEFT,
    A_FOOD_RIGHT,
    A_FOOD_UP,
    A_HEADING,
    N_FEATURES_A,
)
from snake_rl.env.vec_env import VecSnakeEnv
from snake_rl.evaluate import DEFAULT_EVAL_SEED, evaluate
from tests.helpers import FAR_FOOD, STRAIGHT_RIGHT, TURN_LEFT, EnvConfig, RewardConfig, SnakeEnv

N_STATES = 2048
#: Row ``s`` is the observation whose state index is ``s`` (the inverse of the
#: spec formula), float32 like the observations of the environments.
ALL_STATES = ((np.arange(N_STATES)[:, None] >> np.arange(11)) & 1).astype(np.float32)

#: Four arbitrary, distinct rows for the hand-built transitions.
S_A = 16 + 128 + 512  # 656: heading right, food up and left
S_B = 1 + 16 + 128 + 1024  # 1169: danger straight, heading right, food up and right
S_C = 8 + 256 + 512  # 776: heading up, food down and left
S_D = 7 + 64 + 256 + 1024  # 1351: all three dangers, heading left, food down and right

R0 = RewardConfig(variant="R0")


def obs_of(*states: int) -> np.ndarray:
    """Batch ``(len(states), 11)`` float32 of the observations with these state indices."""
    return ALL_STATES[list(states)].copy()


def spec_index(obs) -> int:
    """State index of ONE observation: the spec formula in Python integers."""
    assert len(obs) == 11
    return sum(int(v) << k for k, v in enumerate(obs))


def apply(agent: TabularQAgent, transitions: list[tuple[int, int, float, int, bool]], reward_dtype=np.float64) -> float:
    """``agent.update`` on transitions written as ``(s, a, r, s', terminated)`` with state indices."""
    s, a, r, s2, terminated = zip(*transitions)
    return agent.update(
        obs_of(*s),
        np.array(a, dtype=np.int64),
        np.array(r, dtype=reward_dtype),
        obs_of(*s2),
        np.array(terminated, dtype=bool),
    )


def live_states() -> set[int]:
    """The 8 x 4 x 8 = 256 state indices a snake can act on: any combination
    of the three danger bits, exactly one heading bit, and the food on one
    side or level on each axis but never on the head (8 directions)."""
    states = set()
    for danger in range(8):
        for heading in range(4):
            for vertical in (0, 1 << A_FOOD_UP, 1 << A_FOOD_DOWN):
                for horizontal in (0, 1 << A_FOOD_LEFT, 1 << A_FOOD_RIGHT):
                    if vertical or horizontal:
                        states.add(danger | 1 << (A_HEADING.start + heading) | vertical | horizontal)
    return states


# ------------------------------------------------------------- state index
def test_state_space_size():
    assert N_STATES_A == N_STATES == 2**N_FEATURES_A
    assert len(live_states()) == 256 and max(live_states()) == S_D < N_STATES_A


def test_state_index_hand_computed_examples():
    obs = np.zeros(11, dtype=np.float32)
    obs[[A_DANGER_STRAIGHT, A_HEADING.start + RIGHT, A_FOOD_UP, A_FOOD_RIGHT]] = 1.0  # bits 0, 4, 7, 10
    assert int(state_index(obs)) == 1 + 16 + 128 + 1024 == S_B

    obs = np.zeros(11, dtype=np.float32)
    obs[[A_HEADING.start + UP, A_FOOD_DOWN, A_FOOD_LEFT]] = 1.0  # bits 3, 8, 9
    assert int(state_index(obs)) == 8 + 256 + 512 == S_C

    obs = np.zeros(11, dtype=np.float32)
    obs[[A_DANGER_STRAIGHT, A_DANGER_LEFT, A_DANGER_RIGHT, A_HEADING.start + LEFT, A_FOOD_DOWN, A_FOOD_RIGHT]] = 1.0
    assert int(state_index(obs)) == 1 + 2 + 4 + 64 + 256 + 1024 == S_D

    # Feature k alone is worth 2**k: the bit order is fixed, not just injectivity.
    for k in range(11):
        one_hot = np.zeros(11, dtype=np.float32)
        one_hot[k] = 1.0
        assert int(state_index(one_hot)) == 2**k, f"feature {k}"


def test_state_index_of_a_real_board():
    env = SnakeEnv(EnvConfig(W=10))
    env.reset(seed=0)
    obs = env.set_state(STRAIGHT_RIGHT, FAR_FOOD)  # head (5, 5) heading right, food (0, 0): up and left
    assert obs.tolist() == [0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0]
    assert int(state_index(obs)) == 16 + 128 + 512 == S_A
    obs, *_ = env.step(TURN_LEFT)  # head (4, 5) heading up; the food is still up and left
    assert int(state_index(obs)) == 8 + 128 + 512


def test_state_index_all_zero_and_all_one():
    assert int(state_index(np.zeros(11, dtype=np.float32))) == 0
    assert int(state_index(np.ones(11, dtype=np.float32))) == 2047 == N_STATES_A - 1


def test_state_index_is_the_identity_on_the_2048_binary_vectors():
    index = state_index(ALL_STATES)
    assert len(set(index.tolist())) == N_STATES  # injective
    assert np.array_equal(index, np.arange(N_STATES))
    for s in (0, 1, 2, 1023, 1024, S_A, S_B, S_C, S_D, 2047):  # batch = element-wise
        assert int(state_index(ALL_STATES[s])) == s == spec_index(ALL_STATES[s])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_state_index_shapes_and_dtype(dtype):
    states = ALL_STATES.astype(dtype)
    one = state_index(states[S_B])  # (11,) -> shape (): a 0-d array or a numpy scalar
    assert np.shape(one) == () and np.asarray(one).dtype == np.int64 and int(one) == S_B

    batch = state_index(states[[3, S_B, 2047, 3]])
    assert isinstance(batch, np.ndarray) and batch.shape == (4,) and batch.dtype == np.int64
    assert batch.tolist() == [3, S_B, 2047, 3]

    single = state_index(states[[S_C]])  # a batch of one keeps its axis
    assert isinstance(single, np.ndarray) and single.shape == (1,) and single.tolist() == [S_C]

    grid = state_index(states[:2040].reshape(8, 255, 11))
    assert isinstance(grid, np.ndarray) and grid.shape == (8, 255) and grid.dtype == np.int64
    assert np.array_equal(grid, np.arange(2040).reshape(8, 255))


@pytest.mark.parametrize("shape", [(10,), (12,), (4, 10), (11, 4), (2, 3, 12), (1,)])
def test_state_index_rejects_a_wrong_last_axis(shape):
    with pytest.raises(ValueError):
        state_index(np.zeros(shape, dtype=np.float32))


def real_play(W: int) -> tuple[np.ndarray, np.ndarray]:
    """Observations of real play on a ``W x W`` board: uniformly random
    actions (64 boards x 400 steps) and the BFS-greedy player (10 episodes).

    Returns ``(acted, successors)``: the observations an action was chosen in,
    and the true successor of every transition (terminal and truncated ones
    included).
    """
    cfg = EnvConfig(W=W)
    acted: list[np.ndarray] = []
    successors: list[np.ndarray] = []

    vec = VecSnakeEnv(cfg, n_envs=64, seed=0)
    rng = np.random.default_rng(0)
    obs = vec.reset()
    for _ in range(400):
        acted.append(obs)
        obs, _, _, _, info = vec.step(rng.integers(0, vec.n_actions, size=vec.n_envs))
        successors.append(info["final_obs"])

    env, policy = SnakeEnv(cfg), BFSGreedyPolicy()
    for episode in range(10):
        o, _ = env.reset(seed=episode)
        while True:
            acted.append(o[None])
            o, _, terminated, truncated, _ = env.step(policy.act(o, env))
            successors.append(o[None])
            if terminated or truncated:
                break
    return np.concatenate(acted), np.concatenate(successors)


#: Distinct states acted on by random + BFS-greedy play, measured with the
#: protocol of ``real_play`` (the environments and their RNG streams are
#: frozen, so the counts are exact; random play alone reaches 137 / 130, the
#: BFS player alone 173 / 178). Quoted in the report: the table has 2048 rows,
#: at most 256 of them are live, and plain play reaches about three quarters
#: of those (200k steps of epsilon-greedy training visited all 256 in every
#: measured run; the learning-sanity test asserts at least 200).
REACHED_STATES = {8: 190, 10: 194}


@pytest.mark.parametrize("W", [8, 10])
def test_real_play_reaches_at_most_256_states(W):
    acted, successors = real_play(W)
    every = np.concatenate([acted, successors])
    assert len(every) > 50_000  # 64 x 400 x 2 observations from random play alone
    assert set(np.unique(every).tolist()) == {0.0, 1.0}

    # Every observation of real play, terminal successors included, has a row,
    # and it is the row of the spec formula.
    index = state_index(every)
    assert index.shape == (len(every),) and index.dtype == np.int64
    assert index.min() >= 0 and index.max() < N_STATES_A
    assert np.array_equal(index, (every.astype(np.int64) << np.arange(11)).sum(axis=1))
    for i in range(0, len(every), 997):
        assert int(state_index(every[i])) == index[i] == spec_index(every[i])

    # 3 danger bits x 4 headings x 8 food directions: a snake can only ever
    # act on 256 of the 2048 rows.
    reached = set(state_index(acted).tolist())
    assert reached <= live_states()
    assert len(reached) <= 256
    assert len(reached) == REACHED_STATES[W]


# ------------------------------------------------------------- schedule
def test_linear_schedule_endpoints_and_midpoint():
    assert linear_schedule(0, 1.0, 0.0, 100) == 1.0
    assert linear_schedule(50, 1.0, 0.0, 100) == 0.5
    assert linear_schedule(100, 1.0, 0.0, 100) == 0.0
    assert linear_schedule(2, 1.0, 0.25, 8) == 0.8125  # 1 + (0.25 - 1) * 2 / 8
    assert linear_schedule(3, 0.0, 8.0, 4) == 6.0  # an increasing schedule
    assert linear_schedule(2.5, 1.0, 0.0, 10.0) == 0.75  # float step and float decay_steps
    assert isinstance(linear_schedule(50, 1.0, 0.0, 100), float)
    # The exploration schedule of the experiments, 1.0 -> 0.05 (not dyadic).
    assert linear_schedule(0, 1.0, 0.05, 40_000) == 1.0
    assert linear_schedule(20_000, 1.0, 0.05, 40_000) == pytest.approx(0.525, abs=1e-12)
    assert linear_schedule(40_000, 1.0, 0.05, 40_000) == pytest.approx(0.05, abs=1e-12)
    assert linear_schedule(10_000, 1.0, 0.05, 40_000) == pytest.approx(1.0 + (0.05 - 1.0) * 0.25, abs=1e-12)


def test_linear_schedule_clamps_after_decay_steps():
    assert linear_schedule(101, 1.0, 0.0, 100) == 0.0
    assert linear_schedule(10**9, 1.0, 0.25, 8) == 0.25
    assert linear_schedule(9, 0.0, 8.0, 4) == 8.0
    at_end = linear_schedule(40_000, 1.0, 0.05, 40_000)
    for step in (40_001, 50_000, 10**7):
        assert linear_schedule(step, 1.0, 0.05, 40_000) == at_end  # flat, bit for bit


@pytest.mark.parametrize("decay_steps", [0, 0.0, -1, -250.5])
def test_linear_schedule_without_decay_returns_end(decay_steps):
    for step in (0, 1, 5000):
        assert linear_schedule(step, 1.0, 0.05, decay_steps) == 0.05
        assert linear_schedule(step, 0.2, 0.9, decay_steps) == 0.9


def test_linear_schedule_is_monotone_and_pure():
    steps = list(range(0, 60_000, 997))
    values = [linear_schedule(step, 1.0, 0.05, 40_000) for step in steps]
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert all(a > b for a, b, step in zip(values, values[1:], steps[1:]) if step <= 40_000)  # strictly, while decaying
    assert max(values) == values[0] == 1.0 and min(values) == pytest.approx(0.05, abs=1e-12)
    rising = [linear_schedule(step, 0.1, 0.7, 30_000) for step in steps]
    assert all(a <= b for a, b in zip(rising, rising[1:]))
    # Pure function: no hidden state, whatever the order of the calls.
    assert [linear_schedule(step, 1.0, 0.05, 40_000) for step in reversed(steps)] == values[::-1]


def test_linear_schedule_on_arrays_of_steps():
    out = linear_schedule(np.array([0, 25, 50, 100, 150]), 1.0, 0.0, 100)
    assert isinstance(out, np.ndarray) and out.shape == (5,)
    assert out.tolist() == [1.0, 0.75, 0.5, 0.0, 0.0]
    steps = np.arange(0, 60_000, 997)
    out = linear_schedule(steps, 1.0, 0.05, 40_000)
    assert out.shape == steps.shape
    assert np.allclose(out, [linear_schedule(int(step), 1.0, 0.05, 40_000) for step in steps], rtol=0.0, atol=1e-12)


# ------------------------------------------------------------- construction
def test_config_defaults_and_immutability():
    cfg = TabularQConfig()
    assert [f.name for f in fields(TabularQConfig)] == [
        "alpha", "gamma", "eps_start", "eps_end", "eps_decay_fraction", "q_init"]
    assert (cfg.alpha, cfg.gamma, cfg.eps_start, cfg.eps_end, cfg.eps_decay_fraction, cfg.q_init) == (
        0.1, 0.99, 1.0, 0.05, 0.2, 0.0)
    with pytest.raises(FrozenInstanceError):
        cfg.alpha = 0.5
    assert TabularQConfig(alpha=0.2) == TabularQConfig(alpha=0.2) != cfg


@pytest.mark.parametrize("n_actions", [3, 4])
def test_new_agent_has_an_empty_table(n_actions):
    agent = TabularQAgent(n_actions)
    assert agent.cfg == TabularQConfig() and agent.n_actions == n_actions
    assert isinstance(agent.q, np.ndarray) and agent.q.shape == (2048, n_actions) and agent.q.dtype == np.float64
    assert isinstance(agent.visits, np.ndarray) and agent.visits.shape == (2048, n_actions)
    assert agent.visits.dtype == np.int64
    assert (agent.q == 0.0).all() and (agent.visits == 0).all()
    cfg = TabularQConfig(alpha=0.3, gamma=0.9)
    assert TabularQAgent(n_actions, cfg).cfg == cfg


def test_q_init_fills_the_table_and_unseen_states_bootstrap_from_it():
    agent = TabularQAgent(4, TabularQConfig(alpha=0.5, gamma=0.5, q_init=2.0))
    assert agent.q.shape == (2048, 4) and agent.q.dtype == np.float64 and (agent.q == 2.0).all()
    assert (agent.visits == 0).all()
    out = apply(agent, [(S_A, 3, 3.0, S_B, False)])
    # target = 3 + 0.5 * 2 = 4, td = 4 - 2 = 2, q = 2 + 0.5 * 2
    assert agent.q[S_A, 3] == 3.0 and out == 2.0
    assert TabularQAgent(3, TabularQConfig(q_init=-0.25)).q.tolist() == [[-0.25] * 3] * 2048


def test_agents_do_not_share_their_tables():
    a, b = TabularQAgent(3), TabularQAgent(3)
    assert not np.shares_memory(a.q, b.q) and not np.shares_memory(a.visits, b.visits)
    apply(a, [(S_A, 1, 1.0, S_B, True)])
    assert a.q[S_A, 1] != 0.0 and a.visits[S_A, 1] == 1
    assert (b.q == 0.0).all() and (b.visits == 0).all()
    assert (TabularQAgent(3).q == 0.0).all()  # nor does a later instance inherit anything


# ------------------------------------------------------------- update: hand-built transitions
# alpha = gamma = 0.5 and dyadic values: every expected number below is exact.
def half_agent(n_actions: int = 3) -> TabularQAgent:
    return TabularQAgent(n_actions, TabularQConfig(alpha=0.5, gamma=0.5))


@pytest.mark.parametrize(
    "next_row, expected_q, expected_abs_td",
    [
        ([1.0, 3.0, 2.0], 1.375, 2.25),  # target 1 + 0.5 * 3 = 2.5, td 2.25
        ([4.0, -1.0, 0.0], 1.625, 2.75),  # target 1 + 0.5 * 4 = 3.0, td 2.75
        ([-3.0, -2.0, -1.0], 0.375, 0.25),  # target 1 + 0.5 * -1 = 0.5, td 0.25 (the max may be negative)
    ],
)
def test_update_non_terminal_bootstraps_with_gamma_times_max(next_row, expected_q, expected_abs_td):
    agent = half_agent()
    agent.q[S_B] = next_row
    agent.q[S_A, 1] = 0.25
    out = apply(agent, [(S_A, 1, 1.0, S_B, False)])
    assert agent.q[S_A, 1] == expected_q
    assert type(out) is float and out == expected_abs_td
    assert agent.q[S_A].tolist() == [0.0, expected_q, 0.0] and agent.q[S_B].tolist() == next_row
    assert agent.visits[S_A].tolist() == [0, 1, 0] and agent.visits.sum() == 1


def test_update_of_one_transition_is_the_textbook_rule():
    """Default config (alpha 0.1, gamma 0.99), numbers that are not dyadic:
    Q(S, A) + alpha * [R + gamma * max_a Q(S', a) - Q(S, A)], no tolerance."""
    agent = TabularQAgent(3)
    agent.q[S_B] = [0.3, 0.7, -0.2]
    agent.q[S_A, 2] = 0.4
    out = apply(agent, [(S_A, 2, -0.01, S_B, False)])
    td = (-0.01 + 0.99 * 0.7) - 0.4
    assert agent.q[S_A, 2] == 0.4 + 0.1 * td
    assert out == abs(td)


def test_update_terminal_uses_the_reward_alone():
    agent = half_agent()
    agent.q[S_B] = [5.0, 7.0, 9.0]  # the value of a terminal successor is zero, whatever the table says
    agent.q[S_A, 1] = 0.5
    out = apply(agent, [(S_A, 1, -1.0, S_B, True)])
    assert agent.q[S_A, 1] == -0.25 and out == 1.5  # td = -1 - 0.5

    # The same transition, not terminated: target = -1 + 0.5 * 9 = 3.5, td = 3.
    twin = half_agent()
    twin.q[S_B] = [5.0, 7.0, 9.0]
    twin.q[S_A, 1] = 0.5
    assert apply(twin, [(S_A, 1, -1.0, S_B, False)]) == 3.0 and twin.q[S_A, 1] == 2.0


def test_update_terminated_flag_is_per_transition():
    agent = half_agent()
    agent.q[S_B] = [5.0, 7.0, 9.0]
    out = apply(agent, [(S_A, 0, 1.0, S_B, True), (S_C, 0, 1.0, S_B, False), (S_D, 2, 0.0, S_B, True)])
    assert agent.q[S_A, 0] == 0.5  # td = 1
    assert agent.q[S_C, 0] == 2.75  # target = 1 + 0.5 * 9 = 5.5
    assert agent.q[S_D, 2] == 0.0  # td = 0
    assert out == (1.0 + 5.5 + 0.0) / 3


def test_update_bootstraps_a_starvation_truncation_from_final_obs():
    """A truncated transition has ``terminated`` False: its target bootstraps,
    and from the true successor ``info["final_obs"]``, not from the first
    observation of the next episode that ``step`` returns."""
    H = 4
    vec = VecSnakeEnv(EnvConfig(W=8, starve_limit=H), n_envs=1, seed=3)
    vec.reset()
    vec.set_state(0, [(1, 2), (1, 3), (1, 4)], FAR_FOOD, steps_since_food=H - 1)
    obs = vec.observe()
    actions = np.array([TURN_LEFT])
    reset_obs, reward, terminated, truncated, info = vec.step(actions)
    assert truncated.tolist() == [True] and terminated.tolist() == [False] and reward.tolist() == [0.0]

    s, s_true, s_reset = spec_index(obs[0]), spec_index(info["final_obs"][0]), spec_index(reset_obs[0])
    assert s == S_A  # heading right, food up and left
    assert s_true == 1 + 8 + 512  # head (0, 4) heading up: wall ahead, food level and to the left
    assert len({s, s_true, s_reset}) == 3

    agent = half_agent()
    agent.q[s_true] = [2.0, 8.0, 4.0]
    agent.q[s_reset] = [64.0, 64.0, 64.0]
    out = agent.update(obs, actions, reward, info["final_obs"], terminated)
    # target = 0 + 0.5 * 8 = 4. Treating it as terminal would give 0; the post-reset row 32.
    assert agent.q[s, TURN_LEFT] == 2.0 and out == 4.0
    assert agent.visits[s, TURN_LEFT] == 1


@pytest.mark.parametrize("order", [(0, 1), (1, 0)], ids=["source-first", "reader-first"])
def test_update_targets_use_the_table_from_before_the_call(order):
    # S_C -> S_A -> S_B in one batch: the value of S_A changes in this very
    # call, but the transition out of S_C must still see the old one (0).
    transitions = [(S_A, 0, 4.0, S_B, False), (S_C, 1, 0.0, S_A, False)]
    agent = half_agent()
    out = apply(agent, [transitions[i] for i in order])
    assert agent.q[S_A, 0] == 2.0  # 0 + 0.5 * (4 + 0.5 * 0)
    assert agent.q[S_C, 1] == 0.0  # a sequential sweep would give 0.5 * (0.5 * 2.0) = 0.5
    assert out == 2.0  # (|4| + |0|) / 2

    # Two actions of one state looping back to that state.
    agent = half_agent()
    loop = [(S_A, 0, 4.0, S_A, False), (S_A, 1, 0.0, S_A, False)]
    out = apply(agent, [loop[i] for i in order])
    assert agent.q[S_A].tolist() == [2.0, 0.0, 0.0] and out == 2.0


def test_update_duplicate_cells_take_the_mean_td_and_count_every_visit():
    agent = half_agent()
    agent.q[S_A, 2] = 1.0
    agent.q[S_B] = [0.0, 4.0, 0.0]
    out = apply(
        agent,
        [
            (S_A, 2, 1.0, S_D, True),  # target 1, td 0
            (S_C, 0, -1.0, S_D, True),  # target -1, td -1 (another cell, in between)
            (S_A, 2, 2.0, S_B, False),  # target 2 + 0.5 * 4 = 4, td 3
            (S_A, 2, 7.0, S_D, True),  # target 7, td 6
        ],
    )
    # Cell (S_A, 2): mean td = (0 + 3 + 6) / 3 = 3 -> 1 + 0.5 * 3. The sum would
    # give 5.5 and three sequential updates 4.75.
    assert agent.q[S_A, 2] == 2.5
    assert agent.q[S_C, 0] == -0.5
    assert agent.visits[S_A, 2] == 3 and agent.visits[S_C, 0] == 1 and agent.visits.sum() == 4
    # Mean |td| over the four transitions; not |mean td| = 2 and not the mean over the two cells.
    assert out == 2.5

    # Visits accumulate over calls.
    apply(agent, [(S_A, 2, 0.0, S_D, True), (S_A, 2, 0.0, S_D, True)])
    assert agent.q[S_A, 2] == 1.25  # both td = -2.5
    assert agent.visits[S_A, 2] == 5 and agent.visits.sum() == 6


def test_update_leaves_untouched_cells_bit_identical():
    agent = TabularQAgent(4)
    rng = np.random.default_rng(0)
    agent.q[:] = rng.normal(size=agent.q.shape)
    before = agent.q.copy()
    transitions = [
        (S_A, 0, 1.0, S_B, False),
        (S_C, 3, -1.0, S_D, True),
        (S_A, 0, 0.5, S_C, False),
        (S_D, 2, 0.25, S_A, False),
        (0, 1, 0.0, 2047, False),
    ]
    touched = {(s, a) for s, a, *_ in transitions}
    apply(agent, transitions)
    changed = np.argwhere(agent.q.view(np.uint64) != before.view(np.uint64))
    assert {(int(s), int(a)) for s, a in changed} == touched
    assert {(int(s), int(a)) for s, a in np.argwhere(agent.visits > 0)} == touched
    assert agent.visits[S_A, 0] == 2 and agent.visits.sum() == 5
    assert agent.q.dtype == np.float64 and agent.q.shape == (2048, 4)
    assert agent.visits.dtype == np.int64 and agent.visits.shape == (2048, 4)


def test_update_computes_float32_rewards_in_float64():
    reward32 = np.float32(0.1)
    reward = float(reward32)  # 0.10000000149011612: the widening is exact
    td = (reward + 0.99 * 0.3) - 0.05
    assert float(np.float32(td)) != td  # the expected numbers are not representable in float32

    agent = TabularQAgent(3)  # alpha 0.1, gamma 0.99
    agent.q[S_B] = [0.3, 0.1, 0.2]
    agent.q[S_A, 0] = 0.05
    out = apply(agent, [(S_A, 0, reward32, S_B, False)], reward_dtype=np.float32)
    assert agent.q[S_A, 0] == 0.05 + 0.1 * td
    assert type(out) is float and out == abs(td)
    assert agent.q.dtype == np.float64

    twin = TabularQAgent(3)  # the same reward handed over as float64: the same table
    twin.q[S_B] = [0.3, 0.1, 0.2]
    twin.q[S_A, 0] = 0.05
    assert apply(twin, [(S_A, 0, reward, S_B, False)], reward_dtype=np.float64) == out
    assert np.array_equal(twin.q, agent.q)


def test_update_does_not_write_to_its_arguments():
    agent = half_agent()
    args = [
        obs_of(S_A, S_C, S_A),
        np.array([0, 1, 0], dtype=np.int64),
        np.array([1.0, -1.0, 0.5], dtype=np.float32),
        obs_of(S_B, S_D, S_B),
        np.array([False, True, False]),
    ]
    copies = [arg.copy() for arg in args]
    for arg in args:
        arg.flags.writeable = False
    agent.update(*args)
    for arg, copy in zip(args, copies):
        assert np.array_equal(arg, copy) and arg.dtype == copy.dtype
    assert agent.q[S_A, 0] == 0.375 and agent.q[S_C, 1] == -0.5  # mean td 0.75; td -1


# ------------------------------------------------------------- update: real batches
def reference_batch_update(
    q: np.ndarray, visits: np.ndarray, s, a, r, s2, terminated, alpha: float, gamma: float
) -> float:
    """Spec S2 on plain state indices, cell by cell, in place on ``q`` / ``visits``."""
    old = q.copy()
    tds = [
        (float(r[n]) + gamma * (0.0 if terminated[n] else float(old[s2[n]].max()))) - float(old[s[n], a[n]])
        for n in range(len(s))
    ]
    cells: dict[tuple[int, int], list[float]] = defaultdict(list)
    for n, td in enumerate(tds):
        cells[(s[n], int(a[n]))].append(td)
    for (state, action), cell_tds in cells.items():
        q[state, action] = old[state, action] + alpha * (sum(cell_tds) / len(cell_tds))
        visits[state, action] += len(cell_tds)
    return sum(abs(td) for td in tds) / len(tds)


def test_batch_update_matches_a_cell_by_cell_reference_on_real_transitions():
    """32 boards that all start in the same position: batches full of repeated
    cells, terminal and truncated transitions. The transitions come from an
    epsilon-greedy player on the REFERENCE table, so the stream of batches does
    not depend on the agent."""
    cfg = EnvConfig(W=8, starve_limit=10, reward=RewardConfig(variant="R2", gamma=0.9))
    acfg = TabularQConfig(alpha=0.2, gamma=0.9, q_init=0.1)
    n = 32
    vec = VecSnakeEnv(cfg, n_envs=n, seed=500)
    agent = TabularQAgent(vec.n_actions, acfg)
    q_ref = np.full((2048, vec.n_actions), acfg.q_init)
    visits_ref = np.zeros((2048, vec.n_actions), dtype=np.int64)
    rng = np.random.default_rng(5)
    obs = vec.reset()
    n_terminated = n_truncated = batches_with_triples = 0
    for t in range(150):
        s = [spec_index(o) for o in obs]
        actions = np.where(rng.random(n) < 0.3, rng.integers(0, vec.n_actions, size=n), q_ref[s].argmax(axis=1))
        next_obs, reward, terminated, truncated, info = vec.step(actions)
        s2 = [spec_index(o) for o in info["final_obs"]]
        want = reference_batch_update(q_ref, visits_ref, s, actions, reward, s2, terminated, acfg.alpha, acfg.gamma)
        got = agent.update(obs, actions, reward, info["final_obs"], terminated)

        assert got == pytest.approx(want, rel=0.0, abs=1e-12), f"step {t}"
        assert np.array_equal(agent.visits, visits_ref), f"step {t}"
        assert np.allclose(agent.q, q_ref, rtol=0.0, atol=1e-10), f"step {t}"
        n_terminated += int(terminated.sum())
        n_truncated += int(truncated.sum())
        cell_counts = np.unique(np.array(s) * vec.n_actions + actions, return_counts=True)[1]
        batches_with_triples += int(cell_counts.max() >= 3)
        obs = next_obs
    assert agent.visits.sum() == 150 * n
    assert n_terminated >= 100 and n_truncated >= 20 and batches_with_triples >= 100, (
        n_terminated, n_truncated, batches_with_triples)


# ------------------------------------------------------------- textbook equivalence
class TextbookQLearning:
    """Sutton & Barto (2018), eq. 6.8, one transition at a time in Python floats:

        Q(S, A) <- Q(S, A) + alpha * [R + gamma * max_a Q(S', a) - Q(S, A)]

    with a zero value for a terminal ``S'``. The table is a dict keyed by the
    11 feature bits, so nothing in here knows about state indices or arrays.
    """

    def __init__(self, n_actions: int, alpha: float, gamma: float, q_init: float) -> None:
        self.n_actions, self.alpha, self.gamma, self.q_init = n_actions, alpha, gamma, q_init
        self.q: dict[tuple[int, ...], list[float]] = {}
        self.visits: dict[tuple[int, ...], list[int]] = {}

    @staticmethod
    def key(obs: np.ndarray) -> tuple[int, ...]:
        return tuple(int(v) for v in obs)

    def row(self, obs: np.ndarray) -> list[float]:
        return self.q.setdefault(self.key(obs), [self.q_init] * self.n_actions)

    def greedy(self, obs: np.ndarray) -> int:
        row = self.row(obs)
        best = 0
        for action in range(1, self.n_actions):
            if row[action] > row[best]:  # strict: the lowest index wins a tie
                best = action
        return best

    def learn(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, terminated: bool) -> float:
        row = self.row(obs)
        target = reward if terminated else reward + self.gamma * max(self.row(next_obs))
        td = target - row[action]
        row[action] = row[action] + self.alpha * td
        self.visits.setdefault(self.key(obs), [0] * self.n_actions)[action] += 1
        return abs(td)

    def tables(self) -> tuple[np.ndarray, np.ndarray]:
        """The dicts laid out as ``(2048, n_actions)`` arrays by the spec index."""
        q = np.full((N_STATES, self.n_actions), self.q_init, dtype=np.float64)
        visits = np.zeros((N_STATES, self.n_actions), dtype=np.int64)
        for key, row in self.q.items():
            q[spec_index(key)] = row
        for key, counts in self.visits.items():
            visits[spec_index(key)] = counts
        return q, visits


@dataclass
class LockstepTally:
    terminated: int = 0
    truncated: int = 0
    eats: int = 0
    explored: int = 0
    greedy_with_learned_values: int = 0


def run_textbook_lockstep(
    cfg: EnvConfig, acfg: TabularQConfig, n_steps: int, env_seed: int, explore_seed: int
) -> tuple[TabularQAgent, TextbookQLearning, LockstepTally]:
    """Textbook Q-learning on the scalar env and the agent on ``VecSnakeEnv(n_envs=1)``,
    same seeds, compared after every step.

    Reference side: one ``SnakeEnv``, ``reset()`` without a seed after an
    episode end (the food stream continues, as in the vec env), terminal ->
    no bootstrap, starvation truncation -> bootstrap. Exploration follows the
    RNG protocol of ``act_epsilon`` for ``N = 1`` on its own generator.
    Agent side: exactly the calls of the training loop.
    """
    env = SnakeEnv(cfg)
    n_actions = int(env.action_space.n)
    ref = TextbookQLearning(n_actions, acfg.alpha, acfg.gamma, acfg.q_init)
    ref_rng = np.random.default_rng(explore_seed)
    ref_obs, _ = env.reset(seed=env_seed)

    vec = VecSnakeEnv(cfg, n_envs=1, seed=env_seed)
    agent = TabularQAgent(vec.n_actions, acfg)
    agent_rng = np.random.default_rng(explore_seed)
    obs = vec.reset()

    tally = LockstepTally()
    for t in range(n_steps):
        where = f"step {t}"
        epsilon = max(0.1, 1.0 - 0.9 * t / (0.5 * n_steps))  # 1.0 -> 0.1 over the first half

        # --- reference: epsilon-greedy, one environment step, one textbook update
        explore = ref_rng.random(1)[0] < epsilon
        random_action = int(ref_rng.integers(0, n_actions, size=1)[0])
        action = random_action if explore else ref.greedy(ref_obs)
        tally.explored += int(explore)
        tally.greedy_with_learned_values += int(not explore and len(set(ref.row(ref_obs))) > 1)
        score_before = env.score
        ref_next, ref_reward, ref_terminated, ref_truncated, ref_info = env.step(action)
        abs_td = ref.learn(ref_obs, action, ref_reward, ref_next, ref_terminated)
        cell = (spec_index(ref_obs), action)
        new_value = ref.row(ref_obs)[action]
        tally.eats += ref_info["score"] - score_before
        tally.terminated += int(ref_terminated)
        tally.truncated += int(ref_truncated)
        ref_obs = env.reset()[0] if (ref_terminated or ref_truncated) else ref_next

        # --- agent: the calls of the training loop
        actions = agent.act_epsilon(obs, epsilon, agent_rng)
        assert actions.tolist() == [action], f"{where}: the agent chose {actions.tolist()}, the textbook loop {action}"
        next_obs, reward, terminated, truncated, info = vec.step(actions)
        got = agent.update(obs, actions, reward, info["final_obs"], terminated)
        assert got == abs_td, f"{where}: update returned {got!r}, textbook |td| is {abs_td!r}"
        assert agent.q[cell] == new_value, f"{where}: q{cell} is {agent.q[cell]!r}, textbook value {new_value!r}"
        obs = next_obs
        assert np.array_equal(obs[0], ref_obs), f"{where}: the two environments drifted apart"
    return agent, ref, tally


#: Small starve limit so that truncations are frequent. The R2 cases also move
#: every hyper-parameter away from its default (the shaping discount equals
#: the agent discount, as the R2 guarantee requires).
EQUIVALENCE_CASES = {
    "relative-R0": (EnvConfig(W=8, starve_limit=12, reward=R0), TabularQConfig(), 11),
    "absolute-R0": (EnvConfig(W=8, action_mode="absolute", starve_limit=12, reward=R0), TabularQConfig(), 22),
    "relative-R2": (
        EnvConfig(W=8, starve_limit=12, reward=RewardConfig(variant="R2", gamma=0.9)),
        TabularQConfig(alpha=0.3, gamma=0.9, q_init=0.25),
        33,
    ),
    "absolute-R2": (
        EnvConfig(W=8, action_mode="absolute", starve_limit=12, reward=RewardConfig(variant="R2", gamma=0.9)),
        TabularQConfig(alpha=0.3, gamma=0.9, q_init=0.25),
        44,
    ),
}
EQUIVALENCE_STEPS = 2500


@pytest.mark.parametrize("name", list(EQUIVALENCE_CASES))
def test_agent_on_vec_env_equals_textbook_q_learning_on_scalar_env(name):
    cfg, acfg, seed = EQUIVALENCE_CASES[name]
    agent, ref, tally = run_textbook_lockstep(cfg, acfg, EQUIVALENCE_STEPS, env_seed=seed, explore_seed=seed + 1)

    q, visits = ref.tables()
    assert agent.q.dtype == np.float64 and agent.q.shape == q.shape
    assert np.array_equal(agent.q, q)  # bit-identical tables, no tolerance
    assert agent.visits.dtype == np.int64 and np.array_equal(agent.visits, visits)
    assert visits.sum() == EQUIVALENCE_STEPS

    # The run is only evidence if it went through every branch: terminal and
    # truncated transitions, food, exploration, greedy choices on learned values.
    assert tally.terminated >= 50 and tally.truncated >= 20 and tally.eats >= 50, tally
    assert tally.explored >= 500 and tally.greedy_with_learned_values >= 1000, tally
    assert (visits > 0).sum() >= 150, int((visits > 0).sum())


# ------------------------------------------------------------- action selection
def table_agent(n_actions: int, seed: int, ties: bool = False) -> TabularQAgent:
    """Agent with an arbitrary table; ``ties`` fills it with 0 / 1 / 2 so that most rows have a tie."""
    agent = TabularQAgent(n_actions)
    rng = np.random.default_rng(seed)
    agent.q[:] = rng.integers(0, 3, size=agent.q.shape) if ties else rng.normal(size=agent.q.shape)
    return agent


def protocol_actions(greedy: np.ndarray, epsilon: float, n_actions: int, rng: np.random.Generator) -> np.ndarray:
    """The RNG protocol of ``act_epsilon``, replicated: two draws, in this order, on every call."""
    explore = rng.random(len(greedy)) < epsilon
    random_actions = rng.integers(0, n_actions, size=len(greedy))
    return np.where(explore, random_actions, greedy)


def test_policy_protocol():
    agent = TabularQAgent(3)
    assert isinstance(agent, Policy)
    assert agent.name == "tabular_q" and TabularQAgent.name == "tabular_q"
    agent.q[S_A] = [0.0, 0.0, 1.0]
    q, visits = agent.q.copy(), agent.visits.copy()
    assert agent.reset() is None
    env = SnakeEnv(EnvConfig(W=10))
    env.reset(seed=0)
    obs = env.set_state(STRAIGHT_RIGHT, FAR_FOOD)  # state S_A
    assert agent.act(obs) == 2 and agent.act(obs, env) == 2 and agent.act(obs, None) == 2
    assert type(agent.act(obs)) is int
    assert np.array_equal(agent.q, q) and np.array_equal(agent.visits, visits)  # acting learns nothing


def test_untrained_agent_plays_through_the_evaluation_protocol():
    """An empty table is all ties -> action 0 (straight) -> into the right wall after 4 steps on 8x8."""
    result = evaluate(TabularQAgent(3), EnvConfig(W=8), n_episodes=5, seed=DEFAULT_EVAL_SEED)
    assert result.policy == "tabular_q"
    assert [e.cause for e in result.episodes] == ["wall"] * 5 and [e.steps for e in result.episodes] == [4] * 5


@pytest.mark.parametrize("n_actions", [3, 4])
@pytest.mark.parametrize("ties", [False, True], ids=["distinct", "ties"])
def test_act_agrees_with_act_batch_row_by_row(n_actions, ties):
    agent = table_agent(n_actions, seed=n_actions, ties=ties)
    batch = agent.act_batch(ALL_STATES)
    assert isinstance(batch, np.ndarray) and batch.shape == (2048,) and batch.dtype == np.int64
    assert np.array_equal(batch, np.argmax(agent.q, axis=1))
    if ties:
        assert ((agent.q == agent.q.max(axis=1, keepdims=True)).sum(axis=1) > 1).mean() > 0.3
    for s in range(N_STATES):
        action = agent.act(ALL_STATES[s])
        assert type(action) is int and action == batch[s], f"state {s}"

    # Any batch, in any order, with repeats.
    pick = np.random.default_rng(1).integers(0, N_STATES, size=300)
    some = agent.act_batch(ALL_STATES[pick])
    assert some.shape == (300,) and some.dtype == np.int64 and np.array_equal(some, batch[pick])
    one = agent.act_batch(ALL_STATES[[S_C]])
    assert one.shape == (1,) and one.dtype == np.int64 and one[0] == batch[S_C]


def test_ties_pick_the_lowest_index():
    agent = TabularQAgent(4)
    rows = {
        S_A: ([1.0, 1.0, 0.0, 0.0], 0),
        S_B: ([0.0, 2.0, 2.0, 2.0], 1),
        S_C: ([-1.0, -1.0, -1.0, -1.0], 0),
        S_D: ([0.0, 1.0, 3.0, 3.0], 2),
        5: ([7.0, 0.0, 0.0, 7.0], 0),
    }
    for s, (row, _) in rows.items():
        agent.q[s] = row
    states = list(rows)
    expected = [best for _, best in rows.values()]
    assert [agent.act(ALL_STATES[s]) for s in states] == expected
    assert agent.act_batch(obs_of(*states)).tolist() == expected
    assert agent.act_epsilon(obs_of(*states), 0.0, np.random.default_rng(0)).tolist() == expected
    # A fresh table is one big tie: action 0 everywhere.
    fresh = TabularQAgent(3, TabularQConfig(q_init=0.5))
    assert not fresh.act_batch(ALL_STATES).any() and fresh.act(ALL_STATES[S_A]) == 0


@pytest.mark.parametrize("n_actions", [3, 4])
def test_act_epsilon_zero_is_greedy(n_actions):
    agent = table_agent(n_actions, seed=10, ties=True)
    actions = agent.act_epsilon(ALL_STATES, 0.0, np.random.default_rng(3))
    assert isinstance(actions, np.ndarray) and actions.shape == (2048,) and actions.dtype == np.int64
    assert np.array_equal(actions, agent.act_batch(ALL_STATES))
    one = agent.act_epsilon(ALL_STATES[[S_B]], 0.0, np.random.default_rng(3))
    assert one.shape == (1,) and one.dtype == np.int64 and one[0] == agent.act(ALL_STATES[S_B])


@pytest.mark.parametrize("n_actions", [3, 4])
def test_act_epsilon_one_is_uniform_over_the_actions(n_actions):
    n = 30_000
    agent = TabularQAgent(n_actions)  # greedy would be action 0 on every row
    obs = np.repeat(obs_of(S_A), n, axis=0)
    actions = agent.act_epsilon(obs, 1.0, np.random.default_rng(42))
    assert actions.shape == (n,) and actions.dtype == np.int64
    frequency = np.bincount(actions, minlength=n_actions) / n
    assert frequency.shape == (n_actions,)
    assert np.abs(frequency - 1.0 / n_actions).max() < 0.01, frequency  # about 4 standard errors
    # And it is exactly the second draw of the protocol.
    rng = np.random.default_rng(42)
    rng.random(n)
    assert np.array_equal(actions, rng.integers(0, n_actions, size=n))


def test_act_epsilon_explores_with_probability_epsilon():
    n, epsilon = 30_000, 0.3
    agent = TabularQAgent(3)
    agent.q[S_A] = [0.0, 1.0, 0.0]  # greedy action 1
    actions = agent.act_epsilon(np.repeat(obs_of(S_A), n, axis=0), epsilon, np.random.default_rng(7))
    frequency = np.bincount(actions, minlength=3) / n
    # P(greedy) = 1 - eps + eps / 3 = 0.8, P(other) = eps / 3 = 0.1
    assert np.abs(frequency - np.array([0.1, 0.8, 0.1])).max() < 0.01, frequency


@pytest.mark.parametrize("n_actions", [3, 4])
@pytest.mark.parametrize("epsilon", [0.0, 0.05, 0.3, 1.0])
def test_act_epsilon_follows_the_rng_protocol(n_actions, epsilon):
    agent = table_agent(n_actions, seed=20)
    q = agent.q.copy()
    greedy = np.argmax(q, axis=1)
    rng, twin = np.random.default_rng(99), np.random.default_rng(99)
    pick_rng = np.random.default_rng(0)
    for n in (1, 64, 1, 7, 500):  # the stream is shared by consecutive calls of any size
        pick = pick_rng.integers(0, N_STATES, size=n)
        actions = agent.act_epsilon(ALL_STATES[pick], epsilon, rng)
        assert actions.shape == (n,) and actions.dtype == np.int64
        assert np.array_equal(actions, protocol_actions(greedy[pick], epsilon, n_actions, twin)), f"batch of {n}"
        assert rng.bit_generator.state == twin.bit_generator.state, f"batch of {n}"
    assert np.array_equal(agent.q, q) and not agent.visits.any()  # choosing actions learns nothing


def test_act_epsilon_consumes_the_same_stream_whatever_epsilon():
    obs = ALL_STATES[np.random.default_rng(0).integers(0, N_STATES, size=64)]
    states = []
    for epsilon in (0.0, 0.05, 0.5, 1.0):
        agent = table_agent(3, seed=30)
        rng = np.random.default_rng(1234)
        for _ in range(3):
            agent.act_epsilon(obs, epsilon, rng)
        states.append((rng.bit_generator.state, rng.random(4).tolist()))
    assert all(state == states[0] for state in states[1:])
    # ... and that stream did move: both draws happened even with epsilon = 0.
    untouched = np.random.default_rng(1234)
    assert states[0][0] != untouched.bit_generator.state


# ------------------------------------------------------------- learning sanity (M4 acceptance)
#: 8x8, reward R0, relative actions. With the default starve limit (100) a
#: run of this length never truncates at gamma = 0.95, so the truncation rule
#: would go unexercised; H = 30 gives about a hundred truncations.
LEARN_ENV = EnvConfig(W=8, starve_limit=30, reward=R0)
LEARN_AGENT = TabularQConfig(alpha=0.1, gamma=0.95, eps_start=1.0, eps_end=0.05, eps_decay_fraction=0.2)
LEARN_N_ENVS = 64
LEARN_ENV_STEPS = 200_000
#: Run seed 0 in the convention of the training entry point: environment base
#: seed 100_000 * (run_seed + 1), exploration stream default_rng([run_seed, 1]).
LEARN_ENV_SEED = 100_000
LEARN_EXPLORE_SEED = (0, 1)
#: Measured with a cell-by-cell reference agent: random 0.17, learned 13.4 for
#: this seed and 13.3 .. 16.2 over eight run seeds (78 .. 95 times random).
#: The floor keeps 40 % slack under the worst of them and is still 47 times
#: the random player.
LEARN_MIN_SCORE = 8.0
LEARN_MIN_RATIO = 25.0
#: CPU seconds for the 3125 vec-steps of training, of which the environment
#: takes about 0.8 s. Even a per-transition Python loop stays near 1.5 s in
#: total; the budget only rejects an agent whose update scales with the table.
LEARN_CPU_BUDGET_S = 20.0


def training_steps(
    agent: TabularQAgent, cfg: EnvConfig, *, n_envs: int, total_env_steps: int, env_seed: int, explore_seed
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """The tabular training loop of the M4 spec, written with the agent API
    only. Yields ``(terminated flags passed to update, truncated flags)``
    after every vec-step."""
    env = VecSnakeEnv(cfg, n_envs=n_envs, seed=env_seed)
    rng = np.random.default_rng(explore_seed)
    acfg = agent.cfg
    obs = env.reset()
    for k in range(total_env_steps // n_envs):
        epsilon = linear_schedule(
            k * n_envs, acfg.eps_start, acfg.eps_end, acfg.eps_decay_fraction * total_env_steps)
        actions = agent.act_epsilon(obs, epsilon, rng)
        next_obs, reward, terminated, truncated, info = env.step(actions)
        agent.update(obs, actions, reward, info["final_obs"], terminated)  # true successor, terminated only
        obs = next_obs
        yield terminated, truncated


def test_learns_to_beat_the_random_player_on_8x8():
    agent = TabularQAgent(3, LEARN_AGENT)
    n_terminated = n_truncated = truncated_passed_as_terminated = 0
    t0 = time.process_time()
    for passed_terminated, truncated in training_steps(
        agent, LEARN_ENV, n_envs=LEARN_N_ENVS, total_env_steps=LEARN_ENV_STEPS,
        env_seed=LEARN_ENV_SEED, explore_seed=LEARN_EXPLORE_SEED,
    ):
        n_terminated += int(passed_terminated.sum())
        n_truncated += int(truncated.sum())
        truncated_passed_as_terminated += int((truncated & passed_terminated).sum())
    train_cpu_s = time.process_time() - t0

    # truncated != terminated, structurally: truncations did happen and none
    # of them reached update() as a terminal transition.
    assert n_truncated >= 30 and n_terminated >= 1000, (n_truncated, n_terminated)
    assert truncated_passed_as_terminated == 0

    # The table: one visit per transition, only rows a snake can act on, and
    # values inside the range of discounted returns of rewards in [-1, 1].
    assert agent.visits.sum() == LEARN_ENV_STEPS
    visited = np.flatnonzero(agent.visits.sum(axis=1))
    assert set(visited.tolist()) <= live_states() and 200 <= len(visited) <= 256
    assert np.isfinite(agent.q).all() and np.abs(agent.q).max() <= 1.0 / (1.0 - LEARN_AGENT.gamma)
    assert (agent.q[agent.visits == 0] == 0.0).all()

    # Acceptance: 100 greedy episodes on the standard evaluation boards.
    learned = evaluate(agent, LEARN_ENV, n_episodes=100, seed=DEFAULT_EVAL_SEED)
    floor = evaluate(RandomPolicy(seed=DEFAULT_EVAL_SEED), LEARN_ENV, n_episodes=100, seed=DEFAULT_EVAL_SEED)
    assert learned.policy == "tabular_q" and len(learned.episodes) == 100
    assert 0.0 < floor.mean_score < 1.0, floor.mean_score
    assert learned.mean_score >= LEARN_MIN_SCORE, (learned.mean_score, floor.mean_score)
    assert learned.mean_score >= LEARN_MIN_RATIO * floor.mean_score, (learned.mean_score, floor.mean_score)
    assert learned.mean_score - learned.ci95 > floor.mean_score + floor.ci95

    assert train_cpu_s < LEARN_CPU_BUDGET_S, f"training took {train_cpu_s:.1f} CPU seconds"


# ------------------------------------------------------------- save / load
def awkward_agent(n_actions: int = 4) -> TabularQAgent:
    """Agent whose table and config survive only an exact round trip: doubles
    that float32 cannot hold, a signed zero, a subnormal, counts beyond int32."""
    cfg = TabularQConfig(
        alpha=0.123456789, gamma=1.0 / 3.0, eps_start=0.9, eps_end=0.01, eps_decay_fraction=0.35, q_init=-0.5)
    agent = TabularQAgent(n_actions, cfg)
    rng = np.random.default_rng(7)
    agent.q[:] = rng.normal(size=agent.q.shape) * 1e3
    agent.q[0, 0], agent.q[1, 1], agent.q[2, 2] = np.nextafter(1.0, 2.0), -0.0, 5e-324
    agent.visits[:] = rng.integers(0, 2**40, size=agent.visits.shape)
    return agent


def test_save_load_round_trip_is_exact(tmp_path):
    agent = awkward_agent()
    q, visits = agent.q.copy(), agent.visits.copy()
    path = tmp_path / "q_table.npz"
    out = agent.save(path)
    assert isinstance(out, Path) and out.resolve() == path.resolve() and path.is_file()
    assert [p.name for p in tmp_path.iterdir()] == ["q_table.npz"]  # nothing else was written
    assert agent.q.tobytes() == q.tobytes() and np.array_equal(agent.visits, visits)  # saving changes nothing

    loaded = TabularQAgent.load(path)
    assert isinstance(loaded, TabularQAgent) and loaded is not agent
    assert loaded.n_actions == 4 and loaded.name == "tabular_q"
    assert isinstance(loaded.cfg, TabularQConfig) and loaded.cfg == agent.cfg
    assert loaded.q.dtype == np.float64 and loaded.q.shape == (2048, 4)
    assert loaded.q.tobytes() == q.tobytes()  # bit for bit, signed zero and subnormal included
    assert loaded.visits.dtype == np.int64 and loaded.visits.shape == (2048, 4)
    assert np.array_equal(loaded.visits, visits)
    assert np.array_equal(loaded.act_batch(ALL_STATES), agent.act_batch(ALL_STATES))

    # The loaded agent is a live agent: the same update gives the same table.
    batch = [(S_A, 0, 1.0, S_B, False), (S_C, 3, -1.0, S_D, True), (S_A, 0, 0.5, S_C, False)]
    assert apply(loaded, batch) == apply(agent, batch)
    assert loaded.q.tobytes() == agent.q.tobytes() and np.array_equal(loaded.visits, agent.visits)
    assert TabularQAgent.load(path).q.tobytes() == q.tobytes()  # the file itself is unchanged by all that


def test_save_load_default_agent_and_str_path(tmp_path):
    agent = TabularQAgent(3)
    apply(agent, [(S_A, 1, 1.0, S_B, True)])
    path = str(tmp_path / "three_actions.npz")
    out = agent.save(path)
    assert isinstance(out, Path) and out.resolve() == Path(path).resolve() and out.is_file()
    loaded = TabularQAgent.load(path)
    assert loaded.n_actions == 3 and loaded.cfg == TabularQConfig()
    assert loaded.q.shape == (2048, 3) and np.array_equal(loaded.q, agent.q)
    assert loaded.visits.sum() == 1 and loaded.visits[S_A, 1] == 1


def test_npz_file_layout(tmp_path):
    """A plain numpy archive that a notebook can open without the agent class
    (and without pickle): ``q``, ``visits``, ``n_actions`` and one entry per
    config field."""
    agent = awkward_agent()
    path = agent.save(tmp_path / "q_table.npz")
    with np.load(path) as data:  # allow_pickle is False by default
        assert {"q", "visits", "n_actions"} | {f.name for f in fields(TabularQConfig)} <= set(data.files)
        assert data["q"].dtype == np.float64 and data["q"].tobytes() == agent.q.tobytes()
        assert data["visits"].dtype == np.int64 and np.array_equal(data["visits"], agent.visits)
        assert int(data["n_actions"]) == 4
        for f in fields(TabularQConfig):
            assert float(data[f.name]) == getattr(agent.cfg, f.name), f.name


def test_load_does_not_depend_on_the_working_directory(tmp_path, monkeypatch):
    here, elsewhere = tmp_path / "here", tmp_path / "elsewhere"
    here.mkdir()
    elsewhere.mkdir()
    agent = awkward_agent(3)

    monkeypatch.chdir(here)
    agent.save("relative.npz")  # relative to the working directory at the time of the call
    assert (here / "relative.npz").is_file()
    absolute = agent.save(tmp_path / "absolute.npz")

    monkeypatch.chdir(elsewhere)
    for path in (here / "relative.npz", absolute):
        loaded = TabularQAgent.load(path)
        assert loaded.cfg == agent.cfg and loaded.n_actions == 3
        assert loaded.q.tobytes() == agent.q.tobytes() and np.array_equal(loaded.visits, agent.visits)
    assert list(elsewhere.iterdir()) == []  # loading writes nothing


# ------------------------------------------------------------- determinism
DETERMINISM_ENV = EnvConfig(W=8, starve_limit=20, reward=RewardConfig(variant="R2", gamma=0.99))


def driven_agent(env_seed: int, explore_seed) -> Iterator[TabularQAgent]:
    """Fresh agent driven for 300 vec-steps of 16 boards; yields itself after every vec-step."""
    agent = TabularQAgent(3)
    for _ in training_steps(
        agent, DETERMINISM_ENV, n_envs=16, total_env_steps=4800, env_seed=env_seed, explore_seed=explore_seed
    ):
        yield agent


def run_to_end(steps: Iterator[TabularQAgent]) -> TabularQAgent:
    agent = None
    for agent in steps:
        pass
    assert agent is not None
    return agent


def test_same_seeds_give_identical_tables():
    alone = run_to_end(driven_agent(env_seed=700, explore_seed=(3, 1)))
    assert alone.visits.sum() == 4800 and (alone.q != 0.0).sum() > 100

    # Two more agents with the same seeds, their steps interleaved: equal to
    # the first one, and no state leaks from one agent into another.
    first, second = driven_agent(700, (3, 1)), driven_agent(700, (3, 1))
    for a, b in zip(first, second):
        pass
    assert a is not b
    for other in (a, b):
        assert other.q.tobytes() == alone.q.tobytes()
        assert np.array_equal(other.visits, alone.visits)


def test_different_seeds_give_different_tables():
    base = run_to_end(driven_agent(env_seed=700, explore_seed=(3, 1)))
    other_exploration = run_to_end(driven_agent(env_seed=700, explore_seed=(4, 1)))
    other_boards = run_to_end(driven_agent(env_seed=900, explore_seed=(3, 1)))
    for other in (other_exploration, other_boards):
        assert other.visits.sum() == base.visits.sum() == 4800
        assert not np.array_equal(other.q, base.q)
        assert not np.array_equal(other.visits, base.visits)


# ------------------------------------------------------------- input checks
# Added at integration. The M4 spec does not say what happens on malformed
# input, so this section (unlike the rest of the file) pins what the
# implementation chose: it refuses. The alternative is silent damage to the
# table - numpy would wrap action -1 to the last column, truncate a float
# action and broadcast one reward over the whole batch.
def good_batch() -> dict[str, np.ndarray]:
    return {
        "obs": obs_of(S_A, S_C, S_A),
        "actions": np.array([0, 1, 2], dtype=np.int64),
        "rewards": np.array([1.0, -1.0, 0.5], dtype=np.float32),
        "next_obs": obs_of(S_B, S_D, S_B),
        "terminated": np.array([False, True, False]),
    }


BAD_BATCHES = {
    "empty batch": {
        "obs": np.zeros((0, 11), dtype=np.float32), "actions": np.zeros(0, dtype=np.int64),
        "rewards": np.zeros(0, dtype=np.float32), "next_obs": np.zeros((0, 11), dtype=np.float32),
        "terminated": np.zeros(0, dtype=bool),
    },
    "action below zero": {"actions": np.array([0, -1, 2])},
    "action equal to n_actions": {"actions": np.array([0, 3, 2])},
    "float actions": {"actions": np.array([0.0, 1.0, 2.0])},
    "actions with a second axis": {"actions": np.array([[0], [1], [2]])},
    "one reward for three transitions": {"rewards": np.array([1.0], dtype=np.float32)},
    "one reward and one flag for three transitions": {
        "rewards": np.array([1.0], dtype=np.float32), "terminated": np.array([False]),
    },
    "two flags for three transitions": {"terminated": np.array([False, True])},
    "two successors for three transitions": {"next_obs": obs_of(S_B, S_D)},
    "one observation for three transitions": {"obs": obs_of(S_A)},
    "observations with ten features": {"obs": np.zeros((3, 10), dtype=np.float32)},
}


@pytest.mark.parametrize("name", list(BAD_BATCHES))
def test_update_refuses_a_malformed_batch_and_leaves_the_table_alone(name):
    agent = TabularQAgent(3)
    agent.q[:] = np.random.default_rng(0).normal(size=agent.q.shape)
    apply(agent, [(S_A, 0, 1.0, S_B, False)])
    q, visits = agent.q.copy(), agent.visits.copy()
    with pytest.raises(ValueError):
        agent.update(**{**good_batch(), **BAD_BATCHES[name]})
    assert agent.q.tobytes() == q.tobytes() and np.array_equal(agent.visits, visits)
    # ... while the well-formed batch it was derived from goes through.
    assert type(agent.update(**good_batch())) is float and agent.visits.sum() == 4


def test_act_takes_one_observation_and_act_batch_takes_a_batch():
    agent = table_agent(3, seed=40)
    with pytest.raises(ValueError):
        agent.act(obs_of(S_A, S_B))  # a batch: which of the two actions would it be?
    with pytest.raises(ValueError):
        agent.act_batch(ALL_STATES[S_A])  # one observation without the batch axis
    with pytest.raises(ValueError):
        agent.act(np.zeros(10, dtype=np.float32))
    with pytest.raises(ValueError):
        agent.act_epsilon(ALL_STATES[S_A], 0.5, np.random.default_rng(0))


@pytest.mark.parametrize("n_actions", [0, -1, 3.0, "3", True])
def test_n_actions_must_be_a_positive_integer(n_actions):
    with pytest.raises(ValueError):
        TabularQAgent(n_actions)


def test_n_actions_may_be_a_numpy_integer():
    agent = TabularQAgent(np.int64(4))
    assert agent.n_actions == 4 and type(agent.n_actions) is int and agent.q.shape == (2048, 4)


def test_load_refuses_tables_that_do_not_match_n_actions(tmp_path):
    good = awkward_agent(4).save(tmp_path / "good.npz")
    with np.load(good) as data:
        arrays = {name: data[name] for name in data.files}
    for name, change in {
        "three_actions_declared": {"n_actions": np.int64(3)},
        "visits_of_another_shape": {"visits": arrays["visits"][:, :3]},
        "half_a_table": {"q": arrays["q"][:1024], "visits": arrays["visits"][:1024]},
    }.items():
        path = tmp_path / f"{name}.npz"
        np.savez(path, **{**arrays, **change})
        with pytest.raises(ValueError):
            TabularQAgent.load(path)
    assert TabularQAgent.load(good).q.tobytes() == arrays["q"].tobytes()


def test_save_creates_parent_directories_and_writes_exactly_the_given_name(tmp_path):
    agent = awkward_agent(3)
    nested = agent.save(tmp_path / "runs" / "exp" / "0" / "q_table.npz")
    assert nested == tmp_path / "runs" / "exp" / "0" / "q_table.npz" and nested.is_file()
    assert TabularQAgent.load(nested).q.tobytes() == agent.q.tobytes()
    bare = agent.save(tmp_path / "table")  # numpy alone would write "table.npz"
    assert bare == tmp_path / "table" and bare.is_file() and not (tmp_path / "table.npz").exists()
    assert TabularQAgent.load(bare).q.tobytes() == agent.q.tobytes()

"""Vectorised greedy evaluation: ``evaluate_batched`` against the scalar
protocol ``evaluate`` (the reference; nothing here re-implements the rules).

Contract under test. ``evaluate_batched(policy, cfg, n, seed)`` steps one
``VecSnakeEnv(cfg, n_envs=n, seed=seed)`` and records, for every sub-env
``i``, only its **first** episode, labelled ``seed + i`` and listed in the
order of ``i``. Sub-env ``i`` owns ``default_rng(seed + i)``, so for a
deterministic policy that reads nothing but the observation the episode
records equal those of ``evaluate(policy, cfg, n, seed)`` one by one, and so
do the mean score, its confidence interval and the cause breakdown.

Sub-envs whose first episode is over keep auto-resetting while the slowest
one is still playing. Recording those later episodes ("the first ``n``
episodes to finish") would count short episodes several times and long ones
never; ``test_first_episode_of_each_sub_env_not_the_first_k_to_finish``
computes that forbidden selection for contrast.

Players: (a) a fixed random lookup table over the 2048 states of observation
A, in relative and absolute action mode, (b) a food-seeking rule on
observation A, (c) the same rule read from the planes of observation B,
(d) always straight, plus a Hamiltonian-cycle follower on observation B for
the wins. All are deterministic, stateless functions of the observation with
``act`` defined as ``act_batch`` on a batch of one, so the scalar and the
batched evaluation query the same function (``test_act_and_act_batch_agree``
checks that ``act_batch`` treats rows independently). They do not depend on
``snake_rl.agents.tabular_q``.

Layout: players, the equivalence matrix with its coverage gate (plus a
sampled reference for the four cells whose full scalar reference is too
slow), selection of the first episode, record contents and metadata, API.
"""

from __future__ import annotations

import functools
import inspect
import json
from collections import Counter
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pytest

import snake_rl.evaluate as evaluate_module
import snake_rl.utils
from snake_rl.agents import Policy
from snake_rl.agents.baselines import hamiltonian_cycle
from snake_rl.env.common import CAUSE_NAMES, DELTAS, RELATIVE_TURN, n_actions
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
from snake_rl.evaluate import (
    DEFAULT_EVAL_SEED,
    EpisodeRecord,
    EvalResult,
    env_config_from_dict,
    env_config_to_dict,
    evaluate,
    evaluate_batched,
)
from tests.helpers import STRAIGHT, TURN_LEFT, TURN_RIGHT, EnvConfig, ObsConfig, RewardConfig, SnakeEnv

OBS_A = ObsConfig(kind="A")
OBS_B = ObsConfig(kind="B")
#: Plane indices of observation B (8 planes).
HEAD_PLANE, BODY_PLANE, FOOD_PLANE, HEADING_PLANE0 = 0, 1, 2, 4
N_STATES_A = 2**N_FEATURES_A
#: Bit weights of the 11 features: the index of an observation A in a table.
BIT_WEIGHTS = 2 ** np.arange(N_FEATURES_A)
END_CAUSES = CAUSE_NAMES[1:]  # wall, self, starve, win

#: ``git_hash`` as ``evaluate.py`` imported it, before the fixture below replaces it.
REAL_GIT_HASH = evaluate_module.git_hash
FAKE_GIT = "0000000+test"


@pytest.fixture(autouse=True)
def _fast_git_hash(monkeypatch):
    """``git_hash()`` spawns two git processes (about 50 ms) and every
    evaluation calls it once; a constant keeps this file fast and makes the
    ``git`` field of the scalar and the batched result comparable."""
    monkeypatch.setattr(evaluate_module, "git_hash", lambda: FAKE_GIT)


# ------------------------------------------------------------- players
class ObsPolicy:
    """Deterministic observation-only player. ``act`` is ``act_batch`` on a
    batch of one: the two agree by construction."""

    name = "obs-policy"

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, env: SnakeEnv | None = None) -> int:
        return int(self.act_batch(obs[None])[0])

    def act_batch(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class TablePolicy(ObsPolicy):
    """(a) A fixed random action for each of the 2048 states of observation A:
    what a tabular agent looks like to the evaluator."""

    def __init__(self, n_actions: int, seed: int) -> None:
        self.name = f"table{n_actions}"
        self.table = np.random.default_rng(seed).integers(0, n_actions, size=N_STATES_A)

    def act_batch(self, obs: np.ndarray) -> np.ndarray:
        return self.table[obs.astype(np.int64) @ BIT_WEIGHTS]


class FoodSeekerA(ObsPolicy):
    """(b) Rule on observation A: turn towards the food unless the danger bit
    forbids it. Preference: safe and towards the food > safe > unsafe; ties go
    to straight, then left, then right. Lives long and eats."""

    def __init__(self, action_mode: str = "relative") -> None:
        self.name = f"seeker-{action_mode}"
        self.action_mode = action_mode

    def act_batch(self, obs: np.ndarray) -> np.ndarray:
        heading = obs[:, A_HEADING].argmax(axis=1)
        food_towards = obs[:, [A_FOOD_UP, A_FOOD_RIGHT, A_FOOD_DOWN, A_FOOD_LEFT]]  # indexed by heading id
        turn_heading = (heading[:, None] + RELATIVE_TURN) % 4  # (N, 3): straight, left, right
        towards = np.take_along_axis(food_towards, turn_heading, axis=1)
        danger = obs[:, [A_DANGER_STRAIGHT, A_DANGER_LEFT, A_DANGER_RIGHT]]
        turn = (towards - 2.0 * danger).argmax(axis=1)
        if self.action_mode == "relative":
            return turn
        return turn_heading[np.arange(len(obs)), turn]  # absolute mode: the action is the heading


class PlaneSeekerB(ObsPolicy):
    """(c) The same idea read from the planes of observation B: head, food and
    heading locate the three candidate cells, the body plane says which are
    blocked (the tail cell counts as blocked), Manhattan distance says which
    come closer to the food. Relative actions."""

    name = "planes"

    def act_batch(self, obs: np.ndarray) -> np.ndarray:
        n, _, W, _ = obs.shape
        head = np.stack(np.divmod(obs[:, HEAD_PLANE].reshape(n, -1).argmax(axis=1), W), axis=1)
        food = np.stack(np.divmod(obs[:, FOOD_PLANE].reshape(n, -1).argmax(axis=1), W), axis=1)
        heading = obs[:, HEADING_PLANE0 : HEADING_PLANE0 + 4, 0, 0].argmax(axis=1)
        target = head[:, None, :] + DELTAS[(heading[:, None] + RELATIVE_TURN) % 4]  # (N, 3, 2)
        r, c = target[..., 0], target[..., 1]
        off_grid = (r < 0) | (r >= W) | (c < 0) | (c >= W)
        body = obs[:, BODY_PLANE][np.arange(n)[:, None], r.clip(0, W - 1), c.clip(0, W - 1)] > 0
        closer = np.abs(target - food[:, None, :]).sum(axis=2) < np.abs(head - food).sum(axis=1)[:, None]
        return (closer - 2.0 * (off_grid | body)).argmax(axis=1)


@functools.lru_cache(maxsize=None)
def _cycle_tables(W: int) -> tuple[np.ndarray, np.ndarray]:
    """Hamiltonian cycle cells ``(W*W, 2)`` and the position of each flat cell on it."""
    cycle = hamiltonian_cycle(W)
    position = np.empty(W * W, dtype=np.int64)
    position[cycle[:, 0] * W + cycle[:, 1]] = np.arange(W * W)
    return cycle, position


class CycleFollowerB(ObsPolicy):
    """Follows the Hamiltonian cycle of the baselines using only the head and
    heading planes of observation B (backwards if the cycle would run into the
    neck). Wins unless a small starve limit cuts it short: the source of the
    ``win`` episodes, and of very long episodes next to very short ones.
    Relative actions."""

    name = "cycle"

    def act_batch(self, obs: np.ndarray) -> np.ndarray:
        n, _, W, _ = obs.shape
        cycle, position = _cycle_tables(W)
        head_flat = obs[:, HEAD_PLANE].reshape(n, -1).argmax(axis=1)
        head = np.stack(np.divmod(head_flat, W), axis=1)
        heading = obs[:, HEADING_PLANE0 : HEADING_PLANE0 + 4, 0, 0].argmax(axis=1)
        i = position[head_flat]
        target = cycle[(i + 1) % (W * W)]
        into_neck = (target == head - DELTAS[heading]).all(axis=1)
        target[into_neck] = cycle[(i[into_neck] - 1) % (W * W)]
        desired = (DELTAS[None] == (target - head)[:, None]).all(axis=2).argmax(axis=1)
        # (desired - heading) % 4: 0 straight, 1 right, 3 left; 2 (reversal) cannot occur
        return np.array([STRAIGHT, TURN_RIGHT, STRAIGHT, TURN_LEFT])[(desired - heading) % 4]


class AlwaysStraight(ObsPolicy):
    """(d) Action 0 whatever the observation: with the default starve limit
    every episode ends on the wall after ``W - W // 2`` steps."""

    name = "straight"

    def act_batch(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(len(obs), dtype=np.int64)


@dataclass(frozen=True)
class Player:
    """How to build a policy and the environment settings it is written for."""

    make: Callable[[], ObsPolicy]
    action_mode: str = "relative"
    obs: ObsConfig = OBS_A


PLAYERS: dict[str, Player] = {
    "table-rel": Player(lambda: TablePolicy(3, seed=7)),
    "table-abs": Player(lambda: TablePolicy(4, seed=16), action_mode="absolute"),
    "seeker": Player(lambda: FoodSeekerA("relative")),
    "seeker-abs": Player(lambda: FoodSeekerA("absolute"), action_mode="absolute"),
    "planes": Player(PlaneSeekerB, obs=OBS_B),
    "cycle": Player(CycleFollowerB, obs=OBS_B),
    "straight": Player(AlwaysStraight),
}


def _all_states_A() -> np.ndarray:
    """Every binary vector of length 11, row ``k`` being the bits of ``k``."""
    return ((np.arange(N_STATES_A)[:, None] >> np.arange(N_FEATURES_A)) & 1).astype(np.float32)


def _rollout_observations(player: Player, W: int = 6, n_envs: int = 12, n_steps: int = 60) -> np.ndarray:
    """Observations met by the player itself on ``n_envs`` boards."""
    cfg = EnvConfig(W=W, action_mode=player.action_mode, obs=player.obs)
    policy = player.make()
    env = VecSnakeEnv(cfg, n_envs=n_envs, seed=9)
    obs = env.reset()
    seen = [obs]
    for _ in range(n_steps):
        obs = env.step(policy.act_batch(obs))[0]
        seen.append(obs)
    return np.concatenate(seen)


@pytest.mark.parametrize("name", list(PLAYERS))
def test_act_and_act_batch_agree(name):
    """Both evaluators see the same policy: ``act_batch`` on a batch equals
    ``act`` on each row, in any row order, with valid integer actions."""
    player = PLAYERS[name]
    policy = player.make()
    assert isinstance(policy, Policy) and isinstance(policy.name, str)
    obs = _all_states_A() if player.obs.kind == "A" else _rollout_observations(player)
    assert len(obs) >= 500

    batch = policy.act_batch(obs)
    assert batch.shape == (len(obs),) and np.issubdtype(batch.dtype, np.integer)
    assert batch.min() >= 0 and batch.max() < n_actions(player.action_mode)
    singles = [policy.act(o, None) for o in obs]
    assert all(type(a) is int for a in singles)
    assert batch.tolist() == singles
    perm = np.random.default_rng(0).permutation(len(obs))
    assert np.array_equal(policy.act_batch(obs[perm]), batch[perm])
    assert np.array_equal(player.make().act_batch(obs), batch)  # a fresh instance is the same policy
    assert len(set(batch.tolist())) > 1 or name == "straight"


def test_table_policy_is_a_lookup_over_the_2048_states():
    states = _all_states_A()
    assert np.array_equal(states.astype(np.int64) @ BIT_WEIGHTS, np.arange(N_STATES_A))
    for n_act in (3, 4):
        policy = TablePolicy(n_act, seed=5)
        assert np.array_equal(policy.act_batch(states), policy.table)
        assert set(policy.table.tolist()) == set(range(n_act))


# ------------------------------------------------------------- equivalence matrix
@dataclass(frozen=True)
class Case:
    """One ``(policy, cfg, n, seed)`` on which the two evaluators are compared."""

    player: str
    W: int
    starve_limit: int | None
    n: int
    seed: int

    @property
    def cfg(self) -> EnvConfig:
        p = PLAYERS[self.player]
        return EnvConfig(W=self.W, action_mode=p.action_mode, starve_limit=self.starve_limit, obs=p.obs)

    @property
    def id(self) -> str:
        return f"{self.player}-W{self.W}-H{self.starve_limit or 'default'}-n{self.n}"


#: The four policy families of the contract, (a) in both action modes.
MATRIX_PLAYERS = ("table-rel", "table-abs", "seeker", "planes", "straight")
WIDTHS = (6, 8, 10)
#: Tiny starve limit: most episodes are truncated, on 10x10 even the
#: always-straight player (4 moves without food, one step before the wall).
SMALL_H = 4
STARVE_LIMITS = (None, SMALL_H)  # None: the default, H = 100 on these boards
BATCH_SIZES = (1, 7, 100)
#: 100 default-starve episodes of a long-lived player cost 0.6 to 1.5 s of
#: *scalar* reference depending on the board, so that corner of the matrix
#: (2 players x 3 boards) gets the full reference on the smallest board only;
#: the other four cells are compared with a sampled reference (SPOT_CHECKED).
LONG_LIVED_PLAYERS = ("seeker", "planes")
LONG_LIVED_N100_W = 6


def in_matrix(player: str, W: int, starve_limit: int | None, n: int) -> bool:
    """The full product players x boards x starve limits x batch sizes, minus
    the four expensive cells described above."""
    skipped = n == 100 and starve_limit is None and player in LONG_LIVED_PLAYERS and W != LONG_LIVED_N100_W
    return not skipped


def matrix_case(player: str, W: int, starve_limit: int | None, n: int) -> Case:
    """Matrix member; the seed depends on everything but ``n``, so the three
    batch sizes of a cell see the same leading boards."""
    seed = 1000 * (MATRIX_PLAYERS.index(player) + 1) + 10 * W + (starve_limit or 0)
    return Case(player, W, starve_limit, n, seed)


def _matrix_cases() -> list[Case]:
    cases = []
    for player in MATRIX_PLAYERS:
        for W in WIDTHS:
            for H in STARVE_LIMITS:
                for n in BATCH_SIZES:
                    if in_matrix(player, W, H, n):
                        cases.append(matrix_case(player, W, H, n))
    return cases


#: Long (win, about 300 steps) and short (starve) episodes in one batch.
CYCLE_MIXED = Case("cycle", 6, 30, 16, seed=7042)
#: Self collisions after 18 to 160 steps.
SEEKER_LONG = matrix_case("seeker", LONG_LIVED_N100_W, None, 100)
EXTRA_CASES = (
    Case("cycle", 4, None, 30, seed=7000),  # every episode is a win
    Case("cycle", 6, None, 3, seed=7010),  # wins on a board of the matrix
    CYCLE_MIXED,
    Case("cycle", 6, SMALL_H, 100, seed=7050),  # starvation only
    Case("cycle", 8, SMALL_H, 7, seed=7060),
    Case("cycle", 10, SMALL_H, 100, seed=7070),
    Case("seeker-abs", 6, SMALL_H, 100, seed=8000),  # absolute actions with eating
    Case("seeker-abs", 8, None, 7, seed=8010),
)
CASES: tuple[Case, ...] = (*_matrix_cases(), *EXTRA_CASES)


@functools.lru_cache(maxsize=None)
def run_case(case: Case) -> tuple[EvalResult, EvalResult]:
    """``(scalar, batched)`` results of a case, computed once per session and
    always from inside a test (the coverage gate reuses them)."""
    player = PLAYERS[case.player]
    scalar = evaluate(player.make(), case.cfg, case.n, case.seed)
    batched = evaluate_batched(player.make(), case.cfg, case.n, case.seed)
    return scalar, batched


def assert_same_episodes(got: EvalResult, want: EvalResult) -> None:
    """Record by record, including the Python types (``np.int64(3) == 3``
    would hide a numpy leak that breaks ``json.dumps``)."""
    assert len(got.episodes) == len(want.episodes)
    for i, (g, w) in enumerate(zip(got.episodes, want.episodes)):
        assert isinstance(g, EpisodeRecord)
        assert g == w, f"episode {i}: batched {g} != scalar {w}"
        for name in ("seed", "score", "steps", "length"):
            assert type(getattr(g, name)) is int, f"episode {i}: {name} is a {type(getattr(g, name)).__name__}"
        assert type(g.cause) is str, f"episode {i}: cause is a {type(g.cause).__name__}"


@pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
def test_batched_equals_scalar(case):
    scalar, batched = run_case(case)
    assert isinstance(batched, EvalResult)
    assert len(scalar.episodes) == len(batched.episodes) == case.n
    assert_same_episodes(batched, scalar)
    # Hence the same table row: mean, CI (nan for n = 1), cause breakdown, median, max, steps.
    assert batched.mean_score == scalar.mean_score
    np.testing.assert_equal(batched.ci95, scalar.ci95)
    assert batched.cause_fractions == scalar.cause_fractions
    np.testing.assert_equal(batched.summary(), scalar.summary())
    got, want = batched.to_dict(), scalar.to_dict()
    assert got.pop("wall_time_s") > 0.0 and want.pop("wall_time_s") > 0.0
    np.testing.assert_equal(got, want)  # policy, env, seed, git, summary, episodes


def test_matrix_spans_every_axis():
    """Guards the case table itself: dropping a combination must be deliberate."""
    for player in MATRIX_PLAYERS:
        cells = {(c.W, c.starve_limit, c.n) for c in CASES if c.player == player}
        for W in WIDTHS:
            for H in STARVE_LIMITS:
                have = {n for n in BATCH_SIZES if (W, H, n) in cells}
                if H is None and player in LONG_LIVED_PLAYERS and W != LONG_LIVED_N100_W:
                    assert have == {1, 7}, (player, W, H)  # the only cells left out (SPOT_CHECKED)
                else:
                    assert have == set(BATCH_SIZES), (player, W, H)
    assert {PLAYERS[c.player].action_mode for c in CASES} == {"relative", "absolute"}
    assert {PLAYERS[c.player].obs for c in CASES} == {OBS_A, OBS_B}
    assert {c.player for c in CASES} == set(PLAYERS)
    assert len(set(CASES)) == len(CASES) and len({c.id for c in CASES}) == len(CASES)
    for W in WIDTHS:
        assert EnvConfig(W=W).H == 100 > SMALL_H


def test_matrix_runs_cover_every_end_cause():
    """Equal results are only evidence if the runs reached every way an
    episode can end, long episodes and many eats (counted on the scalar
    reference side)."""
    causes: Counter[str] = Counter()
    eats = longest = 0
    for case in CASES:
        scalar, _ = run_case(case)
        causes.update(e.cause for e in scalar.episodes)
        eats += sum(e.score for e in scalar.episodes)
        longest = max(longest, max(e.steps for e in scalar.episodes))
    assert set(causes) == set(END_CAUSES), causes
    assert causes["wall"] >= 1000, causes
    assert causes["self"] >= 200, causes
    assert causes["starve"] >= 1000, causes
    assert causes["win"] >= 40, causes
    assert eats >= 5000 and longest >= 300, (eats, longest)


def test_episode_i_does_not_depend_on_the_batch_size():
    """Sub-env ``i`` owns ``default_rng(seed + i)``: the 1- and 7-episode
    evaluations are prefixes of the 100-episode one."""
    groups: dict[tuple[str, int, int | None, int], dict[int, EvalResult]] = {}
    for case in CASES:
        groups.setdefault((case.player, case.W, case.starve_limit, case.seed), {})[case.n] = run_case(case)[1]
    full = [g for g in groups.values() if set(g) == set(BATCH_SIZES)]
    assert len(full) >= 20
    for group in full:
        assert group[1].episodes == group[100].episodes[:1]
        assert group[7].episodes == group[100].episodes[:7]


#: The four cells left out of the matrix: 100 default-starve episodes of a
#: long-lived player on 8x8 and 10x10.
SPOT_CHECKED = tuple(
    matrix_case(player, W, None, 100) for player in LONG_LIVED_PLAYERS for W in WIDTHS if W != LONG_LIVED_N100_W
)


def test_scalar_episode_i_is_the_single_evaluation_seeded_seed_plus_i():
    """Property of the scalar protocol that the sampled reference below relies
    on: every episode starts from ``reset(seed=seed + i)``, whatever came before."""
    scalar, _ = run_case(SEEKER_LONG)
    for i in (3, 57):
        single = evaluate(PLAYERS[SEEKER_LONG.player].make(), SEEKER_LONG.cfg, 1, SEEKER_LONG.seed + i)
        assert single.episodes == [scalar.episodes[i]]


@pytest.mark.parametrize("case", SPOT_CHECKED, ids=lambda case: case.id)
def test_left_out_cells_against_a_sampled_scalar_reference(case):
    """All 100 boards are evaluated in one batch; the scalar reference is
    replayed for the two shortest, the median and the two longest episodes,
    and the first seven records are those of the 7-episode matrix case of the
    same cell."""
    assert case not in CASES and len(SPOT_CHECKED) == 4
    player = PLAYERS[case.player]
    batched = evaluate_batched(player.make(), case.cfg, case.n, case.seed)
    assert [e.seed for e in batched.episodes] == list(range(case.seed, case.seed + 100))
    by_length = np.argsort(batched.steps, kind="stable").tolist()
    for i in (*by_length[:2], by_length[50], *by_length[-2:]):
        single = evaluate(player.make(), case.cfg, 1, case.seed + i)
        assert single.episodes == [batched.episodes[i]], f"sub-env {i}"
    scalar_7, batched_7 = run_case(matrix_case(case.player, case.W, None, 7))
    assert batched.episodes[:7] == batched_7.episodes == scalar_7.episodes
    assert batched.steps.max() >= 4 * batched.steps.min()  # short and long episodes in one batch


# ------------------------------------------------------------- first episode only
def finished_in_order(policy: ObsPolicy, cfg: EnvConfig, n_envs: int, seed: int) -> list[tuple[int, EpisodeRecord]]:
    """Every episode that ends, as ``(sub-env, record)`` in the order of
    finishing (vec step, then sub-env index), until each sub-env has finished
    at least one. An independent loop over ``VecSnakeEnv`` that selects nothing."""
    env = VecSnakeEnv(cfg, n_envs=n_envs, seed=seed)
    obs = env.reset()
    n_finished = np.zeros(n_envs, dtype=np.int64)
    out: list[tuple[int, EpisodeRecord]] = []
    while (n_finished == 0).any():
        obs, _, _, _, info = env.step(policy.act_batch(obs))
        for i in np.flatnonzero(info["done"]).tolist():
            record = EpisodeRecord(
                seed=seed + i,
                score=int(info["score"][i]),
                steps=int(info["steps"][i]),
                length=int(info["length"][i]),
                cause=CAUSE_NAMES[int(info["cause"][i])],
            )
            out.append((i, record))
            n_finished[i] += 1
    return out


@pytest.mark.parametrize("case", [SEEKER_LONG, CYCLE_MIXED], ids=lambda case: case.id)
def test_first_episode_of_each_sub_env_not_the_first_k_to_finish(case):
    """Short and long episodes in one batch: the boards that end early restart
    and finish again before the long ones end. ``evaluate_batched`` keeps one
    episode per board; the forbidden "first K to finish" selection is computed
    here for contrast and is biased towards short, low-scoring episodes."""
    assert case in CASES
    K = case.n
    scalar, batched = run_case(case)
    finished = finished_in_order(PLAYERS[case.player].make(), case.cfg, K, case.seed)

    # Short and long episodes are mixed, and sub-envs did auto-reset meanwhile.
    steps = batched.steps
    assert steps.max() >= 5 * steps.min()
    assert len(finished) >= K + K // 6
    per_env = Counter(i for i, _ in finished)
    assert set(per_env) == set(range(K)) and max(per_env.values()) >= 2

    # The forbidden selection: some boards twice or more, others never.
    forbidden_envs = [i for i, _ in finished[:K]]
    forbidden = EvalResult(policy=batched.policy, env=batched.env, seed=case.seed,
                           episodes=[record for _, record in finished[:K]])
    assert len(set(forbidden_envs)) <= K - K // 10
    assert forbidden.steps.mean() < 0.9 * batched.steps.mean()
    assert forbidden.mean_score < 0.95 * batched.mean_score
    assert forbidden.episodes != batched.episodes
    assert sorted(forbidden.episodes, key=lambda e: e.seed) != batched.episodes

    # evaluate_batched: exactly the first episode of each sub-env, ordered by sub-env.
    first: dict[int, EpisodeRecord] = {}
    for i, record in finished:
        first.setdefault(i, record)
    assert batched.episodes == [first[i] for i in range(K)]
    assert batched.episodes == scalar.episodes
    # Not the last one either.
    last = dict(finished)
    assert batched.episodes != [last[i] for i in range(K)]


def test_runs_one_vec_env_until_the_slowest_first_episode_ends(monkeypatch):
    """``VecSnakeEnv(cfg, n_envs=n_episodes, seed=seed)``, one ``reset``, and
    exactly as many vec steps as the longest first episode has steps, while
    the other sub-envs go through several discarded episodes."""
    built: list[VecSnakeEnv] = []

    class SpyVecEnv(VecSnakeEnv):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.n_resets = 0
            self.n_steps = 0
            self.n_finished = np.zeros(self.n_envs, dtype=np.int64)
            built.append(self)

        def reset(self, seed: int | None = None) -> np.ndarray:
            self.n_resets += 1
            return super().reset(seed)

        def step(self, actions):
            out = super().step(actions)
            self.n_steps += 1
            self.n_finished += out[4]["done"]
            return out

    monkeypatch.setattr(evaluate_module, "VecSnakeEnv", SpyVecEnv)
    cfg = EnvConfig(W=8)
    n, seed = 40, 4242
    result = evaluate_batched(FoodSeekerA(), cfg, n, seed)

    assert len(built) == 1
    spy = built[0]
    assert spy.cfg == cfg and spy.n_envs == n and spy.seed == seed and spy.n_resets == 1
    steps = [e.steps for e in result.episodes]
    assert spy.n_steps == max(steps) > 3 * min(steps)
    assert (spy.n_finished >= 1).all()
    assert spy.n_finished[int(np.argmax(steps))] == 1  # the slowest board ends the evaluation
    assert spy.n_finished.max() >= 3 and spy.n_finished.sum() >= 2 * n  # auto-resets happened ...
    assert len(result.episodes) == n  # ... and still one record per sub-env


# ------------------------------------------------------------- record contents
def test_episode_seeds_are_seed_plus_i_in_sub_env_order():
    cfg = EnvConfig(W=6)
    seed, n = 123, 9
    result = evaluate_batched(FoodSeekerA(), cfg, n, seed)
    assert result.seed == seed
    assert [e.seed for e in result.episodes] == list(range(seed, seed + n))
    # Ordered by sub-env, not by finishing time (the boards finish out of order).
    steps = [e.steps for e in result.episodes]
    assert steps != sorted(steps)
    # Episode i is the single-board evaluation with seed + i ...
    for i in (0, 4, n - 1):
        assert evaluate_batched(FoodSeekerA(), cfg, 1, seed + i).episodes == [result.episodes[i]]
    # ... so the evaluation seeded seed + 1 is this one shifted by one sub-env.
    shifted = evaluate_batched(FoodSeekerA(), cfg, n - 1, seed + 1)
    assert shifted.episodes == result.episodes[1:]
    assert len({(e.score, e.steps) for e in result.episodes}) > n // 2  # boards differ


def test_wall_episodes_report_the_counters_of_the_finished_episode():
    """Hand-checkable: straight from (4, 4) heading right on 8x8 leaves the
    grid on step 4. A record read from the state arrays after the auto-reset
    would say 0 steps and length 3 for every board."""
    result = evaluate_batched(AlwaysStraight(), EnvConfig(W=8), 60, seed=500)
    assert {e.cause for e in result.episodes} == {"wall"}
    assert {e.steps for e in result.episodes} == {4}
    assert all(e.length == 3 + e.score for e in result.episodes)
    assert {e.score for e in result.episodes} == {0, 1}  # food on (4, 5..7) is eaten on the way
    assert result.cause_fractions == {"wall": 1.0, "self": 0.0, "starve": 0.0, "win": 0.0}


def test_win_episodes_fill_the_board():
    W = 4
    result = evaluate_batched(CycleFollowerB(), EnvConfig(W=W, obs=OBS_B), 10, seed=12)
    assert [e.cause for e in result.episodes] == ["win"] * 10
    assert all(e.score == W * W - 3 and e.length == W * W for e in result.episodes)
    assert len({e.steps for e in result.episodes}) > 1


def test_starved_episodes_are_recorded_as_starve():
    """A truncation ends the first episode like a termination does: the food
    seeker is cut off the first time the food is more than H steps away."""
    H = 4
    result = evaluate_batched(FoodSeekerA(), EnvConfig(W=6, starve_limit=H), 30, seed=13)
    assert {e.cause for e in result.episodes} == {"starve"}
    assert all(e.steps >= H and e.length == 3 + e.score for e in result.episodes)
    assert any(e.score == 0 and e.steps == H for e in result.episodes)
    assert sum(e.score for e in result.episodes) >= 30  # many were cut off after eating


def test_self_collision_episodes():
    result = evaluate_batched(FoodSeekerA(), EnvConfig(W=6), 20, seed=14)
    assert {e.cause for e in result.episodes} == {"self"}
    assert all(e.length == 3 + e.score and e.score >= 3 and e.steps > e.score for e in result.episodes)


def test_causes_are_the_names_of_common():
    seen: set[str] = set()
    for case in (CYCLE_MIXED, SEEKER_LONG, matrix_case("table-rel", 6, SMALL_H, 100)):
        assert case in CASES
        for e in run_case(case)[1].episodes:
            assert type(e.cause) is str and e.cause in END_CAUSES
            seen.add(e.cause)
    assert seen == set(END_CAUSES)
    assert END_CAUSES == ("wall", "self", "starve", "win")


# ------------------------------------------------------------- metadata, API
def test_metadata_is_filled_like_evaluate(monkeypatch):
    assert REAL_GIT_HASH is snake_rl.utils.git_hash  # evaluate.py uses the repo helper
    monkeypatch.setattr(evaluate_module, "git_hash", lambda: "abc1234+dirty")
    cfg = EnvConfig(
        W=8, action_mode="absolute", starve_limit=9, obs=OBS_A, reward=RewardConfig(variant="R2", gamma=0.9)
    )
    policy = TablePolicy(4, seed=3)
    result = evaluate_batched(policy, cfg, 5, seed=77)
    assert isinstance(result, EvalResult)
    assert result.policy == policy.name == "table4"
    assert result.env == env_config_to_dict(cfg) and env_config_from_dict(result.env) == cfg
    assert result.seed == 77 and type(result.seed) is int
    assert result.git == "abc1234+dirty"
    assert isinstance(result.wall_time_s, float) and 0.0 < result.wall_time_s < 60.0
    reference = evaluate(policy, cfg, 5, seed=77)
    assert (result.policy, result.env, result.seed, result.git) == (
        reference.policy, reference.env, reference.seed, reference.git)
    assert result.episodes == reference.episodes


def test_default_arguments_are_the_evaluation_protocol():
    params = inspect.signature(evaluate_batched).parameters
    assert list(params) == ["policy", "cfg", "n_episodes", "seed"]
    assert params["n_episodes"].default == 100
    assert params["seed"].default == DEFAULT_EVAL_SEED == 10_000
    result = evaluate_batched(AlwaysStraight(), EnvConfig(W=8))
    assert result.seed == DEFAULT_EVAL_SEED
    assert [e.seed for e in result.episodes] == list(range(10_000, 10_100))


@pytest.mark.parametrize("n_episodes", [0, -1, -100])
def test_n_episodes_below_one_raises(n_episodes):
    class MustNotAct(AlwaysStraight):
        def act_batch(self, obs: np.ndarray) -> np.ndarray:
            raise AssertionError("the policy was queried for an invalid evaluation")

    with pytest.raises(ValueError, match="n_episodes"):
        evaluate_batched(MustNotAct(), EnvConfig(W=8), n_episodes, seed=1)
    with pytest.raises(ValueError, match="n_episodes"):
        evaluate_batched(MustNotAct(), EnvConfig(W=8), n_episodes=n_episodes)


def test_policy_needs_only_name_and_act_batch():
    """No ``act``, no ``reset``, no ``env`` argument: the batched evaluator
    hands over the whole observation batch and nothing else."""

    class Minimal:
        name = "minimal"

        def __init__(self) -> None:
            self.calls: list[tuple[tuple[int, ...], np.dtype]] = []

        def act_batch(self, obs: np.ndarray) -> np.ndarray:
            self.calls.append((obs.shape, obs.dtype))
            return np.zeros(len(obs), dtype=np.int64)

    for obs_cfg, shape in ((OBS_A, (5, 11)), (OBS_B, (5, 8, 8, 8))):
        policy = Minimal()
        result = evaluate_batched(policy, EnvConfig(W=8, obs=obs_cfg), 5, seed=2)
        assert result.policy == "minimal"
        assert [e.steps for e in result.episodes] == [4] * 5
        assert policy.calls == [(shape, np.dtype(np.float32))] * 4  # one call per vec step


def test_reset_is_called_once_before_the_first_action():
    """All first episodes start together, so the per-episode ``reset`` of the
    scalar protocol becomes a single call."""

    class Recording(AlwaysStraight):
        def __init__(self) -> None:
            self.events: list[str] = []

        def reset(self) -> None:
            self.events.append("reset")

        def act_batch(self, obs: np.ndarray) -> np.ndarray:
            self.events.append("act")
            return super().act_batch(obs)

    policy = Recording()
    evaluate_batched(policy, EnvConfig(W=8), 6, seed=1)
    assert policy.events == ["reset"] + ["act"] * 4


def test_single_episode():
    result = evaluate_batched(FoodSeekerA(), EnvConfig(W=8), 1, seed=5)
    reference = evaluate(FoodSeekerA(), EnvConfig(W=8), 1, seed=5)
    assert len(result.episodes) == 1 and result.episodes == reference.episodes
    assert result.mean_score == reference.mean_score and np.isnan(result.ci95)


def test_repeated_evaluation_is_identical():
    """The same boards every time: what makes an evaluation curve comparable
    from one checkpoint to the next."""
    cfg = EnvConfig(W=8, starve_limit=20)
    policy = TablePolicy(3, seed=0)
    first = evaluate_batched(policy, cfg, 50, seed=DEFAULT_EVAL_SEED)
    second = evaluate_batched(policy, cfg, 50, seed=DEFAULT_EVAL_SEED)
    other = evaluate_batched(policy, cfg, 50, seed=DEFAULT_EVAL_SEED + 50)
    assert first.episodes == second.episodes

    def outcomes(result: EvalResult) -> list[tuple[int, int, str]]:
        return [(e.score, e.steps, e.cause) for e in result.episodes]

    assert outcomes(first) != outcomes(other)  # other seeds, other boards


def test_result_saves_and_loads(tmp_path):
    cfg = EnvConfig(W=6, starve_limit=30, obs=OBS_B)
    result = evaluate_batched(CycleFollowerB(), cfg, 9, seed=3)
    path = result.save(tmp_path / "eval" / "cycle.json")
    assert path.is_file()
    loaded = EvalResult.load(path)
    assert loaded.episodes == result.episodes
    assert loaded.to_dict() == result.to_dict()
    assert env_config_from_dict(loaded.env) == cfg
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["policy"] == "cycle" and raw["seed"] == 3 and raw["git"] == FAKE_GIT
    assert raw["summary"]["n_episodes"] == 9
    assert [e["seed"] for e in raw["episodes"]] == list(range(3, 12))
    assert all(e["cause"] in END_CAUSES for e in raw["episodes"])

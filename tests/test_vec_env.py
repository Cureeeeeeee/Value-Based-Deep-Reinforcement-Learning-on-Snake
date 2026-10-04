"""Vectorised environment: the scalar / vectorised regression gate (PLAN.md
section 3.6, last bullet of 3.7).

Contract under test. Sub-environment ``i`` of ``VecSnakeEnv(cfg, n_envs, seed)``
behaves exactly like a scalar ``SnakeEnv(cfg)`` that

(a) was ``reset(seed=seed + i)``,
(b) receives the same actions, and
(c) is ``reset()`` *without* a seed (its RNG stream continues) right after
    each of its episodes ends.

"Exactly" means every observation, reward, flag, counter and internal state
array is equal with no tolerance; float arrays are compared including the sign
of zero. The scalar environment is the oracle, so nothing here re-implements
the rules.

Auto-reset convention: ``step`` returns the first observation of the *new*
episode for sub-envs that finished, and ``info["final_obs"]`` holds the true
successor of the transition (what the scalar ``step`` returns). A learner must
store ``final_obs`` as ``next_obs``, otherwise a starvation truncation could
not be bootstrapped (truncated != terminated).

Layout: equivalence harness and its covering set of configurations, a
coverage gate (an equivalence run that never eats proves nothing), direct
semantic tests, API tests.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pytest

from snake_rl.agents.baselines import BFSGreedyPolicy, HamiltonianPolicy, RandomPolicy
from snake_rl.env.common import (
    AGE_DTYPE,
    CAUSE_NAMES,
    CAUSE_NONE,
    CAUSE_SELF,
    CAUSE_STARVE,
    CAUSE_WALL,
    CAUSE_WIN,
    DELTAS,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    body_age_from_cells,
    cells_from_body_age,
    head_positions,
    initial_cells,
    next_heading,
    sample_food,
)
from snake_rl.env.obs import cell_is_danger
from snake_rl.env.rewards import compute_reward
from snake_rl.env.vec_env import VecSnakeEnv
from tests.helpers import (
    BLOCK_OTHER_HEADING,
    BLOCK_TAIL_CHASE,
    BLOCK_TAIL_CHASE_ACTION,
    COIL_TAIL_A,
    COIL_TAIL_B,
    FAR_FOOD,
    HOOK_COLLIDING_ACTION,
    HOOK_SELF_COLLISION,
    STRAIGHT,
    STRAIGHT_RIGHT,
    TURN_LEFT,
    TURN_RIGHT,
    EnvConfig,
    ObsConfig,
    RewardConfig,
    SnakeEnv,
    serpentine_almost_full,
)

CAUSE_IDS = {name: i for i, name in enumerate(CAUSE_NAMES)}
#: Public state arrays of ``VecSnakeEnv`` with their dtypes.
STATE_DTYPES = {
    "body_age": AGE_DTYPE,
    "heading": np.int64,
    "food": np.int64,
    "length": np.int64,
    "score": np.int64,
    "steps_since_food": np.int64,
    "steps": np.int64,
}
#: Keys of the ``info`` dict returned by ``step`` with their dtypes.
INFO_DTYPES = {
    "final_obs": np.float32,
    "done": np.bool_,
    "cause": np.int64,
    "score": np.int64,
    "length": np.int64,
    "steps": np.int64,
}
#: Plane indices of observation B (8 planes).
HEAD_PLANE, BODY_PLANE, FOOD_PLANE, AGE_PLANE, HEADING_PLANE0 = 0, 1, 2, 3, 4

OBS_A = ObsConfig(kind="A")
OBS_B8 = ObsConfig(kind="B")
OBS_B7 = ObsConfig(kind="B", age_plane=False)  # no age plane, heading planes kept
OBS_B4 = ObsConfig(kind="B", heading_planes=False)  # age plane only
OBS_B3 = ObsConfig(kind="B", age_plane=False, heading_planes=False)  # non-Markov ablation
R0 = RewardConfig(variant="R0")
R1 = RewardConfig(variant="R1")
R2 = RewardConfig(variant="R2")
R2_GAMMA09 = RewardConfig(variant="R2", gamma=0.9)
R3 = RewardConfig(variant="R3")  # asymmetric default: closer +0.01, farther 0
R3_SYM = RewardConfig(variant="R3", farther=-0.01)  # symmetric version of Part B


# ------------------------------------------------------------- policies
class SurvivalPolicy:
    """Uniform over the actions that do not end the episode on this step.

    It eats only by accident, so with a small starve limit most episodes end
    by starvation truncation.
    """

    name = "survive"

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, env: SnakeEnv) -> int:
        head = np.array([env.head])
        safe = []
        for a in range(env.action_space.n):
            h = int(next_heading(env.heading, a, env.cfg.action_mode))
            if not cell_is_danger(env.body_age[None], head + DELTAS[h])[0]:
                safe.append(a)
        if not safe:
            return 0
        return safe[int(self._rng.integers(len(safe)))]


def _mixed_policy(i: int):
    return (RandomPolicy(seed=30_000 + i), BFSGreedyPolicy(), SurvivalPolicy(seed=30_000 + i))[i % 3]


#: name -> factory(sub-env index) -> policy. One instance per scalar env.
POLICIES: dict[str, Callable[[int], object]] = {
    "random": lambda i: RandomPolicy(seed=10_000 + i),
    "bfs": lambda i: BFSGreedyPolicy(),
    "hamiltonian": lambda i: HamiltonianPolicy(),
    "survive": lambda i: SurvivalPolicy(seed=20_000 + i),
    "mixed": _mixed_policy,
}


def survival_actions(vec: VecSnakeEnv, rng: np.random.Generator) -> np.ndarray:
    """Batched survival policy computed from the public state arrays of ``vec``
    (for the tests that have no scalar twin to ask)."""
    n = vec.n_envs
    head = head_positions(vec.body_age)
    keys = rng.random((n, vec.n_actions))
    for a in range(vec.n_actions):
        h = next_heading(vec.heading, np.full(n, a), vec.cfg.action_mode)
        keys[:, a] += cell_is_danger(vec.body_age, head + DELTAS[h])  # fatal actions rank last
    return keys.argmin(axis=1)


# ------------------------------------------------------------- harness
@dataclass
class Tally:
    """What an equivalence run saw, counted on the scalar reference side."""

    transitions: int = 0
    eats: int = 0
    resets: int = 0
    causes: dict[str, int] = field(default_factory=lambda: dict.fromkeys(CAUSE_NAMES[1:], 0))

    def count(self, key: str) -> int:
        return self.causes[key] if key in self.causes else int(getattr(self, key))

    def add(self, other: Tally) -> None:
        self.transitions += other.transitions
        self.eats += other.eats
        self.resets += other.resets
        for name, n in other.causes.items():
            self.causes[name] += n


def assert_rows_equal(what: str, got: np.ndarray, want: np.ndarray, where: str) -> None:
    """Exact comparison of a per-sub-env array (leading axis = sub-env).

    Shape and dtype must match as well. The failure message names ``where``
    (the step) and the first sub-env that differs.
    """
    assert isinstance(got, np.ndarray), f"{where}: {what} is a {type(got).__name__}, expected an ndarray"
    assert got.shape == want.shape, f"{where}: {what} has shape {got.shape}, expected {want.shape}"
    assert got.dtype == want.dtype, f"{where}: {what} has dtype {got.dtype}, expected {want.dtype}"
    differs = got != want
    if got.dtype.kind == "f":
        differs |= np.signbit(got) != np.signbit(want)  # +0.0 vs -0.0
    if not differs.any():
        return
    rows = differs.reshape(got.shape[0], -1).any(axis=1)
    i = int(np.argmax(rows))
    pytest.fail(
        f"{where}, sub-env {i}: {what} differs from the scalar reference "
        f"({int(rows.sum())} of {got.shape[0]} sub-envs differ)\nvec:\n{got[i]}\nscalar:\n{want[i]}"
    )


def scalar_state(envs: list[SnakeEnv]) -> dict[str, np.ndarray]:
    """State of the scalar envs in the layout of the vec state arrays."""
    n = len(envs)
    return {
        "body_age": np.stack([e.body_age for e in envs]),
        "heading": np.array([e.heading for e in envs], dtype=np.int64),
        "food": np.array([e.food for e in envs], dtype=np.int64).reshape(n, 2),
        "length": np.array([e.length for e in envs], dtype=np.int64),
        "score": np.array([e.score for e in envs], dtype=np.int64),
        "steps_since_food": np.array([e.steps_since_food for e in envs], dtype=np.int64),
        "steps": np.array([e.steps for e in envs], dtype=np.int64),
    }


def assert_state_equal(vec: VecSnakeEnv, envs: list[SnakeEnv], where: str) -> None:
    for name, want in scalar_state(envs).items():
        assert_rows_equal(f"state array {name}", getattr(vec, name), want, where)


def assert_invariants(vec: VecSnakeEnv, where: str) -> None:
    try:
        vec.check_invariants()
    except AssertionError as err:
        pytest.fail(f"{where}: check_invariants() raised on a state that equals the scalar reference: {err}")


#: The harness cycles through these so that every accepted container is used.
ACTION_FORMS: tuple[Callable[[np.ndarray], object], ...] = (
    lambda a: a,
    lambda a: a.astype(np.int32),
    lambda a: a.tolist(),
)


def run_equivalence(
    cfg: EnvConfig,
    policy: str,
    *,
    n_envs: int,
    n_steps: int,
    seed: int,
    reseed: bool = False,
    resets_at: dict[int, int | None] | None = None,
) -> Tally:
    """Step ``n_envs`` scalar envs and one ``VecSnakeEnv`` with the same actions
    and compare everything after every step.

    ``reseed``: build the vec env with another seed and pass the real one to
    ``reset(seed=...)`` (exercises re-seeding instead of the constructor).
    ``resets_at``: step index -> seed; before that step every board is reset
    by hand, with ``reset(seed=s)`` or, for ``None``, with ``reset()`` so that
    the RNG streams continue.

    Returns the :class:`Tally` of what happened so that callers can assert
    that the run exercised the rules it was meant to.
    """
    N = n_envs
    obs_shape = cfg.obs.shape(cfg.W)
    resets_at = resets_at or {}
    envs = [SnakeEnv(cfg) for _ in range(N)]
    policies = [POLICIES[policy](i) for i in range(N)]
    for p in policies:
        p.reset()
    scalar_obs = [env.reset(seed=seed + i)[0] for i, env in enumerate(envs)]

    if reseed:
        vec = VecSnakeEnv(cfg, n_envs=N, seed=seed + 7919)
        obs = vec.reset(seed=seed)
    else:
        vec = VecSnakeEnv(cfg, n_envs=N, seed=seed)
        obs = vec.reset()
    assert_rows_equal("reset() observation", obs, np.stack(scalar_obs), "initial reset")
    assert_state_equal(vec, envs, "initial reset")
    assert_invariants(vec, "initial reset")

    tally = Tally()
    held: list[tuple[str, np.ndarray, np.ndarray]] = [("reset() observation", obs, obs.copy())]
    for t in range(n_steps):
        where = f"step {t}"
        if t in resets_at:
            s = resets_at[t]
            scalar_obs = [env.reset(seed=None if s is None else s + i)[0] for i, env in enumerate(envs)]
            for p in policies:
                p.reset()
            obs = vec.reset() if s is None else vec.reset(seed=s)
            assert_rows_equal(f"reset(seed={s}) observation", obs, np.stack(scalar_obs), f"before {where}")
            assert_state_equal(vec, envs, f"reset(seed={s}) before {where}")
            assert_invariants(vec, f"reset(seed={s}) before {where}")

        actions = np.array([p.act(o, e) for p, o, e in zip(policies, scalar_obs, envs)], dtype=np.int64)
        result = vec.step(ACTION_FORMS[t % len(ACTION_FORMS)](actions))
        assert isinstance(result, tuple) and len(result) == 5, f"{where}: step() must return a 5-tuple"
        obs, reward, terminated, truncated, info = result

        # Scalar reference, including the reset-without-seed after an episode end.
        want_obs = np.empty((N, *obs_shape), dtype=np.float32)
        want_final_obs = np.empty((N, *obs_shape), dtype=np.float32)
        want_reward = np.empty(N, dtype=np.float32)
        want_terminated = np.empty(N, dtype=bool)
        want_truncated = np.empty(N, dtype=bool)
        want_info = {key: np.empty(N, dtype=np.int64) for key in ("cause", "score", "length", "steps")}
        for i, env in enumerate(envs):
            score_before = env.score
            o, r, term, trunc, s_info = env.step(int(actions[i]))
            want_final_obs[i] = o
            want_reward[i] = r
            want_terminated[i] = term
            want_truncated[i] = trunc
            want_info["cause"][i] = CAUSE_IDS[s_info["cause"]]
            for key in ("score", "length", "steps"):
                want_info[key][i] = s_info[key]
            tally.eats += s_info["score"] - score_before
            if term or trunc:
                tally.causes[s_info["cause"]] += 1
                tally.resets += 1
                o, _ = env.reset()  # no seed: the stream continues
                policies[i].reset()
            want_obs[i] = o
            scalar_obs[i] = o
        tally.transitions += N

        # Nothing handed out earlier may have been written to by this step.
        for what, arr, snapshot in held:
            if not np.array_equal(arr, snapshot):
                pytest.fail(f"{where}: {what} returned by the previous call was modified in place")

        assert_rows_equal("reward", reward, want_reward, where)
        assert_rows_equal("terminated", terminated, want_terminated, where)
        assert_rows_equal("truncated", truncated, want_truncated, where)
        assert isinstance(info, dict), f"{where}: info is a {type(info).__name__}, expected a dict"
        missing = sorted(set(INFO_DTYPES) - set(info))
        assert not missing, f"{where}: info lacks the keys {missing}"
        assert_rows_equal("info['done']", info["done"], want_terminated | want_truncated, where)
        for key, want in want_info.items():
            assert_rows_equal(f"info[{key!r}]", info[key], want, where)
        assert_rows_equal("info['final_obs']", info["final_obs"], want_final_obs, where)
        assert_rows_equal("observation", obs, want_obs, where)
        observed = vec.observe()
        assert_rows_equal("observe()", observed, want_obs, where)
        assert_state_equal(vec, envs, where)
        assert_invariants(vec, where)

        held = [("observation", obs, obs.copy()), ("reward", reward, reward.copy())]
        held += [("terminated", terminated, terminated.copy()), ("truncated", truncated, truncated.copy())]
        held += [(f"info[{key!r}]", info[key], info[key].copy()) for key in INFO_DTYPES]
        held += [("observe()", observed, observed.copy())]
    return tally


@dataclass(frozen=True)
class Case:
    """One equivalence run and the minimum it must exercise (``expect`` keys:
    cause names, ``eats``, ``resets``)."""

    cfg: EnvConfig
    policy: str
    n_envs: int
    n_steps: int
    seed: int
    reseed: bool = False
    resets_at: dict[int, int | None] = field(default_factory=dict)
    expect: dict[str, int] = field(default_factory=dict)


#: Covering (not cartesian) set: every observation variant, every reward
#: variant, both action modes, W in {4, 6, 8, 10}, default and small starve
#: limits, and a policy for each way an episode can end.
CASES: dict[str, Case] = {
    # --- uniform random actions: wall deaths, resets from short snakes
    "random-A-R0-rel-W8": Case(
        EnvConfig(W=8, obs=OBS_A, reward=R0), "random", n_envs=32, n_steps=120, seed=100,
        expect={"wall": 150, "eats": 20},
    ),
    "random-B8-R1-abs-W10": Case(
        EnvConfig(W=10, action_mode="absolute", obs=OBS_B8, reward=R1), "random", n_envs=16, n_steps=200, seed=200,
        reseed=True, expect={"wall": 100, "eats": 10},
    ),
    "random-B3-R2-rel-W6": Case(
        EnvConfig(W=6, obs=OBS_B3, reward=R2), "random", n_envs=16, n_steps=120, seed=300,
        expect={"wall": 100, "eats": 25},
    ),
    "random-B7-R3-abs-W4": Case(
        EnvConfig(W=4, action_mode="absolute", obs=OBS_B7, reward=R3), "random", n_envs=16, n_steps=120, seed=400,
        resets_at={40: None, 80: 4040}, expect={"wall": 250, "eats": 50},
    ),
    # --- BFS greedy: long snakes, many eats, self collisions
    "bfs-A-R2-rel-W6": Case(
        EnvConfig(W=6, obs=OBS_A, reward=R2_GAMMA09), "bfs", n_envs=8, n_steps=400, seed=500,
        expect={"self": 10, "eats": 400},
    ),
    "bfs-B8-R3sym-abs-W8": Case(
        EnvConfig(W=8, action_mode="absolute", obs=OBS_B8, reward=R3_SYM), "bfs", n_envs=8, n_steps=500, seed=600,
        reseed=True, expect={"self": 8, "eats": 400},
    ),
    "bfs-B7-R0-rel-W10": Case(
        EnvConfig(W=10, obs=OBS_B7, reward=R0), "bfs", n_envs=6, n_steps=500, seed=700,
        expect={"self": 3, "eats": 250},
    ),
    "bfs-B8-R1-rel-W4": Case(
        EnvConfig(W=4, obs=OBS_B8, reward=R1), "bfs", n_envs=8, n_steps=300, seed=800,
        expect={"self": 14, "win": 3, "eats": 500},
    ),
    # --- Hamiltonian: wins, the no-draw-on-win rule, the reset right after a win
    "ham-B8-R2-rel-W4": Case(
        EnvConfig(W=4, obs=OBS_B8, reward=R2), "hamiltonian", n_envs=6, n_steps=400, seed=900,
        expect={"win": 25, "eats": 400},
    ),
    "ham-A-R1-abs-W6": Case(
        EnvConfig(W=6, action_mode="absolute", obs=OBS_A, reward=R1), "hamiltonian", n_envs=3, n_steps=1400, seed=1000,
        expect={"win": 8, "eats": 300},
    ),
    "ham-B8-R0-rel-W10": Case(  # the project's main board, all the way to 100 cells
        EnvConfig(W=10, obs=OBS_B8, reward=R0), "hamiltonian", n_envs=2, n_steps=3000, seed=1800,
        expect={"win": 2, "eats": 195},
    ),
    "ham-B3-R3-rel-W4-H9": Case(
        EnvConfig(W=4, starve_limit=9, obs=OBS_B3, reward=R3), "hamiltonian", n_envs=6, n_steps=300, seed=1100,
        expect={"starve": 30, "win": 10, "eats": 200},
    ),
    # --- survival with a small starve limit: frequent truncations
    "survive-A-R0-rel-W8-H6": Case(
        EnvConfig(W=8, starve_limit=6, obs=OBS_A, reward=R0), "survive", n_envs=16, n_steps=150, seed=1200,
        expect={"starve": 200, "eats": 25},
    ),
    "survive-B4-R3-abs-W6-H12": Case(
        EnvConfig(W=6, action_mode="absolute", starve_limit=12, obs=OBS_B4, reward=R3), "survive", n_envs=12,
        n_steps=200, seed=1300, resets_at={77: None}, expect={"starve": 100, "eats": 35},
    ),
    "survive-B7-R2-rel-W10-H2": Case(
        EnvConfig(W=10, starve_limit=2, obs=OBS_B7, reward=R2), "survive", n_envs=8, n_steps=100, seed=1400,
        expect={"starve": 250},
    ),
    # --- mixed policies in one batch: sub-envs finish at unrelated times
    "mixed-B8-R2-rel-W8-H20": Case(
        EnvConfig(W=8, starve_limit=20, obs=OBS_B8, reward=R2), "mixed", n_envs=12, n_steps=400, seed=1500,
        reseed=True, resets_at={60: 77, 61: None}, expect={"wall": 60, "starve": 45, "eats": 150},
    ),
    "mixed-A-R3sym-abs-W10-H30": Case(
        EnvConfig(W=10, action_mode="absolute", starve_limit=30, obs=OBS_A, reward=R3_SYM), "mixed", n_envs=9,
        n_steps=400, seed=1600, expect={"wall": 40, "starve": 20, "eats": 90},
    ),
    # --- a batch as wide as the training runs, and a batch of one
    "wide-B8-R3-rel-W10-H12": Case(
        EnvConfig(W=10, starve_limit=12, obs=OBS_B8, reward=R3), "mixed", n_envs=160, n_steps=40, seed=1900,
        expect={"wall": 30, "starve": 130, "eats": 170},
    ),
    "single-B8-R0-rel-W6": Case(
        EnvConfig(W=6, obs=OBS_B8, reward=R0), "bfs", n_envs=1, n_steps=300, seed=1700,
        expect={"eats": 30, "resets": 2},
    ),
}


@functools.lru_cache(maxsize=None)
def run_case(name: str) -> Tally:
    """Run a case once per session; the coverage gate reuses the tallies."""
    case = CASES[name]
    return run_equivalence(
        case.cfg, case.policy, n_envs=case.n_envs, n_steps=case.n_steps, seed=case.seed,
        reseed=case.reseed, resets_at=case.resets_at,
    )


# ------------------------------------------------------------- equivalence
@pytest.mark.parametrize("name", list(CASES))
def test_equivalence_with_scalar_reference(name):
    tally = run_case(name)
    for key, minimum in CASES[name].expect.items():
        assert tally.count(key) >= minimum, f"{name}: only {tally.count(key)} x {key}, expected >= {minimum}: {tally}"


def test_equivalence_runs_cover_every_rule():
    """The equivalence runs are only evidence if they reached every branch of
    ``step``: each cause, many eats, many auto-resets."""
    total = Tally()
    for name in CASES:
        total.add(run_case(name))
    assert total.causes["wall"] >= 1000, total
    assert total.causes["self"] >= 50, total
    assert total.causes["starve"] >= 700, total
    assert total.causes["win"] >= 50, total
    assert total.eats >= 3000, total
    assert total.resets >= 2000, total
    assert total.resets == sum(total.causes.values())
    assert total.transitions >= 40_000, total


def test_covering_set_spans_every_axis():
    """Guards the case table itself: dropping a variant must be deliberate."""
    cfgs = [case.cfg for case in CASES.values()]
    assert {c.obs for c in cfgs} == {OBS_A, OBS_B8, OBS_B7, OBS_B4, OBS_B3}
    assert {(c.reward.variant, c.reward.farther) for c in cfgs} == {
        ("R0", 0.0), ("R1", 0.0), ("R2", 0.0), ("R3", 0.0), ("R3", -0.01)}
    assert {c.action_mode for c in cfgs} == {"relative", "absolute"}
    assert {c.W for c in cfgs} == {4, 6, 8, 10}
    assert {c.starve_limit is None for c in cfgs} == {True, False}
    assert {case.policy for case in CASES.values()} == set(POLICIES)


# ------------------------------------------------------------- semantics
def make_vec(W: int = 10, n_envs: int = 3, seed: int = 0, **kwargs) -> VecSnakeEnv:
    vec = VecSnakeEnv(EnvConfig(W=W, **kwargs), n_envs=n_envs, seed=seed)
    vec.reset()
    return vec


def twin_of(vec: VecSnakeEnv, i: int, seed: int = 0) -> SnakeEnv:
    """Scalar env that sub-env ``i`` of ``make_vec(..., seed=seed)`` mirrors."""
    env = SnakeEnv(vec.cfg)
    env.reset(seed=seed + i)
    return env


def actions_with(vec: VecSnakeEnv, i: int, action: int, others: int = STRAIGHT) -> np.ndarray:
    actions = np.full(vec.n_envs, others, dtype=np.int64)
    actions[i] = action
    return actions


def state_snapshot(vec: VecSnakeEnv) -> dict[str, np.ndarray]:
    return {name: getattr(vec, name).copy() for name in STATE_DTYPES}


def test_initial_state():
    W, n, seed = 10, 5, 30
    vec = VecSnakeEnv(EnvConfig(W=W, obs=OBS_B8), n_envs=n, seed=seed)
    obs = vec.reset()
    assert vec.length.tolist() == [3] * n and vec.heading.tolist() == [RIGHT] * n
    assert vec.score.tolist() == vec.steps.tolist() == vec.steps_since_food.tolist() == [0] * n
    for i in range(n):
        assert cells_from_body_age(vec.body_age[i]) == initial_cells(W) == [(5, 3), (5, 4), (5, 5)]
        twin = SnakeEnv(vec.cfg)
        twin_obs, _ = twin.reset(seed=seed + i)
        assert tuple(vec.food[i]) == twin.food
        assert vec.body_age[i][twin.food] == 0
        assert np.array_equal(obs[i], twin_obs)
    assert len({tuple(f) for f in vec.food.tolist()}) > 1  # one generator per sub-env


def test_autoreset_returns_new_episode_obs_and_final_obs_is_the_true_successor():
    vec = make_vec(10, n_envs=3, seed=11, obs=OBS_B8)
    twin = twin_of(vec, 1, seed=11)
    cells = [(2, 7), (2, 8), (2, 9)]  # head on the right wall, heading right
    vec.set_state(1, cells, FAR_FOOD)
    twin.set_state(cells, FAR_FOOD)

    obs, reward, terminated, truncated, info = vec.step(actions_with(vec, 1, STRAIGHT))
    s_obs, s_reward, s_term, s_trunc, s_info = twin.step(STRAIGHT)
    assert s_term and s_info["cause"] == "wall"
    assert terminated.tolist() == [False, True, False] and not truncated.any()
    assert info["done"].tolist() == [False, True, False]
    assert info["cause"].tolist() == [CAUSE_NONE, CAUSE_WALL, CAUSE_NONE]
    assert reward[1] == s_reward == -1.0

    # final_obs: the board the snake died on (wall: grid untouched).
    final = info["final_obs"][1]
    assert np.array_equal(final, s_obs)
    assert final[HEAD_PLANE, 2, 9] == 1.0 and final[BODY_PLANE].sum() == 3.0
    # obs: first observation of the next episode, with the continued RNG stream.
    reset_obs, _ = twin.reset()
    assert np.array_equal(obs[1], reset_obs)
    assert obs[1, HEAD_PLANE, 5, 5] == 1.0
    assert not np.array_equal(obs[1], final)
    assert tuple(vec.food[1]) == twin.food
    # Sub-envs that did not finish: final_obs is simply the next observation.
    assert np.array_equal(info["final_obs"][[0, 2]], obs[[0, 2]])
    assert np.array_equal(obs, vec.observe())


def test_heading_and_step_counter_are_updated_on_the_fatal_step():
    vec = make_vec(10, n_envs=2, seed=3, obs=OBS_B8)
    twin = twin_of(vec, 0, seed=3)
    cells = [(0, 3), (0, 4), (0, 5)]  # top row heading right; turning left leaves the grid
    vec.set_state(0, cells, food=(9, 9))
    twin.set_state(cells, food=(9, 9))
    obs, reward, terminated, truncated, info = vec.step(actions_with(vec, 0, TURN_LEFT))
    s_obs, *_ = twin.step(TURN_LEFT)
    assert terminated[0] and info["cause"][0] == CAUSE_WALL and info["steps"][0] == 1
    final = info["final_obs"][0]
    assert np.array_equal(final, s_obs)
    assert (final[HEADING_PLANE0 + UP] == 1.0).all() and (final[HEADING_PLANE0 + RIGHT] == 0.0).all()
    assert final[HEAD_PLANE, 0, 5] == 1.0  # the head itself did not move


def test_self_collision_into_age2_cell():
    vec = make_vec(10, n_envs=3, seed=6, obs=OBS_B8)
    twin = twin_of(vec, 1, seed=6)
    vec.set_state(1, HOOK_SELF_COLLISION, FAR_FOOD)
    twin.set_state(HOOK_SELF_COLLISION, FAR_FOOD)
    assert vec.heading[1] == LEFT
    obs, reward, terminated, truncated, info = vec.step(actions_with(vec, 1, HOOK_COLLIDING_ACTION))
    s_obs, s_reward, s_term, _, s_info = twin.step(HOOK_COLLIDING_ACTION)
    assert s_term and s_info["cause"] == "self"
    assert terminated[1] and not truncated[1]
    assert info["cause"][1] == CAUSE_SELF and reward[1] == s_reward == -1.0
    assert info["length"][1] == 5 and info["score"][1] == 0 and info["steps"][1] == 1
    # True successor: the tail (3, 4) vacated, the head was not placed on (4, 4).
    final = info["final_obs"][1]
    assert np.array_equal(final, s_obs)
    assert final[BODY_PLANE, 3, 4] == 0.0 and final[BODY_PLANE].sum() == 4.0
    assert final[HEAD_PLANE, 5, 4] == 1.0 and (final[HEADING_PLANE0 + UP] == 1.0).all()
    # The state arrays already hold the next episode.
    assert vec.length[1] == 3 and cells_from_body_age(vec.body_age[1]) == initial_cells(10)
    assert np.array_equal(obs[1], twin.reset()[0])


def test_tail_chase_is_legal():
    """Moving into the cell the tail vacates this step is allowed (decrement
    first, collision test second)."""
    vec = make_vec(10, n_envs=3, seed=1)
    vec.set_state(1, BLOCK_TAIL_CHASE, FAR_FOOD)
    assert vec.heading[1] == LEFT
    obs, reward, terminated, truncated, info = vec.step(actions_with(vec, 1, BLOCK_TAIL_CHASE_ACTION))
    assert not terminated[1] and not truncated[1] and not info["done"][1]
    assert info["cause"][1] == CAUSE_NONE and reward[1] == 0.0
    assert cells_from_body_age(vec.body_age[1]) == [(4, 5), (5, 5), (5, 4), (4, 4)]
    assert vec.heading[1] == UP and vec.length[1] == 4
    assert vec.steps[1] == 1 and vec.steps_since_food[1] == 1
    # Round the 2x2 block for a while: never a collision.
    for _ in range(12):
        _, _, terminated, truncated, _ = vec.step(actions_with(vec, 1, TURN_RIGHT))
        assert not terminated[1] and not truncated[1]
    vec.check_invariants()


def test_growth_on_eating():
    vec = make_vec(10, n_envs=3, seed=8, obs=OBS_B8)
    twin = twin_of(vec, 2, seed=8)
    vec.set_state(2, STRAIGHT_RIGHT, food=(5, 6))
    twin.set_state(STRAIGHT_RIGHT, food=(5, 6))
    obs, reward, terminated, truncated, info = vec.step(actions_with(vec, 2, STRAIGHT, others=TURN_LEFT))
    s_obs, *_ = twin.step(STRAIGHT)
    assert not terminated[2] and not truncated[2] and reward[2] == 1.0
    assert vec.length[2] == 4 and vec.score[2] == 1 and vec.steps_since_food[2] == 0
    assert info["score"][2] == 1 and info["length"][2] == 4 and info["cause"][2] == CAUSE_NONE
    assert cells_from_body_age(vec.body_age[2]) == [(5, 3), (5, 4), (5, 5), (5, 6)]  # tail did not move
    # The new food comes from the sub-env's own generator, one draw, same protocol.
    assert tuple(vec.food[2]) == twin.food != (5, 6)
    assert vec.body_age[2][twin.food] == 0
    assert np.array_equal(obs[2], s_obs) and np.array_equal(info["final_obs"][2], s_obs)


def test_r2_reward_after_eating_uses_the_new_food():
    """Phi(s') of an eating step is taken at the freshly drawn food (d_next),
    not at the eaten one (d_moved = 0)."""
    W, n, seed = 10, 4, 40
    vec = make_vec(W, n_envs=n, seed=seed, reward=R2)
    twins = [twin_of(vec, i, seed=seed) for i in range(n)]
    for i, twin in enumerate(twins):
        vec.set_state(i, STRAIGHT_RIGHT, food=(5, 6))
        twin.set_state(STRAIGHT_RIGHT, food=(5, 6))
    _, reward, terminated, _, _ = vec.step(np.full(n, STRAIGHT))
    assert not terminated.any() and vec.score.tolist() == [1] * n

    def eating_reward(d_next: int) -> np.ndarray:
        return compute_reward(R2, W, ate=True, died=False, terminal=False, d_prev=1, d_moved=0, d_next=d_next)

    for i, twin in enumerate(twins):
        _, s_reward, *_ = twin.step(STRAIGHT)
        assert reward[i] == np.float32(s_reward), f"sub-env {i}"
        d_next = abs(5 - int(vec.food[i, 0])) + abs(6 - int(vec.food[i, 1]))  # head (5, 6) -> new food
        assert reward[i] == eating_reward(d_next) != eating_reward(0), f"sub-env {i}"


def test_eating_step_skips_the_starvation_check():
    H = 20
    vec = make_vec(10, n_envs=2, starve_limit=H)
    vec.set_state(0, STRAIGHT_RIGHT, food=(5, 6), steps_since_food=H - 1)
    _, reward, terminated, truncated, info = vec.step(actions_with(vec, 0, STRAIGHT))
    assert not terminated[0] and not truncated[0] and info["cause"][0] == CAUSE_NONE
    assert reward[0] == 1.0 and vec.steps_since_food[0] == 0 and vec.score[0] == 1


#: R2 reward of the starvation step below (head (1, 4) -> (1, 5), food (0, 0):
#: distance 5 -> 6) with the successor potential kept, i.e. NOT treated as
#: terminal. Computed rather than typed so that it is float32-exact.
R2_STARVE_REWARD = float(
    compute_reward(R2, 8, ate=False, died=False, terminal=False, d_prev=5, d_moved=6, d_next=6)
)


@pytest.mark.parametrize(
    "reward_cfg, expected", [(R0, 0.0), (R1, -0.01), (R2, R2_STARVE_REWARD)], ids=["R0", "R1", "R2"]
)
def test_starvation_is_a_truncation_without_death_penalty(reward_cfg, expected):
    """truncated != terminated: no -1, the shaping potential of the successor
    is kept (R2), and final_obs is the real successor so that the learner can
    bootstrap from it."""
    H = 9
    vec = make_vec(8, n_envs=3, seed=5, starve_limit=H, obs=OBS_B8, reward=reward_cfg)
    twin = twin_of(vec, 2, seed=5)
    cells = [(1, 2), (1, 3), (1, 4)]
    vec.set_state(2, cells, FAR_FOOD, steps_since_food=H - 1)
    twin.set_state(cells, FAR_FOOD, steps_since_food=H - 1)

    obs, reward, terminated, truncated, info = vec.step(actions_with(vec, 2, STRAIGHT, others=TURN_LEFT))
    s_obs, s_reward, s_term, s_trunc, s_info = twin.step(STRAIGHT)
    assert s_trunc and not s_term and s_info["cause"] == "starve"
    assert truncated.tolist() == [False, False, True]
    assert not terminated.any()
    assert info["done"].tolist() == [False, False, True]
    assert info["cause"][2] == CAUSE_STARVE
    assert reward[2] == np.float32(expected) == np.float32(s_reward)
    assert info["steps"][2] == 1 and info["length"][2] == 3 and info["score"][2] == 0

    final = info["final_obs"][2]
    assert np.array_equal(final, s_obs)
    assert final[HEAD_PLANE, 1, 5] == 1.0  # the ordinary next state: the snake did move
    reset_obs, _ = twin.reset()
    assert np.array_equal(obs[2], reset_obs)
    assert not np.array_equal(final, obs[2])
    assert vec.steps_since_food[2] == 0 and vec.steps[2] == 0 and vec.length[2] == 3


@pytest.mark.parametrize("H", [1, 2, 25])
def test_starvation_truncates_at_exactly_H(H):
    n = 3
    vec = make_vec(8, n_envs=n, starve_limit=H)
    for i in range(n):
        vec.set_state(i, [(4, 2), (4, 3), (4, 4)], FAR_FOOD)
    # Always turning left cycles through a 2x2 block and never eats.
    for t in range(1, H + 1):
        _, reward, terminated, truncated, info = vec.step(np.full(n, TURN_LEFT))
        assert not terminated.any() and (reward == 0.0).all()
        if t < H:
            assert not truncated.any() and (info["cause"] == CAUSE_NONE).all()
            assert vec.steps_since_food.tolist() == [t] * n
        else:
            assert truncated.all() and (info["cause"] == CAUSE_STARVE).all()
            assert info["steps"].tolist() == [H] * n
    assert vec.steps_since_food.tolist() == [0] * n and vec.steps.tolist() == [0] * n


def test_win_on_full_board_draws_no_food():
    W, n, seed = 4, 6, 21
    vec = make_vec(W, n_envs=n, seed=seed, obs=OBS_B8)
    cells, food = serpentine_almost_full(W)
    twins = []
    for i in range(n):
        vec.set_state(i, cells, food)
        twins.append(twin_of(vec, i, seed=seed))
        twins[i].set_state(cells, food)
    vec.check_invariants()

    obs, reward, terminated, truncated, info = vec.step(np.full(n, STRAIGHT))
    assert terminated.all() and not truncated.any() and info["done"].all()
    assert (info["cause"] == CAUSE_WIN).all() and (reward == 1.0).all()
    assert (info["length"] == W * W).all() and (info["score"] == 1).all() and (info["steps"] == 1).all()
    assert (info["final_obs"][:, BODY_PLANE] == 1.0).all()  # full board
    for i, twin in enumerate(twins):
        s_obs, _, s_term, _, s_info = twin.step(STRAIGHT)
        assert s_term and s_info["cause"] == "win"
        assert np.array_equal(info["final_obs"][i], s_obs)
        reset_obs, _ = twin.reset()
        assert np.array_equal(obs[i], reset_obs), f"sub-env {i}"
        assert tuple(vec.food[i]) == twin.food, f"sub-env {i}"
    assert vec.length.tolist() == [3] * n and vec.score.tolist() == [0] * n
    vec.check_invariants()

    # Teeth: had the winning step burnt a food draw on the full board, the
    # food of the next episode would come from the third draw, not the second.
    initial = body_age_from_cells(initial_cells(W), W)
    told_apart = 0
    for i in range(n):
        rng = np.random.default_rng(seed + i)
        sample_food(rng, initial)  # reset()
        second = sample_food(rng, initial)  # auto-reset after the win
        third = sample_food(rng, initial)
        assert tuple(vec.food[i]) == second
        told_apart += second != third
    assert told_apart > 0


def test_info_reports_the_finished_episode_while_state_holds_the_new_one():
    W, seed = 10, 4
    vec = make_vec(W, n_envs=3, seed=seed)
    twin = twin_of(vec, 1, seed=seed)
    cells = [(7, 5), (7, 6), (7, 7)]  # food in front of the head, then the right wall
    vec.set_state(1, cells, food=(7, 8))
    twin.set_state(cells, food=(7, 8))
    for i in (0, 2):  # the neighbours circle a 2x2 block and stay alive
        vec.set_state(i, STRAIGHT_RIGHT, FAR_FOOD)

    actions = actions_with(vec, 1, STRAIGHT, others=TURN_LEFT)
    for t in (1, 2):
        _, _, terminated, truncated, info = vec.step(actions)
        _, _, s_term, _, s_info = twin.step(STRAIGHT)
        assert not terminated.any() and not truncated.any() and not s_term
        assert (info["cause"] == CAUSE_NONE).all()
        # Nobody finished: info mirrors the live state arrays.
        assert np.array_equal(info["score"], vec.score) and np.array_equal(info["steps"], vec.steps)
        assert np.array_equal(info["length"], vec.length)
        assert info["score"][1] == s_info["score"] >= 1 and vec.steps[1] == t

    _, reward, terminated, truncated, info = vec.step(actions)
    _, s_reward, s_term, _, s_info = twin.step(STRAIGHT)
    assert s_term and s_info["cause"] == "wall" and s_info["score"] >= 1
    assert terminated.tolist() == [False, True, False] and not truncated.any()
    assert info["cause"].tolist() == [CAUSE_NONE, CAUSE_WALL, CAUSE_NONE]
    assert reward[1] == s_reward == -1.0
    # info: the episode that just ended ...
    assert info["score"][1] == s_info["score"]
    assert info["length"][1] == s_info["length"] == 3 + s_info["score"]
    assert info["steps"][1] == s_info["steps"] == 3
    # ... state arrays: the episode that just started.
    assert vec.score[1] == 0 and vec.length[1] == 3 and vec.steps[1] == 0 and vec.steps_since_food[1] == 0
    assert vec.heading[1] == RIGHT
    assert cells_from_body_age(vec.body_age[1]) == initial_cells(W)
    assert vec.body_age[1][tuple(vec.food[1])] == 0
    twin.reset()
    assert tuple(vec.food[1]) == twin.food
    for i in (0, 2):
        assert info["score"][i] == vec.score[i] == 0
        assert info["length"][i] == vec.length[i] == 3
        assert info["steps"][i] == vec.steps[i] == 3


def test_absolute_mode_reverse_is_straight():
    vec = make_vec(10, n_envs=2, action_mode="absolute")
    for i in range(2):
        vec.set_state(i, STRAIGHT_RIGHT, FAR_FOOD)  # heading right
    _, _, terminated, _, _ = vec.step(np.array([LEFT, UP]))  # sub-env 0: 180-degree reversal
    assert not terminated.any()
    assert vec.heading.tolist() == [RIGHT, UP]
    assert head_positions(vec.body_age).tolist() == [[5, 6], [4, 5]]
    vec.step(np.array([UP, DOWN]))  # sub-env 1: reversal again
    assert vec.heading.tolist() == [UP, UP]
    assert head_positions(vec.body_age).tolist() == [[4, 6], [3, 5]]


# ------------------------------------------------------------- API
@pytest.mark.parametrize(
    "kwargs, obs_shape, n_actions",
    [
        ({}, (11,), 3),
        ({"action_mode": "absolute"}, (11,), 4),
        ({"obs": OBS_B8}, (8, 10, 10), 3),
        ({"obs": OBS_B7}, (7, 10, 10), 3),
        ({"obs": OBS_B4}, (4, 10, 10), 3),
        ({"obs": OBS_B3, "action_mode": "absolute"}, (3, 10, 10), 4),
    ],
)
def test_attributes_shapes_and_dtypes(kwargs, obs_shape, n_actions):
    n, W = 5, 10
    cfg = EnvConfig(W=W, **kwargs)
    vec = VecSnakeEnv(cfg, n_envs=n, seed=17)
    assert vec.cfg == cfg and vec.n_envs == n and vec.W == W and vec.H == cfg.H == 100
    assert vec.n_actions == n_actions and tuple(vec.obs_shape) == obs_shape and vec.seed == 17

    obs = vec.reset()
    assert isinstance(obs, np.ndarray)  # the observation only, no (obs, info) tuple
    assert obs.shape == (n, *obs_shape) and obs.dtype == np.float32
    state_shapes = {"body_age": (n, W, W), "food": (n, 2)}
    for name, dtype in STATE_DTYPES.items():
        arr = getattr(vec, name)
        assert isinstance(arr, np.ndarray), name
        assert arr.shape == state_shapes.get(name, (n,)), name
        assert arr.dtype == dtype, name

    result = vec.step(np.zeros(n, dtype=np.int64))
    assert isinstance(result, tuple) and len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert obs.shape == (n, *obs_shape) and obs.dtype == np.float32
    assert reward.shape == (n,) and reward.dtype == np.float32
    assert terminated.shape == (n,) and terminated.dtype == np.bool_
    assert truncated.shape == (n,) and truncated.dtype == np.bool_
    assert isinstance(info, dict) and set(INFO_DTYPES) <= set(info)
    for key, dtype in INFO_DTYPES.items():
        assert isinstance(info[key], np.ndarray), key
        assert info[key].shape == ((n, *obs_shape) if key == "final_obs" else (n,)), key
        assert info[key].dtype == dtype, key

    observed = vec.observe()
    assert observed.shape == (n, *obs_shape) and observed.dtype == np.float32
    assert np.array_equal(observed, obs)


def test_default_constructor():
    vec = VecSnakeEnv()
    assert vec.cfg == EnvConfig() and vec.n_envs == 128 and vec.seed == 0
    assert vec.W == 10 and vec.H == 100 and vec.n_actions == 3 and tuple(vec.obs_shape) == (11,)
    assert vec.reset().shape == (128, 11)


@pytest.mark.parametrize("W, H", [(8, 100), (10, 100), (16, 256)])
def test_default_starve_limit(W, H):
    assert VecSnakeEnv(EnvConfig(W=W), n_envs=2).H == H
    assert VecSnakeEnv(EnvConfig(W=W, starve_limit=7), n_envs=2).H == 7


def test_invalid_construction_rejected():
    with pytest.raises(ValueError):
        VecSnakeEnv(EnvConfig(W=3), n_envs=4)
    with pytest.raises(ValueError):
        VecSnakeEnv(EnvConfig(W=8), n_envs=0)
    with pytest.raises(ValueError):
        VecSnakeEnv(EnvConfig(W=8), n_envs=-2)


def test_step_before_reset_raises():
    vec = VecSnakeEnv(EnvConfig(W=8), n_envs=4)
    with pytest.raises(RuntimeError):
        vec.step(np.zeros(4, dtype=np.int64))


@pytest.mark.parametrize(
    "action_mode, bad",
    [
        pytest.param("relative", np.zeros(5, dtype=np.int64), id="too-long"),
        pytest.param("relative", np.zeros(3, dtype=np.int64), id="too-short"),
        pytest.param("relative", np.zeros((4, 1), dtype=np.int64), id="shape-N-1"),
        pytest.param("relative", np.zeros((1, 4), dtype=np.int64), id="shape-1-N"),
        pytest.param("relative", 0, id="scalar"),
        pytest.param("relative", np.array([0, 1, 2, 3]), id="3-of-3-actions"),
        pytest.param("absolute", np.array([0, 1, 2, 4]), id="4-of-4-actions"),
        pytest.param("relative", np.array([0, -1, 0, 0]), id="negative"),
        pytest.param("relative", np.zeros(4, dtype=np.float32), id="float32-array"),
        pytest.param("relative", np.array([0.0, 1.0, 2.0, 1.0]), id="float64-array"),
        pytest.param("relative", [0.0, 1.0, 2.0, 1.0], id="float-list"),
        pytest.param("relative", [0, 1.5, 2, 1], id="fractional-list"),
    ],
)
def test_invalid_actions_rejected(action_mode, bad):
    vec = make_vec(8, n_envs=4, seed=2, action_mode=action_mode)
    before = state_snapshot(vec)
    with pytest.raises(ValueError):
        vec.step(bad)
    # Validation comes before any mutation, as in the scalar env.
    for name, snapshot in before.items():
        assert np.array_equal(getattr(vec, name), snapshot), name
    vec.step(np.zeros(4, dtype=np.int64))  # and the env is still usable


def test_single_env_batch_takes_shape_1_actions():
    vec = make_vec(8, n_envs=1)
    with pytest.raises(ValueError):
        vec.step(0)
    with pytest.raises(ValueError):
        vec.step(np.int64(0))
    obs, reward, terminated, truncated, info = vec.step([0])
    assert obs.shape == (1, 11) and reward.shape == terminated.shape == truncated.shape == (1,)
    assert info["final_obs"].shape == (1, 11) and info["cause"].shape == (1,)
    assert vec.steps.tolist() == [1]


def test_integer_action_containers_are_equivalent():
    cfg = EnvConfig(W=8, obs=OBS_B8, reward=R2)
    n = 6
    rng = np.random.default_rng(0)
    forms = {
        "int64": lambda a: a.astype(np.int64),
        "int32": lambda a: a.astype(np.int32),
        "int8": lambda a: a.astype(np.int8),
        "uint8": lambda a: a.astype(np.uint8),
        "list": lambda a: a.tolist(),
        "tuple": lambda a: tuple(a.tolist()),
    }
    vecs = {name: VecSnakeEnv(cfg, n_envs=n, seed=12) for name in forms}
    for vec in vecs.values():
        vec.reset()
    for _ in range(40):
        actions = rng.integers(3, size=n)
        results = {name: vecs[name].step(form(actions)) for name, form in forms.items()}
        ref = results["int64"]
        for name, res in results.items():
            assert np.array_equal(res[0], ref[0]) and np.array_equal(res[1], ref[1]), name
            assert np.array_equal(res[2], ref[2]) and np.array_equal(res[3], ref[3]), name
            assert np.array_equal(vecs[name].body_age, vecs["int64"].body_age), name


def test_step_does_not_modify_the_actions_argument():
    vec = make_vec(8, n_envs=4)
    actions = np.array([0, 1, 2, 1], dtype=np.int64)
    vec.step(actions)
    assert actions.tolist() == [0, 1, 2, 1]


def _rollout(vec: VecSnakeEnv, n_steps: int, action_seed: int) -> list[tuple]:
    """Outputs of ``n_steps`` survival-policy steps (the policy only reads the
    public state, so equal states give equal actions)."""
    rng = np.random.default_rng(action_seed)
    out = []
    for _ in range(n_steps):
        obs, reward, terminated, truncated, info = vec.step(survival_actions(vec, rng))
        food = vec.food.copy()
        out.append((obs, reward, terminated, truncated, info["final_obs"], info["cause"], info["score"], food))
    return out


def _assert_rollouts_equal(a: list[tuple], b: list[tuple]) -> None:
    assert len(a) == len(b)
    for t, (sa, sb) in enumerate(zip(a, b)):
        for k, (xa, xb) in enumerate(zip(sa, sb)):
            assert np.array_equal(xa, xb), f"step {t}, output {k}"


def test_reset_with_seed_reproduces():
    cfg = EnvConfig(W=8, starve_limit=15, obs=OBS_B8, reward=R1)
    a = VecSnakeEnv(cfg, n_envs=6, seed=0)
    b = VecSnakeEnv(cfg, n_envs=6, seed=99)  # the constructor seed is overridden by reset(seed=...)
    first_a = a.reset(seed=42)
    first_b = b.reset(seed=42)
    assert np.array_equal(first_a, first_b) and np.array_equal(a.food, b.food)
    roll_a = _rollout(a, 80, action_seed=1)
    roll_b = _rollout(b, 80, action_seed=1)
    _assert_rollouts_equal(roll_a, roll_b)
    assert sum(int(step[5].astype(bool).sum()) for step in roll_a) > 5  # the rollout crossed auto-resets
    # Same object, same seed again: the whole trajectory replays.
    assert np.array_equal(a.reset(seed=42), first_a)
    _assert_rollouts_equal(_rollout(a, 80, action_seed=1), roll_a)


def test_first_reset_without_seed_uses_the_constructor_seed():
    cfg = EnvConfig(W=8, obs=OBS_B8)
    a = VecSnakeEnv(cfg, n_envs=6, seed=31)
    b = VecSnakeEnv(cfg, n_envs=6, seed=0)
    assert np.array_equal(a.reset(), b.reset(seed=31))
    assert np.array_equal(a.food, b.food)


def test_reset_with_seed_zero_reseeds():
    """Seed 0 is falsy: reset(seed=0) must still re-create the generators."""
    cfg = EnvConfig(W=8, obs=OBS_B8)
    a = VecSnakeEnv(cfg, n_envs=6, seed=5)
    a.reset()
    b = VecSnakeEnv(cfg, n_envs=6, seed=0)
    assert np.array_equal(a.reset(seed=0), b.reset()) and np.array_equal(a.food, b.food)
    assert a.seed == 0


def test_seed_attribute_tracks_the_generators_in_use():
    vec = make_vec(8, n_envs=3, seed=1)
    twin = make_vec(8, n_envs=3, seed=1)
    vec.reset(seed=42)
    twin.reset(seed=42)
    assert vec.seed == 42
    vec.reset()  # the streams continue, the base seed is unchanged
    twin.reset()
    assert vec.seed == 42
    # A rejected seed leaves the attribute, the boards and the streams alone.
    before = state_snapshot(vec)
    with pytest.raises(ValueError):
        vec.reset(seed=-1)
    assert vec.seed == 42
    for name, snapshot in before.items():
        assert np.array_equal(getattr(vec, name), snapshot), name
    assert np.array_equal(vec.reset(), twin.reset()) and np.array_equal(vec.food, twin.food)


def test_different_seeds_differ_and_sub_env_i_uses_seed_plus_i():
    cfg = EnvConfig(W=10, obs=OBS_B8)
    n = 16
    a = VecSnakeEnv(cfg, n_envs=n, seed=0)
    b = VecSnakeEnv(cfg, n_envs=n, seed=1000)
    c = VecSnakeEnv(cfg, n_envs=n, seed=1)
    obs_a, obs_b, obs_c = a.reset(), b.reset(), c.reset()
    assert not np.array_equal(obs_a, obs_b) and not np.array_equal(a.food, b.food)
    assert len({tuple(f) for f in a.food.tolist()}) > n // 2  # sub-envs do not share a stream
    # seed + i: the batch seeded 1 is the batch seeded 0 shifted by one sub-env.
    assert np.array_equal(obs_c[:-1], obs_a[1:]) and np.array_equal(c.food[:-1], a.food[1:])


def test_reset_without_seed_continues_the_rng_streams():
    cfg = EnvConfig(W=10, obs=OBS_B8)
    n, seed = 5, 7
    vec = VecSnakeEnv(cfg, n_envs=n, seed=0)
    vec.reset(seed=seed)
    first = [vec.reset().copy() for _ in range(5)]
    vec.reset(seed=seed)
    second = [vec.reset().copy() for _ in range(5)]
    assert all(np.array_equal(x, y) for x, y in zip(first, second))
    assert len({x.tobytes() for x in first}) > 1  # the streams moved on between resets
    # The scalar env does the same with seed + i.
    for i in range(n):
        twin = SnakeEnv(cfg)
        twin.reset(seed=seed + i)
        for k in range(5):
            assert np.array_equal(first[k][i], twin.reset()[0]), f"sub-env {i}, reset {k}"


def test_reset_restarts_every_board_mid_episode():
    W, n = 8, 6
    vec = make_vec(W, n_envs=n, seed=3, starve_limit=30)
    rng = np.random.default_rng(0)
    for _ in range(25):
        vec.step(survival_actions(vec, rng))
    assert (vec.steps > 0).any()
    obs = vec.reset()
    assert vec.length.tolist() == [3] * n and vec.heading.tolist() == [RIGHT] * n
    assert vec.score.tolist() == vec.steps.tolist() == vec.steps_since_food.tolist() == [0] * n
    for i in range(n):
        assert cells_from_body_age(vec.body_age[i]) == initial_cells(W)
        assert vec.body_age[i][tuple(vec.food[i])] == 0
    assert np.array_equal(obs, vec.observe())
    vec.check_invariants()


def test_sub_env_is_independent_of_batch_size_and_of_its_neighbours():
    """Sub-env i of a batch seeded ``s`` is the single-env batch seeded ``s + i``."""
    cfg = EnvConfig(W=8, starve_limit=25, obs=OBS_B8, reward=R2)
    N, seed = 24, 300
    picks = (0, 7, 23)
    big = VecSnakeEnv(cfg, n_envs=N, seed=seed)
    singles = {i: VecSnakeEnv(cfg, n_envs=1, seed=seed + i) for i in picks}
    obs = big.reset()
    for i, single in singles.items():
        assert np.array_equal(single.reset()[0], obs[i])

    rng = np.random.default_rng(1)
    n_done = n_eaten = 0
    for t in range(250):
        actions = survival_actions(big, rng)
        score_before = big.score[list(picks)].copy()
        obs, reward, terminated, truncated, info = big.step(actions)
        for k, (i, single) in enumerate(singles.items()):
            o1, r1, term1, trunc1, info1 = single.step(actions[i : i + 1])
            where = f"step {t}, sub-env {i}"
            assert np.array_equal(o1[0], obs[i]), where
            assert r1[0] == reward[i] and term1[0] == terminated[i] and trunc1[0] == truncated[i], where
            for key in INFO_DTYPES:
                assert np.array_equal(info1[key][0], info[key][i]), f"{where}: info[{key!r}]"
            for name in STATE_DTYPES:
                assert np.array_equal(getattr(single, name)[0], getattr(big, name)[i]), f"{where}: {name}"
            n_done += int(info["done"][i])
            n_eaten += int(info["score"][i] - score_before[k])
    assert n_done >= 6 and n_eaten >= 6  # the comparison crossed resets and food draws


def test_returned_arrays_are_never_modified_by_later_calls():
    cfg = EnvConfig(W=6, starve_limit=8, obs=OBS_B8, reward=R1)
    n = 6
    vec = VecSnakeEnv(cfg, n_envs=n, seed=2)
    rng = np.random.default_rng(0)
    held: list[tuple[str, np.ndarray, np.ndarray]] = []

    def hold(what: str, arr: np.ndarray) -> None:
        held.append((what, arr, arr.copy()))

    hold("reset() observation", vec.reset())
    hold("observe() after reset", vec.observe())
    n_done = 0
    for t in range(120):
        obs, reward, terminated, truncated, info = vec.step(survival_actions(vec, rng))
        hold(f"obs of step {t}", obs)
        hold(f"reward of step {t}", reward)
        hold(f"terminated of step {t}", terminated)
        hold(f"truncated of step {t}", truncated)
        for key in INFO_DTYPES:
            hold(f"info[{key!r}] of step {t}", info[key])
        if t % 7 == 0:
            hold(f"observe() after step {t}", vec.observe())
        n_done += int(info["done"].sum())
    assert n_done > 20  # auto-resets happened while the arrays were held

    vec.set_state(0, [(1, 1), (1, 2), (1, 3)], food=(4, 4))
    hold("observe() after set_state", vec.observe())
    vec.step(np.zeros(n, dtype=np.int64))
    hold("reset() mid-episode", vec.reset())
    vec.step(np.zeros(n, dtype=np.int64))
    hold("reset(seed=5)", vec.reset(seed=5))
    vec.step(np.zeros(n, dtype=np.int64))
    vec.observe()
    for what, arr, snapshot in held:
        assert np.array_equal(arr, snapshot), f"{what} was modified in place by a later call"


def test_obs_and_final_obs_are_separate_arrays():
    """Editing the acting observation in place must never reach the successor
    that the replay buffer stores - on steps with and without an episode end."""
    vec = make_vec(8, n_envs=4, seed=2)
    with_done = without_done = 0
    for _ in range(30):
        obs, _, _, _, info = vec.step(np.zeros(4, dtype=np.int64))  # straight into the wall, again and again
        assert not np.shares_memory(obs, info["final_obs"])
        final = info["final_obs"].copy()
        obs *= 0.5
        assert np.array_equal(info["final_obs"], final)
        with_done += bool(info["done"].any())
        without_done += not info["done"].any()
    assert with_done >= 3 and without_done >= 3


def test_head_property_is_a_copy_of_the_head_cells():
    vec = make_vec(8, n_envs=3, seed=1)
    vec.step(np.array([STRAIGHT, TURN_LEFT, TURN_RIGHT]))  # heads now differ and have r != c
    head = vec.head
    assert head.dtype == np.int64 and head.tolist() == [[4, 5], [3, 4], [5, 4]]
    assert np.array_equal(head, head_positions(vec.body_age))
    head[:] = -5  # writing to the returned array must not reach the env
    assert np.array_equal(vec.head, head_positions(vec.body_age))
    vec.check_invariants()


# ------------------------------------------------------------- set_state
def test_set_state_matches_scalar_and_leaves_other_sub_envs_untouched():
    cfg = EnvConfig(W=10, obs=OBS_B8)
    vec = VecSnakeEnv(cfg, n_envs=4, seed=9)
    vec.reset()
    vec.set_state(2, STRAIGHT_RIGHT, food=(5, 6))
    vec.step(np.full(4, STRAIGHT))  # sub-env 2 eats: score and steps are now non-zero
    assert vec.score[2] == 1 and vec.steps[2] == 1
    before = state_snapshot(vec)

    vec.set_state(2, COIL_TAIL_A, food=(9, 9))
    twin = SnakeEnv(cfg)
    twin.reset(seed=0)
    want_obs = twin.set_state(COIL_TAIL_A, food=(9, 9))
    others = [0, 1, 3]
    for name, snapshot in before.items():
        assert np.array_equal(getattr(vec, name)[others], snapshot[others]), name
    assert np.array_equal(vec.body_age[2], twin.body_age) and vec.body_age.dtype == AGE_DTYPE
    assert vec.heading[2] == twin.heading == LEFT  # neck -> head
    assert tuple(vec.food[2]) == (9, 9)
    assert vec.length[2] == len(COIL_TAIL_A)
    assert vec.score[2] == 0 and vec.steps[2] == 0 and vec.steps_since_food[2] == 0
    assert np.array_equal(vec.observe()[2], want_obs)
    vec.check_invariants()


def test_set_state_heading_and_starvation_counter():
    vec = make_vec(10, n_envs=3)
    vec.set_state(0, BLOCK_TAIL_CHASE, FAR_FOOD)
    vec.set_state(1, BLOCK_OTHER_HEADING, FAR_FOOD)  # same cells and head, other way round
    vec.set_state(2, STRAIGHT_RIGHT, FAR_FOOD, heading=UP, steps_since_food=17)
    assert vec.heading.tolist() == [LEFT, DOWN, UP]
    assert vec.steps_since_food.tolist() == [0, 0, 17]
    assert vec.length.tolist() == [4, 4, 3]
    vec.check_invariants()
    # The explicit heading is what step() uses: straight now means up.
    vec.step(np.full(3, STRAIGHT))
    assert head_positions(vec.body_age).tolist() == [[5, 3], [6, 4], [4, 5]]
    assert vec.steps_since_food.tolist() == [1, 1, 18]


def test_set_state_then_step_matches_scalar():
    """Same cell set, same head, different tails: only the age order tells
    the two coils apart, and the batched step must respect it."""
    cfg = EnvConfig(W=10, obs=OBS_B8, reward=R2)
    seed = 13
    vec = VecSnakeEnv(cfg, n_envs=2, seed=seed)
    vec.reset()
    twins = []
    for i, cells in enumerate((COIL_TAIL_A, COIL_TAIL_B)):
        vec.set_state(i, cells, food=(8, 8))
        twins.append(twin_of(vec, i, seed=seed))
        twins[i].set_state(cells, food=(8, 8))
    for t, action in enumerate((STRAIGHT, TURN_LEFT, TURN_LEFT, STRAIGHT, TURN_RIGHT)):
        obs, reward, terminated, truncated, info = vec.step(np.full(2, action))
        for i, twin in enumerate(twins):
            s_obs, s_reward, s_term, s_trunc, s_info = twin.step(action)
            assert not s_term and not s_trunc
            assert np.array_equal(obs[i], s_obs) and reward[i] == np.float32(s_reward)
            assert np.array_equal(vec.body_age[i], twin.body_age)
        if t == 0:  # different cells vacated: (3, 4) in the first coil, (4, 5) in the second
            assert vec.body_age[0, 3, 4] == 0 and vec.body_age[0, 4, 5] > 0
            assert vec.body_age[1, 4, 5] == 0 and vec.body_age[1, 3, 4] > 0


def test_set_state_validation():
    vec = make_vec(10, n_envs=3, seed=4)
    before = state_snapshot(vec)
    with pytest.raises(ValueError):
        vec.set_state(1, [(0, 0), (0, 2)], food=(5, 5))  # not adjacent
    # The gap must be found by the pairwise check itself, not only as a side
    # effect of inferring the heading from a broken neck -> head pair.
    with pytest.raises(ValueError):
        vec.set_state(1, [(0, 0), (0, 2), (0, 3)], food=(5, 5))  # gap behind the neck
    with pytest.raises(ValueError):
        vec.set_state(1, [(0, 0), (0, 2)], food=(5, 5), heading=RIGHT)  # explicit heading, same gap
    with pytest.raises(ValueError):
        vec.set_state(1, [(0, 0), (0, 1), (0, 0)], food=(5, 5))  # cell listed twice
    with pytest.raises(ValueError):
        vec.set_state(1, STRAIGHT_RIGHT, food=(5, 4))  # food on the snake
    with pytest.raises(TypeError):
        vec.set_state(1, STRAIGHT_RIGHT, FAR_FOOD, steps_since_food=None)  # converted before any write
    for bad_index in (-1, 3, 99):
        with pytest.raises(IndexError):
            vec.set_state(bad_index, STRAIGHT_RIGHT, FAR_FOOD)
    # A rejected state changes nothing, as in the scalar env.
    for name, snapshot in before.items():
        assert np.array_equal(getattr(vec, name), snapshot), name
    vec.check_invariants()


def test_set_state_before_reset_raises():
    vec = VecSnakeEnv(EnvConfig(W=10), n_envs=2)
    with pytest.raises(RuntimeError):
        vec.set_state(0, STRAIGHT_RIGHT, FAR_FOOD)


# ------------------------------------------------------------- invariants
def test_check_invariants_passes_on_healthy_states():
    vec = make_vec(8, n_envs=6, seed=1, starve_limit=20)
    vec.check_invariants()
    vec.set_state(0, HOOK_SELF_COLLISION, FAR_FOOD)
    vec.set_state(1, BLOCK_TAIL_CHASE, FAR_FOOD)
    vec.set_state(2, COIL_TAIL_A, food=(7, 7), steps_since_food=19)
    vec.set_state(3, COIL_TAIL_B, food=(7, 7), heading=LEFT)
    cells, food = serpentine_almost_full(8)  # 63 of 64 cells; the survival policy wins on its first move
    vec.set_state(4, cells, food)
    vec.check_invariants()
    rng = np.random.default_rng(0)
    for _ in range(60):
        vec.step(survival_actions(vec, rng))
        vec.check_invariants()


def _empty_cell(vec: VecSnakeEnv, i: int) -> tuple[int, int]:
    """First empty cell of sub-env ``i`` that is neither food nor next to the snake."""
    occupied = vec.body_age[i] > 0
    near = occupied.copy()
    near[1:] |= occupied[:-1]
    near[:-1] |= occupied[1:]
    near[:, 1:] |= occupied[:, :-1]
    near[:, :-1] |= occupied[:, 1:]
    near[tuple(vec.food[i])] = True
    r, c = np.argwhere(~near)[0]
    return int(r), int(c)


def _corrupt_duplicate_age(vec: VecSnakeEnv, i: int) -> None:
    vec.body_age[i][_empty_cell(vec, i)] = 1  # a second tail


def _corrupt_missing_age(vec: VecSnakeEnv, i: int) -> None:
    neck = cells_from_body_age(vec.body_age[i])[-2]
    vec.body_age[i][neck] = 0


def _corrupt_age_above_length(vec: VecSnakeEnv, i: int) -> None:
    head = cells_from_body_age(vec.body_age[i])[-1]
    vec.body_age[i][head] = vec.length[i] + 1


def _corrupt_stray_cell(vec: VecSnakeEnv, i: int) -> None:
    vec.body_age[i][_empty_cell(vec, i)] = vec.length[i] + 1  # ages 1..length+1, length unchanged


def _corrupt_length(vec: VecSnakeEnv, i: int) -> None:
    vec.length[i] += 1


def _corrupt_chain(vec: VecSnakeEnv, i: int) -> None:
    tail, neck, _ = cells_from_body_age(vec.body_age[i])
    vec.body_age[i][tail], vec.body_age[i][neck] = 2, 1  # ages 2 and 3 are now two cells apart


def _corrupt_food_on_body(vec: VecSnakeEnv, i: int) -> None:
    vec.food[i] = cells_from_body_age(vec.body_age[i])[0]


def _corrupt_food_below_grid(vec: VecSnakeEnv, i: int) -> None:
    vec.food[i] = (vec.W, 0)


def _corrupt_food_left_of_grid(vec: VecSnakeEnv, i: int) -> None:
    vec.food[i] = (0, -1)


def _corrupt_heading_high(vec: VecSnakeEnv, i: int) -> None:
    vec.heading[i] = 4


def _corrupt_heading_negative(vec: VecSnakeEnv, i: int) -> None:
    vec.heading[i] = -1


def _corrupt_starvation_counter_at_limit(vec: VecSnakeEnv, i: int) -> None:
    vec.steps_since_food[i] = vec.H


def _corrupt_starvation_counter_negative(vec: VecSnakeEnv, i: int) -> None:
    vec.steps_since_food[i] = -1


CORRUPTIONS = [
    _corrupt_duplicate_age,
    _corrupt_missing_age,
    _corrupt_age_above_length,
    _corrupt_stray_cell,
    _corrupt_length,
    _corrupt_chain,
    _corrupt_food_on_body,
    _corrupt_food_below_grid,
    _corrupt_food_left_of_grid,
    _corrupt_heading_high,
    _corrupt_heading_negative,
    _corrupt_starvation_counter_at_limit,
    _corrupt_starvation_counter_negative,
]


@pytest.mark.parametrize("corrupt", CORRUPTIONS, ids=lambda f: f.__name__.removeprefix("_corrupt_"))
def test_check_invariants_detects_corruption(corrupt):
    """Each invariant of the spec, violated in one sub-env by writing to the
    public state arrays; the message must name that sub-env."""
    n, i = 40, 37  # 37 cannot be mistaken for a coordinate, an age or a count on an 8x8 board
    vec = make_vec(8, n_envs=n, seed=0, starve_limit=20)
    vec.check_invariants()
    corrupt(vec, i)
    with pytest.raises(AssertionError) as err:
        vec.check_invariants()
    assert re.search(rf"(?<!\d){i}(?!\d)", str(err.value)), f"message does not name sub-env {i}: {err.value}"


def test_debug_mode_runs_clean():
    cfg = EnvConfig(W=6, starve_limit=10, obs=OBS_B8, reward=R3)
    vec = VecSnakeEnv(cfg, n_envs=16, seed=0, debug=True)
    vec.reset()
    rng = np.random.default_rng(0)
    n_done = n_eaten = 0
    for t in range(300):
        score_before = vec.score.copy()
        # Mostly survival moves (eats, starvation), some uniformly random ones (wall deaths).
        actions = np.where(rng.random(16) < 0.2, rng.integers(vec.n_actions, size=16), survival_actions(vec, rng))
        _, _, _, _, info = vec.step(actions)
        n_done += int(info["done"].sum())
        n_eaten += int((info["score"] - score_before).sum())
        if t == 150:
            vec.reset()
        if t == 200:
            vec.reset(seed=9)
    assert n_done > 50 and n_eaten > 50


def test_debug_mode_checks_the_state_that_step_leaves_behind():
    vec = VecSnakeEnv(EnvConfig(W=8), n_envs=4, seed=0, debug=True)
    vec.reset()
    vec.step(np.full(4, STRAIGHT))
    vec.length[2] += 2  # the next head would be placed with age length + 2
    with pytest.raises(AssertionError):
        vec.step(np.full(4, TURN_LEFT))

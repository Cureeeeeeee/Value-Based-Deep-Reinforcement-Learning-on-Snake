"""Evaluation with value estimates (M7, spec D5): ``evaluate(..., gamma)`` and
``evaluate_batched(..., gamma)``, the four new ``EpisodeRecord`` fields,
``EvalResult.value_summary()`` and the two JSON forms.

The measurement of van Hasselt et al. 2016 (their Figure 3): for every step
``t`` of an evaluation episode, ``v_t = max_a q_values(s_t)[a]`` is what the
network claims for the state it acted from and ``G_t = r_t + gamma * G_{t+1}``
(``G_T = 0``, no bootstrap at a starvation truncation either) is what the
greedy policy then collected. Per episode: ``value_start = v_0``,
``return_start = G_0``, ``value_visited = mean_t v_t``, ``return_visited =
mean_t G_t``. ``value_summary()`` averages the start quantities over episodes
and the visited quantities over all visited states (episodes weighted by
their step counts).

Players: a random action-value table over the 2048 codes of observation A
(greedy: ``act_batch`` is the row-wise argmax), and an always-straight player
whose ``q_values`` are ``[1 + d, d, d - 1]`` with ``d`` the danger-straight
bit, so that ``v_t`` is known by hand on the straight path from the initial
position of an 8x8 board: 4 steps to the wall (rewards 0 0 0 -1 under R0
unless food lies on the way) or a starvation truncation with
``starve_limit = 3``. Seeds 0 (food off the path), 79 (food two cells ahead)
and 7 (food one cell ahead) were chosen by replaying the boards; the tests
re-check the premise before using it.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import FrozenInstanceError, fields, replace

import numpy as np
import pytest

import snake_rl.evaluate as evaluate_module
from snake_rl.env.obs import A_DANGER_STRAIGHT
from snake_rl.evaluate import DEFAULT_EVAL_SEED, EpisodeRecord, EvalResult, env_config_to_dict, evaluate, evaluate_batched
from tests.helpers import EnvConfig, SnakeEnv
from tests.test_dqn import randomise, small_agent
from tests.test_evaluate_batched import BIT_WEIGHTS, N_STATES_A, AlwaysStraight, ObsPolicy, TablePolicy

FAKE_GIT = "0000000+m7"
OLD_FIELDS = ("seed", "score", "steps", "length", "cause")
NEW_FIELDS = ("value_start", "return_start", "value_visited", "return_visited")
#: 8x8 boards for the straight player (see the module docstring).
CLEAR_SEED, FOOD_AHEAD2_SEED, FOOD_AHEAD1_SEED = 0, 79, 7
PROTOCOLS = [evaluate, evaluate_batched]
PROTOCOL_IDS = ["scalar", "batched"]


@pytest.fixture(autouse=True)
def _fast_git_hash(monkeypatch):
    monkeypatch.setattr(evaluate_module, "git_hash", lambda: FAKE_GIT)


# ------------------------------------------------------------- players
class QTablePolicy(ObsPolicy):
    """A random action-value table over the 2048 codes of observation A; greedy.
    ``act_batch`` does not go through ``q_values``, so a spy can count the calls of each."""

    def __init__(self, seed: int, n_actions: int = 3) -> None:
        self.name = f"qtable{n_actions}"
        self.table = np.random.default_rng(seed).normal(size=(N_STATES_A, n_actions)).astype(np.float32)

    def _lookup(self, obs: np.ndarray) -> np.ndarray:
        return self.table[np.asarray(obs).astype(np.int64) @ BIT_WEIGHTS]

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        return self._lookup(obs)

    def act_batch(self, obs: np.ndarray) -> np.ndarray:
        return self._lookup(obs).argmax(axis=1).astype(np.int64)


class SpyQTable(QTablePolicy):
    """Records every ``q_values`` / ``act_batch`` call with a copy of its observations."""

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self.events: list[tuple[str, np.ndarray]] = []

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        self.events.append(("q", np.array(obs)))
        return super().q_values(obs)

    def act_batch(self, obs: np.ndarray) -> np.ndarray:
        self.events.append(("act", np.array(obs)))
        return super().act_batch(obs)


class StraightWithQ(AlwaysStraight):
    """Always straight, with ``q_values = [1 + d, d, d - 1]`` (``d`` = danger-straight bit): the greedy
    action is straight too, and ``v_t = 1 + d(s_t)``."""

    name = "straight-q"

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        d = np.asarray(obs, dtype=np.float32)[:, A_DANGER_STRAIGHT : A_DANGER_STRAIGHT + 1]
        return np.concatenate([1.0 + d, d, d - 1.0], axis=1).astype(np.float32)


def straight_play(cfg: EnvConfig, seed: int) -> tuple[list[float], list[int], str]:
    """Replay of the straight player on the scalar env: (rewards, danger-straight bits, cause)."""
    env = SnakeEnv(cfg)
    obs, _ = env.reset(seed=seed)
    rewards, dangers = [], []
    while True:
        dangers.append(int(obs[A_DANGER_STRAIGHT]))
        obs, reward, terminated, truncated, info = env.step(0)
        rewards.append(float(reward))
        if terminated or truncated:
            return rewards, dangers, info["cause"]


def old(record: EpisodeRecord) -> tuple:
    return tuple(getattr(record, name) for name in OLD_FIELDS)


def new(record: EpisodeRecord) -> tuple:
    return tuple(getattr(record, name) for name in NEW_FIELDS)


# ------------------------------------------------------------- signatures and records
def test_signatures_take_gamma_last_with_default_none():
    for protocol in PROTOCOLS:
        params = inspect.signature(protocol).parameters
        assert list(params) == ["policy", "cfg", "n_episodes", "seed", "gamma"]
        assert params["gamma"].default is None and params["n_episodes"].default == 100
        assert params["seed"].default == DEFAULT_EVAL_SEED


def test_episode_record_gains_four_optional_fields_after_cause():
    assert [f.name for f in fields(EpisodeRecord)] == [*OLD_FIELDS, *NEW_FIELDS]
    record = EpisodeRecord(seed=1, score=0, steps=4, length=3, cause="wall")
    assert new(record) == (None, None, None, None)
    full = EpisodeRecord(seed=1, score=0, steps=4, length=3, cause="wall", value_start=1.0, return_start=-0.125,
                         value_visited=1.25, return_visited=-0.46875)
    assert new(full) == (1.0, -0.125, 1.25, -0.46875) and old(full) == old(record)
    with pytest.raises(FrozenInstanceError):
        full.value_start = 2.0


# ------------------------------------------------------------- the two protocols agree
@pytest.mark.parametrize(
    "W, starve_limit, n, seed, gamma",
    [(8, None, 20, 300, 0.9), (6, 4, 50, 310, 0.5), (10, None, 7, 320, 1.0), (8, 20, 30, 330, 0.0), (6, 12, 100, 340, 0.99)],
    ids=lambda v: str(v),
)
def test_scalar_and_batched_agree_with_gamma(W, starve_limit, n, seed, gamma):
    cfg = EnvConfig(W=W, starve_limit=starve_limit)
    policy = QTablePolicy(seed=seed)
    scalar = evaluate(policy, cfg, n, seed, gamma=gamma)
    batched = evaluate_batched(policy, cfg, n, seed, gamma=gamma)
    assert len(scalar.episodes) == len(batched.episodes) == n
    for s, b in zip(scalar.episodes, batched.episodes):
        assert old(s) == old(b)  # exactly, as without gamma
        assert all(v is not None and type(v) is float for v in new(s) + new(b))
        assert new(b) == pytest.approx(new(s), abs=1e-5)  # the rounding of a batched forward pass
    np.testing.assert_equal(scalar.summary(), batched.summary())
    assert batched.value_summary() == pytest.approx(scalar.value_summary(), abs=1e-5)
    assert set(scalar.value_summary()) == set(NEW_FIELDS)


def test_the_agreement_cases_cover_terminations_and_truncations():
    causes = set()
    for W, starve_limit, n, seed in ((8, None, 20, 300), (6, 4, 50, 310), (6, 12, 100, 340)):
        result = evaluate_batched(QTablePolicy(seed=seed), EnvConfig(W=W, starve_limit=starve_limit), n, seed, gamma=0.9)
        causes |= {e.cause for e in result.episodes}
        assert len({e.steps for e in result.episodes}) > 1  # sub-envs finish at different times and auto-reset
    assert {"wall", "starve"} <= causes


def test_straight_player_agrees_exactly_between_the_protocols():
    cfg = EnvConfig(W=8, starve_limit=3)
    scalar = evaluate(StraightWithQ(), cfg, 100, 0, gamma=0.5)
    batched = evaluate_batched(StraightWithQ(), cfg, 100, 0, gamma=0.5)
    assert scalar.episodes == batched.episodes  # exact arithmetic on both sides
    assert {e.cause for e in scalar.episodes} == {"wall", "starve"}  # boards with and without food on the way
    assert scalar.value_summary() == batched.value_summary()


def test_dqn_agent_values_agree_between_the_protocols_to_rounding():
    agent = small_agent(seed=1)
    randomise(agent.q_net, seed=1)
    cfg = EnvConfig(W=8, starve_limit=30)
    scalar = evaluate(agent, cfg, 10, DEFAULT_EVAL_SEED, gamma=0.9)
    batched = evaluate_batched(agent, cfg, 10, DEFAULT_EVAL_SEED, gamma=0.9)
    assert scalar.policy == batched.policy == "dqn" and len(scalar.episodes) == 10
    for s, b in zip(scalar.episodes, batched.episodes):
        assert old(s) == old(b)
        assert new(b) == pytest.approx(new(s), abs=1e-5)
    # v_0 is max_a Q(s_0) of the online network on the reset observation of the episode's board.
    env = SnakeEnv(cfg)
    for i, e in enumerate(scalar.episodes):
        obs, _ = env.reset(seed=DEFAULT_EVAL_SEED + i)
        assert e.value_start == pytest.approx(float(agent.q_values(obs[None])[0].max()), abs=1e-5)


def test_first_episode_values_do_not_depend_on_the_batch_size():
    cfg = EnvConfig(W=6, starve_limit=10)
    policy = QTablePolicy(seed=2)
    full = evaluate_batched(policy, cfg, 40, 500, gamma=0.8)
    assert evaluate_batched(policy, cfg, 1, 500, gamma=0.8).episodes == full.episodes[:1]
    assert evaluate_batched(policy, cfg, 7, 500, gamma=0.8).episodes == full.episodes[:7]


# ------------------------------------------------------------- hand-computed episodes
@pytest.mark.parametrize("protocol", PROTOCOLS, ids=PROTOCOL_IDS)
@pytest.mark.parametrize(
    "gamma, return_start, return_visited",
    [(0.5, -0.125, -0.46875), (1.0, -1.0, -1.0), (0.0, 0.0, -0.25), (0.9, -0.729, -(0.729 + 0.81 + 0.9 + 1.0) / 4)],
    ids=["gamma0.5", "gamma1", "gamma0", "gamma0.9"],
)
def test_hand_computed_values_and_returns_of_a_wall_death(protocol, gamma, return_start, return_visited):
    """Straight from the initial position of an 8x8 board, no food on the way: four steps, the fourth into
    the wall. r = [0, 0, 0, -1], d = [0, 0, 0, 1] so v = [1, 1, 1, 2]; G_3 = -1, G_t = gamma * G_{t+1}."""
    cfg = EnvConfig(W=8)
    assert straight_play(cfg, CLEAR_SEED) == ([0.0, 0.0, 0.0, -1.0], [0, 0, 0, 1], "wall")
    result = protocol(StraightWithQ(), cfg, 1, CLEAR_SEED, gamma=gamma)
    (e,) = result.episodes
    assert old(e) == (CLEAR_SEED, 0, 4, 3, "wall")
    assert e.value_start == 1.0 and e.value_visited == 1.25
    assert e.return_start == pytest.approx(return_start, abs=1e-12) and e.return_visited == pytest.approx(return_visited, abs=1e-12)
    assert all(type(v) is float for v in new(e))
    assert result.value_summary() == pytest.approx(
        {"value_start": 1.0, "return_start": return_start, "value_visited": 1.25, "return_visited": return_visited}, abs=1e-12)


@pytest.mark.parametrize("protocol", PROTOCOLS, ids=PROTOCOL_IDS)
def test_hand_computed_returns_with_food_on_the_way(protocol):
    """Food two cells ahead: r = [0, 1, 0, -1] at gamma 0.5 gives G = [0.375, 0.75, -0.5, -1]; the danger bit
    is still [0, 0, 0, 1] (food is not a danger), so v = [1, 1, 1, 2]."""
    cfg = EnvConfig(W=8)
    assert straight_play(cfg, FOOD_AHEAD2_SEED) == ([0.0, 1.0, 0.0, -1.0], [0, 0, 0, 1], "wall")
    (e,) = protocol(StraightWithQ(), cfg, 1, FOOD_AHEAD2_SEED, gamma=0.5).episodes
    assert old(e) == (FOOD_AHEAD2_SEED, 1, 4, 4, "wall")
    assert new(e) == (1.0, 0.375, 1.25, -0.09375)
    # Food one cell ahead: r = [1, 0, 0, -1] -> G = [0.875, -0.25, -0.5, -1].
    assert straight_play(cfg, FOOD_AHEAD1_SEED) == ([1.0, 0.0, 0.0, -1.0], [0, 0, 0, 1], "wall")
    (e,) = protocol(StraightWithQ(), cfg, 1, FOOD_AHEAD1_SEED, gamma=0.5).episodes
    assert old(e) == (FOOD_AHEAD1_SEED, 1, 4, 4, "wall")
    assert new(e) == (1.0, 0.875, 1.25, -0.21875)


@pytest.mark.parametrize("protocol", PROTOCOLS, ids=PROTOCOL_IDS)
@pytest.mark.parametrize("gamma", [0.5, 0.9, 1.0])
def test_a_starvation_truncation_is_a_realised_return_without_bootstrap(protocol, gamma):
    """``starve_limit = 3`` on the clear board: three steps with reward 0, then truncation. The realised
    returns are exactly 0 - a bootstrap from the successor (danger ahead, v = 2) would give
    G_2 = gamma * 2 instead."""
    cfg = EnvConfig(W=8, starve_limit=3)
    assert straight_play(cfg, CLEAR_SEED) == ([0.0, 0.0, 0.0], [0, 0, 0], "starve")
    (e,) = protocol(StraightWithQ(), cfg, 1, CLEAR_SEED, gamma=gamma).episodes
    assert old(e) == (CLEAR_SEED, 0, 3, 3, "starve")
    assert new(e) == (1.0, 0.0, 1.0, 0.0)
    assert e.return_visited != pytest.approx(gamma * 2.0 / 3, abs=1e-9)


# ------------------------------------------------------------- gamma = None: as before
def test_gamma_none_leaves_the_fields_none_and_the_json_byte_for_byte_as_before(tmp_path):
    policy = QTablePolicy(seed=3)
    cfg = EnvConfig(W=6, starve_limit=20)
    for protocol in PROTOCOLS:
        result = protocol(policy, cfg, 5, 40)
        assert all(v is None for e in result.episodes for v in new(e))
        assert result.value_summary() is None
        d = result.to_dict()
        assert list(d) == ["policy", "env", "seed", "git", "wall_time_s", "summary", "episodes"]
        assert all(list(e) == list(OLD_FIELDS) for e in d["episodes"])
        assert set(result.summary()) == {"policy", "n_episodes", "mean_score", "ci95", "median_score", "max_score",
                                         "mean_steps", "cause_fractions"}
        path = result.save(tmp_path / f"{protocol.__name__}.json")
        expected = {
            "policy": policy.name,
            "env": env_config_to_dict(cfg),
            "seed": 40,
            "git": FAKE_GIT,
            "wall_time_s": result.wall_time_s,
            "summary": result.summary(),
            "episodes": [{"seed": e.seed, "score": e.score, "steps": e.steps, "length": e.length, "cause": e.cause}
                         for e in result.episodes],
        }
        # The M6 file: ``json.dumps(to_dict(), indent=1)`` through ``Path.write_text`` (platform newlines).
        reference = tmp_path / f"{protocol.__name__}_expected.json"
        reference.write_text(json.dumps(expected, indent=1), encoding="utf-8")
        raw = path.read_bytes()
        assert raw == reference.read_bytes()
        assert b"value" not in raw
        loaded = EvalResult.load(path)
        assert loaded.episodes == result.episodes and loaded.to_dict() == result.to_dict()
        assert loaded.value_summary() is None
    assert evaluate(policy, cfg, 3, 40, gamma=None).episodes == evaluate(policy, cfg, 3, 40).episodes
    assert evaluate_batched(policy, cfg, 3, 40, gamma=None).episodes == evaluate_batched(policy, cfg, 3, 40).episodes


def test_q_values_is_never_consulted_without_gamma():
    cfg = EnvConfig(W=6, starve_limit=10)
    for protocol in PROTOCOLS:
        spy = SpyQTable(seed=1)
        result = protocol(spy, cfg, 4, 60)
        assert len(result.episodes) == 4
        assert spy.events and all(kind == "act" for kind, _ in spy.events)


# ------------------------------------------------------------- gamma given: the values key and the fields
def test_values_key_and_the_episode_fields_appear_only_with_gamma(tmp_path):
    policy = QTablePolicy(seed=3)
    cfg = EnvConfig(W=6, starve_limit=20)
    result = evaluate_batched(policy, cfg, 5, 40, gamma=0.9)
    d = result.to_dict()
    assert set(d) == {"policy", "env", "seed", "git", "wall_time_s", "summary", "episodes", "values"}
    assert d["values"] == result.value_summary() and list(d["values"]) == list(NEW_FIELDS)
    assert all(set(e) == {*OLD_FIELDS, *NEW_FIELDS} for e in d["episodes"])
    assert set(d["summary"]) == set(evaluate_batched(policy, cfg, 5, 40).summary())  # summary() unchanged
    path = result.save(tmp_path / "values.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["values"] == result.value_summary()
    loaded = EvalResult.load(path)
    assert loaded.episodes == result.episodes  # the floats round-trip exactly
    assert loaded.value_summary() == result.value_summary() and loaded.to_dict() == result.to_dict()
    # ``values`` is recomputed from the records, not read from the file.
    raw["values"] = {name: 999.0 for name in NEW_FIELDS}
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert EvalResult.load(path).to_dict()["values"] == result.value_summary()
    # An old file (no value keys at all) loads with None.
    for e in raw["episodes"]:
        for name in NEW_FIELDS:
            del e[name]
    del raw["values"]
    path.write_text(json.dumps(raw), encoding="utf-8")
    older = EvalResult.load(path)
    assert [old(e) for e in older.episodes] == [old(e) for e in result.episodes]
    assert all(v is None for e in older.episodes for v in new(e)) and older.value_summary() is None
    assert "values" not in older.to_dict()


def test_value_fields_are_plain_python_floats():
    result = evaluate_batched(QTablePolicy(seed=4), EnvConfig(W=6), 3, 7, gamma=0.9)
    for e in result.episodes:
        assert all(type(v) is float for v in new(e))
    assert all(type(v) is float for v in result.value_summary().values())
    json.dumps(result.to_dict())


# ------------------------------------------------------------- errors
def test_a_policy_without_q_values_is_a_type_error_naming_q_values():
    policy = TablePolicy(3, seed=7)  # act_batch only
    cfg = EnvConfig(W=6)
    for protocol in PROTOCOLS:
        with pytest.raises(TypeError, match="q_values"):
            protocol(policy, cfg, 3, 1, gamma=0.9)
        assert len(protocol(policy, cfg, 3, 1).episodes) == 3  # without gamma: as before


@pytest.mark.parametrize("gamma", [-0.01, 1.01, -1.0, 2.0])
def test_a_gamma_outside_the_unit_interval_is_a_value_error(gamma):
    for protocol in PROTOCOLS:
        with pytest.raises(ValueError):
            protocol(QTablePolicy(seed=1), EnvConfig(W=6), 2, 1, gamma=gamma)


def test_gamma_boundaries_are_accepted():
    for protocol in PROTOCOLS:
        for gamma in (0.0, 1.0):
            result = protocol(QTablePolicy(seed=1), EnvConfig(W=6, starve_limit=5), 2, 1, gamma=gamma)
            assert all(v is not None for e in result.episodes for v in new(e))


# ------------------------------------------------------------- value_summary arithmetic
def make_record(seed: int, steps: int, values: tuple[float, float, float, float] | None) -> EpisodeRecord:
    record = EpisodeRecord(seed=seed, score=0, steps=steps, length=3, cause="wall")
    return record if values is None else replace(record, **dict(zip(NEW_FIELDS, values)))


def test_value_summary_arithmetic_weights_the_visited_quantities_by_the_steps():
    env = env_config_to_dict(EnvConfig(W=6))
    # (value_start, return_start, value_visited, return_visited); steps 1 and 3.
    episodes = [make_record(0, 1, (3.0, 0.5, 2.0, 1.0)), make_record(1, 3, (1.0, 1.5, 4.0, -1.0))]
    summary = EvalResult(policy="x", env=env, seed=0, episodes=episodes).value_summary()
    assert list(summary) == list(NEW_FIELDS)
    # start: plain means over episodes; visited: (2 * 1 + 4 * 3) / 4 and (1 * 1 - 1 * 3) / 4.
    assert summary == {"value_start": 2.0, "return_start": 1.0, "value_visited": 3.5, "return_visited": -0.5}
    assert all(type(v) is float for v in summary.values())
    # Three episodes with steps 1, 1, 2: (1 + 2 + 2 * 4) / 4.
    episodes = [make_record(0, 1, (0.0, 0.0, 1.0, 0.0)), make_record(1, 1, (0.0, 0.0, 2.0, 0.0)),
                make_record(2, 2, (0.0, 0.0, 4.0, 0.0))]
    assert EvalResult(policy="x", env=env, seed=0, episodes=episodes).value_summary()["value_visited"] == 2.75
    # None: no episodes, or any episode without the fields.
    assert EvalResult(policy="x", env=env, seed=0, episodes=[]).value_summary() is None
    assert EvalResult(policy="x", env=env, seed=0, episodes=[episodes[0], make_record(3, 2, None)]).value_summary() is None
    assert EvalResult(policy="x", env=env, seed=0, episodes=[make_record(3, 2, None)]).value_summary() is None


def test_value_summary_of_a_real_evaluation_follows_the_formula():
    result = evaluate_batched(QTablePolicy(seed=9), EnvConfig(W=6, starve_limit=8), 30, 90, gamma=0.9)
    steps = np.array([e.steps for e in result.episodes], dtype=np.float64)
    assert len(set(steps.tolist())) > 1  # the weighting matters
    want = {
        "value_start": float(np.mean([e.value_start for e in result.episodes])),
        "return_start": float(np.mean([e.return_start for e in result.episodes])),
        "value_visited": float(np.sum([e.value_visited * e.steps for e in result.episodes]) / steps.sum()),
        "return_visited": float(np.sum([e.return_visited * e.steps for e in result.episodes]) / steps.sum()),
    }
    assert result.value_summary() == pytest.approx(want, rel=1e-12)
    unweighted = float(np.mean([e.value_visited for e in result.episodes]))
    assert result.value_summary()["value_visited"] != pytest.approx(unweighted, rel=1e-9)


# ------------------------------------------------------------- the acting protocol: q_values first
def test_batched_protocol_asks_q_values_once_per_vec_step_before_act_batch_on_the_same_batch():
    cfg = EnvConfig(W=6, starve_limit=20)
    n = 6
    spy = SpyQTable(seed=5)
    result = evaluate_batched(spy, cfg, n, 50, gamma=0.9)
    n_vec_steps = max(e.steps for e in result.episodes)
    assert [kind for kind, _ in spy.events] == ["q", "act"] * n_vec_steps
    for (_, q_obs), (_, a_obs) in zip(spy.events[::2], spy.events[1::2]):
        assert q_obs.shape == a_obs.shape == (n, 11) and np.array_equal(q_obs, a_obs)
    # The scalar protocol: q_values(obs[None]) once per step, on the observation the action is chosen from.
    spy = SpyQTable(seed=5)
    result = evaluate(spy, cfg, n, 50, gamma=0.9)
    assert len(spy.events) == 2 * sum(e.steps for e in result.episodes)
    for (kind_a, obs_a), (kind_b, obs_b) in zip(spy.events[::2], spy.events[1::2]):
        assert sorted((kind_a, kind_b)) == ["act", "q"]
        assert obs_a.shape == obs_b.shape == (1, 11) and np.array_equal(obs_a, obs_b)

"""Behaviour analysis (M8, spec section 3): ``snake_rl/analysis/behaviour.py``.

Rollouts are played on small boards with scripted policies whose outcome is
known by hand: ``AlwaysStraight`` (from the initial position of a 6x6 board
the wall is three moves away), ``LeftTurner`` (a left turn every step traces
the 2x2 square (3,3) (2,3) (2,2) (3,2) with heading right / up / left /
down, so the ``(row, col, heading)`` states repeat with period 4 from the
first state on and the episode is starved after ``H`` steps), the ``bfs``
baseline for episodes that eat, and a ``DQNAgent`` with random weights for
``q_values``. The statistics are checked on ``Trajectory`` objects built by
hand from state lists, with every expected number written out here.

Run directories for the checkpoint selection are written by hand:
``config.yaml`` from ``configs/smoke_8x8.yaml`` (validated through
``load_config``) and tiny ``DQNAgent`` checkpoints; no training.
"""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Rectangle

from snake_rl.agents.baselines import make_policy
from snake_rl.agents.dqn.agent import DQNAgent
from snake_rl.agents.dqn.config import DQNConfig
from snake_rl.analysis.behaviour import (
    CHECKPOINT_ALIASES,
    DEFAULT_EPISODES,
    DEFAULT_SEED,
    MIN_LOOP_STEPS,
    Trajectory,
    analyse_player,
    border_fraction,
    checkpoint_files,
    episode_table,
    hacking_candidates,
    heatmap_figure,
    last_steps_figure,
    loop_stats,
    plot_heatmaps,
    plot_last_steps,
    resolve_checkpoint,
    rollout,
    rollouts,
    summarise,
    visit_heatmap,
)
from snake_rl.config import config_to_dict, load_config
from snake_rl.env.common import DOWN, LEFT, RIGHT, UP
from snake_rl.env.obs import N_FEATURES_A
from snake_rl.evaluate import DEFAULT_EVAL_SEED, env_config_from_dict, env_config_to_dict, evaluate
from snake_rl.gui import app as gui_app
from snake_rl.gui.app import GameSession, Recording
from snake_rl.utils import REPO_ROOT, git_hash
from tests.helpers import STRAIGHT, TURN_LEFT, EnvConfig, RewardConfig, SnakeEnv
from tests.test_dqn import randomise, small_agent
from tests.test_evaluate_batched import AlwaysStraight, FoodSeekerA
from tests.test_plots import png_size

#: Spec section 0 names the GUI's replay player ``ReplayPlayer``; the class in ``gui/app.py`` is ``RecordedPlayer``.
ReplayPlayer = getattr(gui_app, "ReplayPlayer", None) or gui_app.RecordedPlayer

ENV6 = env_config_to_dict(EnvConfig(W=6))
ENV4 = env_config_to_dict(EnvConfig(W=4))
TABLE_COLUMNS = ("seed", "score", "steps", "length", "cause", "truncated", "cut", "tail_without_food", "period",
                 "loop_steps", "shaping_reward", "border_fraction", "q_max_mean")
SUMMARY_KEYS = {
    "n_episodes", "score_mean", "steps_mean", "length_mean", "frac_wall", "frac_self", "frac_starve", "frac_win",
    "frac_truncated", "frac_cut", "tail_without_food_mean", "loop_fraction", "loop_steps_mean", "shaping_reward_mean",
    "border_fraction_mean", "q_max_mean",
}
LOOP_KEYS = {"tail_without_food", "period", "loop_steps", "food_steps", "n_food", "truncated", "cut"}
#: The 2x2 square the left-turning snake runs round on a 6x6 board, as (head, heading) states in step order.
A, B, C, D = ((3, 3), RIGHT), ((2, 3), UP), ((2, 2), LEFT), ((3, 2), DOWN)
SQUARE = (A, B, C, D)
SQUARE_CELLS = {rc for rc, _ in SQUARE}


# ------------------------------------------------------------------- policies
class LeftTurner:
    """Turns left every step; counts the ``reset`` calls."""

    name = "left"

    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def act(self, obs: np.ndarray, env: SnakeEnv | None = None) -> int:
        return TURN_LEFT


class SpyStraight:
    """Always straight, with action values that depend on the observation and a log of every call."""

    name = "spy"

    def __init__(self) -> None:
        self.events: list[tuple[str, np.ndarray | None]] = []
        self.weights = np.random.default_rng(5).normal(size=(N_FEATURES_A, 3))

    def reset(self) -> None:
        self.events.append(("reset", None))

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64)
        self.events.append(("q", obs.copy()))
        return obs @ self.weights

    def act(self, obs: np.ndarray, env: SnakeEnv | None = None) -> int:
        self.events.append(("act", np.array(obs, dtype=np.float64)))
        return STRAIGHT


class NoReset:
    """The bare policy protocol: a name and ``act``."""

    name = "noreset"

    def act(self, obs: np.ndarray, env: SnakeEnv | None = None) -> int:
        return STRAIGHT


def value_agent() -> DQNAgent:
    """A DQN agent with random weights on observation A (``q_values`` and greedy actions)."""
    agent = small_agent(seed=1)
    randomise(agent.q_net, seed=1)
    return agent


def tiny_agent(seed: int) -> DQNAgent:
    """The smallest MLP agent, with weights that differ from seed to seed."""
    agent = DQNAgent((N_FEATURES_A,), 3, DQNConfig(hidden=(8,), device="cpu"), seed=seed)
    randomise(agent.q_net, seed=seed)
    return agent


def cfg6(**kwargs) -> EnvConfig:
    return EnvConfig(W=6, **kwargs)


# --------------------------------------------------------------- trajectories
def make_traj(states, *, scores=None, rewards=None, cause: str = "starve", seed: int = 0, q_values=None,
              cut: bool = False, env: dict | None = None, player: str = "hand") -> Trajectory:
    """A ``Trajectory`` from the ``(head, heading)`` states ``s_0 .. s_T``; food off the board's paths at (0, 0)."""
    T = len(states) - 1
    heads = np.array([rc for rc, _ in states], dtype=np.int64).reshape(T + 1, 2)
    headings = np.array([h for _, h in states], dtype=np.int64)
    scores = np.zeros(T + 1, dtype=np.int64) if scores is None else np.asarray(scores, dtype=np.int64)
    rewards = np.zeros(T, dtype=np.float64) if rewards is None else np.asarray(rewards, dtype=np.float64)
    assert scores.shape == (T + 1,) and rewards.shape == (T,)
    score = int(scores[-1])
    return Trajectory(
        env=ENV6 if env is None else env, seed=seed, player=player, heads=heads, headings=headings,
        foods=np.zeros((T + 1, 2), dtype=np.int64), scores=scores, actions=np.full(T, STRAIGHT, dtype=np.int64),
        rewards=rewards, q_values=q_values, final_cells=[tuple(int(v) for v in heads[-1])], score=score, steps=T,
        length=3 + score, cause=cause, truncated=cause == "starve", cut=cut,
    )


def cycle_states(n: int, phase: int = 0) -> list:
    """``n`` consecutive states of the square cycle, starting at ``SQUARE[phase]``."""
    return [SQUARE[(phase + i) % 4] for i in range(n)]


def walk_states(n: int, row: int = 0, col: int = 0) -> list:
    """``n`` states of a straight walk to the right along ``row``."""
    return [((row, col + i), RIGHT) for i in range(n)]


def three_trajectories():
    """Starve with a 21-state loop (interior), wall after a border walk, cut with a 13-state loop."""
    q = np.random.default_rng(0).normal(size=(20, 3))
    a = make_traj(cycle_states(21), seed=3, q_values=q)
    b = make_traj(walk_states(6), seed=1, cause="wall", rewards=[0.0, 0.0, 0.0, 0.0, -1.0])
    c = make_traj(cycle_states(13), seed=2, cause="none", cut=True)
    return a, b, c, q


# ---------------------------------------------------------------- run layout
def write_dqn_run(exp_dir: Path, seed: int, *, checkpoint_steps=(), total_env_steps: int = 1000,
                  eval_every: int = 100, W: int = 6, starve_limit: int = 20, agent_seed: int | None = None,
                  model: bool = True) -> Path:
    """``<exp_dir>/<seed>/`` the way ``snake_rl.train`` lays it out, without training: a validated
    ``config.yaml`` (plus a ``run`` block), ``model.pt`` and ``checkpoints/step_XXXXXXXX.pt`` files of
    tiny agents whose weights differ from file to file."""
    overrides = [
        f"exp={exp_dir.name}", f"env.W={W}", f"env.starve_limit={starve_limit}", "agent.hidden=[8]",
        "agent.device=cpu", "agent.batch_size=64", "agent.buffer_size=1000", "agent.learning_starts=64",
        "train.n_envs=10", f"train.eval_every={eval_every}", f"train.total_env_steps={total_env_steps}",
        "train.eval_episodes=2",
    ]
    cfg = load_config(REPO_ROOT / "configs" / "smoke_8x8.yaml", overrides)
    run_dir = exp_dir / str(seed)
    run_dir.mkdir(parents=True)
    doc = config_to_dict(cfg)
    doc["run"] = {"seed": seed, "env_seed": 100_000 * (seed + 1), "git": "abc1234", "argv": []}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    base = 50 * seed + 1 if agent_seed is None else agent_seed
    if model:
        tiny_agent(base).save(run_dir / "model.pt")
    for k, step in enumerate(checkpoint_steps):
        tiny_agent(base + 1 + k).save(run_dir / "checkpoints" / f"step_{step:08d}.pt", include_optimizer=False)
    return run_dir


# ------------------------------------------------------------- figure helpers
def axes_title(ax) -> str:
    """The title of an axes wherever it is aligned (matplotlib keeps left, centre and right titles apart)."""
    titles = [ax.get_title(loc=loc) for loc in ("left", "center", "right")]
    found = [t for t in titles if t]
    assert len(found) <= 1, titles
    return found[0] if found else ""


def assert_no_tick_labels(ax) -> None:
    if not ax.axison:
        return
    labels = list(ax.get_xticklabels()) + list(ax.get_yticklabels())
    assert all(t.get_text() == "" or not t.get_visible() for t in labels), [t.get_text() for t in labels]


def board_offset(ax) -> float:
    """0 when cells are centred on integers (limits -0.5 .. W - 0.5), 0.5 when the board spans 0 .. W."""
    return float(min(ax.get_xlim())) + 0.5


def assert_board_limits(ax, W: int) -> None:
    x_lo, x_hi = sorted(float(v) for v in ax.get_xlim())
    y0, y1 = (float(v) for v in ax.get_ylim())
    assert x_lo in (-0.5, 0.0) and x_hi - x_lo == pytest.approx(W)
    assert y0 > y1  # row 0 at the top
    assert min(y0, y1) == x_lo and y0 - y1 == pytest.approx(W)


def polyline_at(ax, heads):
    """The line of ``ax`` whose vertices are exactly the ``(row, col)`` cells ``heads``, or None."""
    heads = np.asarray(heads, dtype=np.float64)
    offset = board_offset(ax)
    for line in ax.lines:
        x = np.asarray(line.get_xdata(), dtype=np.float64) - offset
        y = np.asarray(line.get_ydata(), dtype=np.float64) - offset
        if len(x) == len(heads) and np.allclose(x, heads[:, 1]) and np.allclose(y, heads[:, 0]):
            return line
    return None


def marker_at(ax, cell) -> bool:
    """A point marker, a collection offset or a non-rectangular patch centred on ``cell``."""
    r, c = cell
    offset = board_offset(ax)
    target = np.array([c + offset, r + offset], dtype=np.float64)

    def close(xy) -> bool:
        return bool(np.allclose(np.asarray(xy, dtype=np.float64), target, atol=1e-9))

    for line in ax.lines:
        x, y = np.asarray(line.get_xdata(), dtype=np.float64), np.asarray(line.get_ydata(), dtype=np.float64)
        if len(x) == 1 and close((x[0], y[0])):
            return True
    for coll in ax.collections:
        offsets = np.asarray(coll.get_offsets(), dtype=np.float64)
        if offsets.ndim == 2 and any(close(o) for o in offsets):
            return True
    for patch in ax.patches:
        if isinstance(patch, Rectangle):
            continue  # the snake's squares
        centre = getattr(patch, "center", None)
        if centre is not None and close(centre):
            return True
    return False


# ----------------------------------------------------- constants and signatures
def test_constants_and_signatures():
    assert DEFAULT_EPISODES == 100 and DEFAULT_SEED == DEFAULT_EVAL_SEED == 10_000
    assert CHECKPOINT_ALIASES == ("early", "mid", "late", "final") and MIN_LOOP_STEPS == 20
    params = inspect.signature(rollout).parameters
    assert list(params) == ["policy", "cfg", "seed", "max_steps"] and params["max_steps"].default is None
    params = inspect.signature(rollouts).parameters
    assert list(params) == ["policy", "cfg", "n_episodes", "seed", "max_steps"]
    assert (params["n_episodes"].default, params["seed"].default, params["max_steps"].default) == (100, 10_000, None)
    params = inspect.signature(analyse_player).parameters
    assert list(params) == ["policy", "cfg", "out_dir", "n_episodes", "seed", "max_steps", "record_top", "k_last", "meta"]
    assert all(params[name].kind is inspect.Parameter.KEYWORD_ONLY
               for name in ("n_episodes", "seed", "max_steps", "record_top", "k_last", "meta"))
    assert [params[name].default for name in ("n_episodes", "seed", "max_steps", "record_top", "k_last", "meta")] == [
        100, 10_000, None, 3, 10, None]
    assert inspect.signature(hacking_candidates).parameters["min_loop_steps"].default == MIN_LOOP_STEPS
    assert inspect.signature(last_steps_figure).parameters["k"].default == 10
    assert [f for f in Trajectory.__dataclass_fields__] == [
        "env", "seed", "player", "heads", "headings", "foods", "scores", "actions", "rewards", "q_values",
        "final_cells", "score", "steps", "length", "cause", "truncated", "cut"]


# ------------------------------------------------------------------- rollout
def test_rollout_arrays_dtypes_and_the_reset_state():
    cfg = cfg6(starve_limit=12)
    traj = rollout(LeftTurner(), cfg, seed=10_000)
    assert isinstance(traj, Trajectory)
    T = traj.steps
    assert T == 12 == len(traj.actions)  # starved after H = 12 steps without food
    assert traj.heads.shape == (T + 1, 2) and traj.heads.dtype == np.int64
    assert traj.headings.shape == (T + 1,) and traj.headings.dtype == np.int64
    assert traj.foods.shape == (T + 1, 2) and traj.foods.dtype == np.int64
    assert traj.scores.shape == (T + 1,) and traj.scores.dtype == np.int64
    assert traj.actions.shape == (T,) and traj.actions.dtype == np.int64
    assert traj.rewards.shape == (T,) and traj.rewards.dtype == np.float64
    assert traj.q_values is None
    env = SnakeEnv(cfg)
    env.reset(seed=10_000)
    assert tuple(traj.heads[0]) == env.head == (3, 3) and traj.headings[0] == env.heading == RIGHT
    assert tuple(traj.foods[0]) == env.food and traj.scores[0] == 0
    assert traj.env == env_config_to_dict(cfg) and traj.seed == 10_000 and traj.player == "left"
    assert (traj.cause, bool(traj.truncated), bool(traj.cut)) == ("starve", True, False)
    assert traj.score == int(traj.scores[-1]) and traj.length == 3 + traj.score
    assert traj.actions.tolist() == [TURN_LEFT] * T
    assert type(traj.score) is int and type(traj.steps) is int and type(traj.length) is int and type(traj.cause) is str
    # the left turn traces the square: states repeat every four steps from s_0 on
    assert traj.heads[:5].tolist() == [[3, 3], [2, 3], [2, 2], [3, 2], [3, 3]]
    assert traj.headings[:5].tolist() == [RIGHT, UP, LEFT, DOWN, RIGHT]
    # final_cells: the board after the last step, head first (spec 3.1; ``env.snake_cells()`` lists tail -> head)
    for action in traj.actions.tolist():
        env.step(action)
    assert traj.final_cells[0] == tuple(traj.heads[-1]) == env.head, "final_cells must start with the head"
    assert traj.final_cells == list(reversed(env.snake_cells())) and len(traj.final_cells) == traj.length
    assert all(type(r) is int and type(c) is int for r, c in traj.final_cells)


def test_rollout_replays_the_environment_step_by_step():
    cfg = cfg6()
    traj = rollout(make_policy("bfs"), cfg, seed=10_000)
    assert traj.steps > 20 and traj.score > 3 and traj.cause in ("wall", "self")  # a long, eating episode
    env = SnakeEnv(cfg)
    env.reset(seed=10_000)
    assert tuple(traj.heads[0]) == env.head and traj.headings[0] == env.heading and tuple(traj.foods[0]) == env.food
    for t, action in enumerate(traj.actions.tolist()):
        _, reward, terminated, truncated, info = env.step(action)
        assert tuple(traj.heads[t + 1]) == env.head, t  # the head after action t
        assert traj.headings[t + 1] == env.heading and tuple(traj.foods[t + 1]) == env.food, t
        assert traj.scores[t + 1] == env.score, t
        assert traj.rewards[t] == reward, t  # the env's reward, exactly
        assert (terminated or truncated) == (t == traj.steps - 1)
    assert (info["score"], info["steps"], info["length"], info["cause"]) == (traj.score, traj.steps, traj.length, traj.cause)
    assert bool(traj.truncated) == truncated and not traj.cut
    assert traj.final_cells == list(reversed(env.snake_cells())), "final_cells: head first (spec 3.1)"
    assert traj.scores[-1] == traj.score and (np.diff(traj.scores) >= 0).all() and traj.scores.max() == traj.score
    assert set(traj.rewards.tolist()) <= {-1.0, 0.0, 1.0}  # R0


def test_q_values_are_recorded_before_act_and_none_without_the_method():
    cfg = cfg6(starve_limit=5)
    spy = SpyStraight()
    traj = rollout(spy, cfg, seed=10_001)
    T = traj.steps
    assert T == 3 and traj.cause == "wall"  # straight from (3, 3): (3, 4), (3, 5), then the wall
    assert traj.q_values is not None and traj.q_values.shape == (T, 3) and traj.q_values.dtype == np.float64
    assert [kind for kind, _ in spy.events] == ["reset"] + ["q", "act"] * T
    env = SnakeEnv(cfg)
    obs, _ = env.reset(seed=10_001)
    for t in range(T):
        (_, q_obs), (_, act_obs) = spy.events[1 + 2 * t], spy.events[2 + 2 * t]
        assert q_obs.shape == (1, N_FEATURES_A) and np.array_equal(q_obs[0], act_obs)  # the same observation
        assert np.array_equal(q_obs[0], obs)  # the observation the action is chosen from
        assert np.array_equal(traj.q_values[t], (q_obs @ spy.weights)[0])
        obs, *_ = env.step(STRAIGHT)
    assert rollout(AlwaysStraight(), cfg, seed=10_001).q_values is None
    plain = rollout(NoReset(), cfg, seed=10_001)  # no reset method: optional
    assert plain.q_values is None and plain.steps == 3 and plain.player == "noreset"


def test_policy_reset_is_called_once_per_episode():
    cfg = cfg6(starve_limit=5)
    policy = LeftTurner()
    rollout(policy, cfg, seed=10_000)
    assert policy.resets == 1
    policy = LeftTurner()
    trajs = rollouts(policy, cfg, n_episodes=4, seed=10_000)
    assert policy.resets == 4 and len(trajs) == 4


def test_max_steps_cuts_the_rollout_and_is_validated():
    cfg = cfg6()
    bfs = make_policy("bfs")
    full = rollout(bfs, cfg, seed=10_000)
    T = full.steps
    assert T > 10 and not full.cut
    cut = rollout(bfs, cfg, seed=10_000, max_steps=5)
    assert cut.steps == 5 and cut.cut and cut.cause == "none" and not cut.truncated
    assert cut.heads.shape == (6, 2) and cut.actions.shape == (5,) and cut.rewards.shape == (5,)
    assert np.array_equal(cut.heads, full.heads[:6]) and np.array_equal(cut.actions, full.actions[:5])
    assert np.array_equal(cut.rewards, full.rewards[:5]) and np.array_equal(cut.scores, full.scores[:6])
    assert cut.score == int(cut.scores[-1]) and cut.length == 3 + cut.score
    assert cut.final_cells[0] == tuple(cut.heads[-1]), "final_cells: head first (spec 3.1)"
    assert len(cut.final_cells) == cut.length
    one = rollout(bfs, cfg, seed=10_000, max_steps=1)
    assert one.steps == 1 and one.cut and one.heads.shape == (2, 2)
    big = rollout(bfs, cfg, seed=10_000, max_steps=10_000)
    assert big.steps == T and not big.cut and big.cause == full.cause and np.array_equal(big.heads, full.heads)
    exact = rollout(bfs, cfg, seed=10_000, max_steps=T)  # the episode ends on the capped step: its real cause
    assert exact.steps == T and exact.cause == full.cause
    for bad in (0, -1, -10):
        with pytest.raises(ValueError):
            rollout(bfs, cfg, seed=10_000, max_steps=bad)
        with pytest.raises(ValueError):
            rollouts(bfs, cfg, 2, 10_000, max_steps=bad)
    trajs = rollouts(bfs, cfg, 3, 10_000, max_steps=5)
    assert [t.steps for t in trajs] == [5, 5, 5] and all(t.cut and t.cause == "none" for t in trajs)


@pytest.mark.parametrize(
    "make, cfg, n, seed",
    [
        (lambda: make_policy("bfs"), EnvConfig(W=6), 3, 10_000),
        (FoodSeekerA, EnvConfig(W=6, starve_limit=30), 5, 2_000),
        (AlwaysStraight, EnvConfig(W=8), 4, 500),
        (LeftTurner, EnvConfig(W=6, starve_limit=12), 3, 10_000),
        (value_agent, EnvConfig(W=8, starve_limit=30), 5, DEFAULT_EVAL_SEED),
    ],
    ids=["bfs", "seeker", "straight", "left", "dqn"],
)
def test_rollouts_equal_evaluate_episode_by_episode(make, cfg, n, seed):
    trajs = rollouts(make(), cfg, n, seed)
    result = evaluate(make(), cfg, n, seed)
    assert len(trajs) == n and all(isinstance(t, Trajectory) for t in trajs)
    assert [(t.score, t.steps, t.length, t.cause) for t in trajs] == [
        (e.score, e.steps, e.length, e.cause) for e in result.episodes]
    assert [t.seed for t in trajs] == [e.seed for e in result.episodes] == list(range(seed, seed + n))
    assert all(t.steps == len(t.actions) == len(t.heads) - 1 and not t.cut for t in trajs)
    assert all(t.env == result.env and t.player == result.policy for t in trajs)
    assert all((t.cause == "starve") == bool(t.truncated) for t in trajs)
    # episode i alone is episode i of the batch
    single = rollout(make(), cfg, seed + n - 1)
    assert (single.score, single.steps, single.cause) == (trajs[-1].score, trajs[-1].steps, trajs[-1].cause)


# -------------------------------------------------------------- to_recording
def test_to_recording_round_trip_resimulation_and_replay(tmp_path):
    cfg = cfg6()
    traj = rollout(make_policy("bfs"), cfg, seed=10_000)
    rec = traj.to_recording()
    assert isinstance(rec, Recording)
    assert rec.env == traj.env and rec.seed == 10_000 and rec.player == "bfs"
    assert rec.actions == traj.actions.tolist() and all(type(a) is int for a in rec.actions)
    assert rec.q_values is None
    assert (rec.score, rec.steps, rec.cause) == (traj.score, traj.steps, traj.cause)
    assert all(type(v) is int for v in (rec.seed, rec.score, rec.steps)) and type(rec.cause) is str
    path = rec.save(tmp_path / "recordings" / "bfs.json")
    loaded = Recording.load(path)
    assert loaded == rec and json.loads(path.read_text(encoding="utf-8"))["actions"] == rec.actions
    # re-simulation: the environment is a function of the seed and the actions
    env = SnakeEnv(env_config_from_dict(loaded.env))
    env.reset(seed=loaded.seed)
    for action in loaded.actions:
        _, _, terminated, truncated, info = env.step(action)
    assert terminated or truncated
    assert (info["score"], info["steps"], info["cause"]) == (rec.score, rec.steps, rec.cause)
    # the GUI's replay player hands out every recorded action
    player = ReplayPlayer(loaded)
    env = SnakeEnv(loaded.env_config())
    obs, _ = env.reset(seed=loaded.seed)
    if hasattr(player, "reset"):
        player.reset()
    played = []
    while True:
        action = int(player.act(obs, env))
        played.append(action)
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    assert played == rec.actions and (info["score"], info["steps"], info["cause"]) == (rec.score, rec.steps, rec.cause)
    # and the GUI session in replay mode reproduces the episode
    session = GameSession(loaded.env_config(), player=ReplayPlayer(loaded), seed=loaded.seed, mode="replay")
    for _ in range(rec.steps):
        session.agent_step()
    assert session.done and (session.score, session.steps, session.cause) == (rec.score, rec.steps, rec.cause)


def test_to_recording_q_values_are_a_list_for_a_dqn_and_none_otherwise(tmp_path):
    agent = value_agent()
    traj = rollout(agent, EnvConfig(W=8, starve_limit=30), seed=DEFAULT_EVAL_SEED)
    assert traj.q_values is not None and traj.q_values.shape == (traj.steps, 3)
    rec = traj.to_recording()
    assert rec.player == "dqn" and isinstance(rec.q_values, list) and len(rec.q_values) == traj.steps
    assert all(isinstance(row, list) and len(row) == 3 and all(type(v) is float for v in row) for row in rec.q_values)
    assert np.array_equal(np.array(rec.q_values), traj.q_values)
    path = rec.save(tmp_path / "dqn.json")
    assert Recording.load(path) == rec
    # a truncated episode replays too
    env = SnakeEnv(rec.env_config())
    env.reset(seed=rec.seed)
    for action in rec.actions:
        _, _, terminated, truncated, info = env.step(action)
    assert (terminated or truncated) and (info["score"], info["steps"], info["cause"]) == (rec.score, rec.steps, rec.cause)
    assert rollout(LeftTurner(), cfg6(starve_limit=5), seed=1).to_recording().q_values is None


# -------------------------------------------------- visit_heatmap, border_fraction
def test_visit_heatmap_manual_counts_and_normalisation():
    a = make_traj([((0, 0), RIGHT), ((0, 1), RIGHT), ((0, 2), RIGHT)], env=ENV4)  # T = 2: three counts
    b = make_traj([((0, 0), DOWN), ((1, 0), DOWN)], env=ENV4)  # T = 1: two counts
    heat = visit_heatmap([a, b], 4)
    assert heat.shape == (4, 4) and heat.dtype == np.float64
    expected = np.zeros((4, 4))
    expected[0, 0], expected[0, 1], expected[0, 2], expected[1, 0] = 2 / 5, 1 / 5, 1 / 5, 1 / 5
    np.testing.assert_allclose(heat, expected, rtol=1e-12)
    assert abs(heat.sum() - 1.0) < 1e-12
    np.testing.assert_allclose(visit_heatmap([a, a], 4), visit_heatmap([a], 4), rtol=1e-12)  # normalised
    alone = np.zeros((4, 4))
    alone[0, :3] = 1 / 3
    np.testing.assert_allclose(visit_heatmap((a,), 4), alone, rtol=1e-12)  # a tuple is a sequence too
    # a trajectory of T steps contributes T + 1 counts, the initial state included
    walk = make_traj(walk_states(4), env=ENV4)
    assert walk.steps == 3
    heat = visit_heatmap([walk], 4)
    assert heat[0].tolist() == [0.25, 0.25, 0.25, 0.25] and heat[1:].sum() == 0.0
    # repeated cells accumulate
    heat = visit_heatmap([make_traj(cycle_states(8))], 6)
    assert sorted(heat[heat > 0].tolist()) == [0.25] * 4 and {tuple(rc) for rc in np.argwhere(heat > 0).tolist()} == SQUARE_CELLS


def test_visit_heatmap_errors():
    with pytest.raises(ValueError):
        visit_heatmap([], 6)
    with pytest.raises(ValueError):
        visit_heatmap([make_traj([((4, 0), UP)], env=ENV4)], 4)  # row W
    with pytest.raises(ValueError):
        visit_heatmap([make_traj([((0, -1), UP)], env=ENV4)], 4)
    with pytest.raises(ValueError):
        visit_heatmap([make_traj(walk_states(4), env=ENV4), make_traj([((2, 7), UP)], env=ENV4)], 4)
    heat = visit_heatmap([make_traj([((3, 3), UP), ((0, 0), UP)], env=ENV4)], 4)  # the corners are inside
    assert heat[3, 3] == 0.5 and heat[0, 0] == 0.5


def test_border_fraction_by_hand():
    border = make_traj([((0, c), RIGHT) for c in range(5)] + [((r, 4), DOWN) for r in range(1, 5)],
                       env=env_config_to_dict(EnvConfig(W=5)))
    assert border_fraction(visit_heatmap([border], 5)) == pytest.approx(1.0, abs=1e-12)
    interior = make_traj([((1, 1), RIGHT), ((1, 2), RIGHT), ((1, 3), DOWN), ((2, 3), DOWN), ((3, 3), LEFT), ((3, 2), LEFT)],
                         env=env_config_to_dict(EnvConfig(W=5)))
    assert border_fraction(visit_heatmap([interior], 5)) == 0.0
    mixed = np.zeros((4, 4))
    mixed[0, 0], mixed[1, 1], mixed[3, 2] = 0.5, 0.25, 0.25
    assert border_fraction(mixed) == pytest.approx(0.75, abs=1e-12)
    corner = np.zeros((4, 4))
    corner[0, 0] = 1.0
    assert border_fraction(corner) == 1.0  # a corner is counted once, not as a row and a column
    assert border_fraction(np.ones((3, 3))) == pytest.approx(8.0)  # the sum over the border cells of the array given
    assert type(border_fraction(mixed)) is float
    for bad in (np.zeros((3, 4)), np.zeros(5), np.zeros((2, 3, 3))):
        with pytest.raises(ValueError):
            border_fraction(bad)


# ---------------------------------------------------------------- heatmap figure
def three_heats():
    a = np.zeros((6, 6))
    a[0, 0], a[2, 2] = 0.5, 0.5  # border 50 %, maximum 0.5
    b = np.zeros((6, 6))
    b[1, 1] = b[1, 2] = b[2, 1] = b[2, 2] = 0.25  # border 0 %, maximum 0.25
    c = np.zeros((6, 6))
    c[0, :] = 1 / 6  # border 100 %
    return a, b, c


def test_heatmap_figure_panels_share_colour_limits_and_carry_border_titles():
    a, b, c = three_heats()
    heats = [("a", a), ("b", b), ("c", c)]
    fig = heatmap_figure(heats)
    assert isinstance(fig.canvas, FigureCanvasAgg) and len(fig.axes) >= 3
    for i, (label, heat) in enumerate(heats):
        ax = fig.axes[i]
        assert len(ax.images) == 1
        image = ax.images[0]
        assert np.array_equal(np.asarray(image.get_array()), heat)
        assert tuple(float(v) for v in image.get_clim()) == (0.0, 0.5)  # vmin 0, vmax over ALL heats
        assert image.origin == "upper"
        assert axes_title(ax) == f"{label} (border {100 * border_fraction(heat):.0f}%)"
        assert_no_tick_labels(ax)
    assert [axes_title(ax) for ax in fig.axes[:3]] == ["a (border 50%)", "b (border 0%)", "c (border 100%)"]
    assert fig.get_suptitle() == ""
    fig.draw_without_rendering()
    boxes = [ax.get_position() for ax in fig.axes[:3]]
    assert boxes[0].x0 < boxes[1].x0 < boxes[2].x0 and all(abs(box.y0 - boxes[0].y0) < 1e-6 for box in boxes)
    assert heatmap_figure([("a", a)], title="visits").get_suptitle() == "visits"
    assert tuple(float(v) for v in heatmap_figure([("b", b)]).axes[0].images[0].get_clim()) == (0.0, 0.25)
    assert axes_title(heatmap_figure([("one", c), ("two", b)]).axes[1]) == "two (border 0%)"


def test_heatmap_figure_errors_and_plot_heatmaps_writes_the_png(tmp_path):
    a, b, _ = three_heats()
    with pytest.raises(ValueError):
        heatmap_figure([])
    with pytest.raises(ValueError):
        heatmap_figure([("a", a), ("b", np.zeros((8, 8)))])  # not the same shape
    with pytest.raises(ValueError):
        heatmap_figure([("a", np.zeros((3, 4)))])  # not square
    with pytest.raises(ValueError):
        plot_heatmaps([], tmp_path / "x.png")
    assert not (tmp_path / "x.png").exists()
    out = tmp_path / "figures" / "nested" / "heat.png"
    returned = plot_heatmaps([("a", a), ("b", b)], out, title="visits")
    assert returned == out and isinstance(returned, Path) and out.is_file() and png_size(out)[0] > 0
    assert plot_heatmaps([("a", a)], str(out)) == out


# ------------------------------------------------------------------ loop_stats
def test_loop_stats_keys_and_a_period_four_cycle_without_food():
    traj = make_traj(cycle_states(13))  # s_0 .. s_12 on the square: A B C D A B C D A B C D A
    stats = loop_stats(traj)
    assert set(stats) == LOOP_KEYS
    assert stats == {"tail_without_food": 12, "period": 4, "loop_steps": 13, "food_steps": [], "n_food": 0,
                     "truncated": True, "cut": False}
    assert stats["loop_steps"] >= 2 * stats["period"]
    assert isinstance(stats["food_steps"], list)
    cut = loop_stats(make_traj(cycle_states(9), cause="none", cut=True))
    assert cut["cut"] == True and cut["truncated"] == False and (cut["period"], cut["loop_steps"]) == (4, 9)  # noqa: E712


def test_loop_stats_of_a_cycle_that_starts_after_the_last_food():
    states = walk_states(5) + cycle_states(10)  # s_0 .. s_4 straight, s_5 .. s_14 = A B C D A B C D A B
    scores = [0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]  # food on steps 2 and 5
    stats = loop_stats(make_traj(states, scores=scores))
    assert stats["food_steps"] == [2, 5] and stats["n_food"] == 2
    assert stats["tail_without_food"] == 9  # T = 14, t_last = 5
    assert stats["period"] == 4 and stats["loop_steps"] == 10  # the whole tail
    # without the food the tail is the whole episode; its straight prefix is not periodic
    stats = loop_stats(make_traj(states))
    assert stats["food_steps"] == [] and stats["n_food"] == 0 and stats["tail_without_food"] == 14
    assert stats["period"] == 4 and stats["loop_steps"] == 10


def test_loop_stats_without_a_cycle():
    stats = loop_stats(make_traj(walk_states(6), cause="wall", rewards=[0, 0, 0, 0, -1]))
    assert stats == {"tail_without_food": 5, "period": 0, "loop_steps": 0, "food_steps": [], "n_food": 0,
                     "truncated": False, "cut": False}
    assert loop_stats(make_traj([A, B]))["period"] == 0  # two states: nothing can repeat twice


def test_loop_stats_period_two_and_period_one_by_hand():
    P, Q = ((1, 1), UP), ((1, 2), RIGHT)
    stats = loop_stats(make_traj([P, Q, P, Q, P, Q, P]))
    assert (stats["period"], stats["loop_steps"], stats["tail_without_food"]) == (2, 7, 6)
    stats = loop_stats(make_traj([Q, Q, P, Q, P, Q]))  # the periodic suffix is Q P Q P Q
    assert (stats["period"], stats["loop_steps"]) == (2, 5)
    stats = loop_stats(make_traj([P, P, P, P]))
    assert (stats["period"], stats["loop_steps"]) == (1, 4)


def test_loop_stats_periodic_prefix_then_non_periodic_suffix():
    stats = loop_stats(make_traj(cycle_states(8) + [((0, 0), UP)]))
    assert stats == {"tail_without_food": 8, "period": 0, "loop_steps": 0, "food_steps": [], "n_food": 0,
                     "truncated": True, "cut": False}


def test_loop_stats_non_periodic_prefix_then_periodic_suffix():
    stats = loop_stats(make_traj([((0, 0), RIGHT), ((0, 1), RIGHT)] + cycle_states(9)))
    assert (stats["period"], stats["loop_steps"], stats["tail_without_food"]) == (4, 9, 10)
    # the same cycle entered at another phase
    stats = loop_stats(make_traj([((0, 0), RIGHT)] + cycle_states(11, phase=2)))
    assert (stats["period"], stats["loop_steps"], stats["tail_without_food"]) == (4, 11, 11)


def test_loop_stats_food_on_the_last_step_leaves_an_empty_tail():
    stats = loop_stats(make_traj(walk_states(5), scores=[0, 0, 0, 0, 1], cause="win"))
    assert stats == {"tail_without_food": 0, "period": 0, "loop_steps": 0, "food_steps": [4], "n_food": 1,
                     "truncated": False, "cut": False}


def test_loop_stats_of_a_left_turning_rollout_by_hand():
    """The real thing: a left turn every step on a 6x6 board, H = 20. From s_0 = ((3, 3), right) the head visits
    (2, 3) up, (2, 2) left, (3, 2) down, (3, 3) right, ... so the states have period 4 from the very first one,
    nothing is eaten on this board, and the episode is starved after 20 steps: the whole tail of 21 states loops."""
    cfg = cfg6(starve_limit=20)
    traj = rollout(LeftTurner(), cfg, seed=10_000)
    assert not any(tuple(f) in SQUARE_CELLS for f in traj.foods.tolist())  # the premise: no food on the square
    assert (traj.cause, traj.steps, traj.score) == ("starve", 20, 0)
    assert [(tuple(h), int(d)) for h, d in zip(traj.heads[:8].tolist(), traj.headings[:8])] == cycle_states(8)
    assert loop_stats(traj) == {"tail_without_food": 20, "period": 4, "loop_steps": 21, "food_steps": [],
                                "n_food": 0, "truncated": True, "cut": False}
    env = SnakeEnv(cfg)
    env.reset(seed=10_000)
    for action in traj.actions.tolist():
        env.step(action)
    assert env.steps_since_food == 20 == loop_stats(traj)["tail_without_food"]


# ------------------------------------------------------------- shaping_reward
def shaping(traj: Trajectory, reward: RewardConfig) -> float:
    column = episode_table([traj], reward)["shaping_reward"]
    assert len(column) == 1
    return float(column.iloc[0])


def test_shaping_reward_r0_r3_and_the_death_penalty():
    r0, r3 = RewardConfig(variant="R0"), RewardConfig(variant="R3")
    assert (r0.food, r0.death, r3.closer, r3.farther) == (1.0, -1.0, 0.01, 0.0)  # R3 asymmetric by default
    # R0: a food and a wall death leave nothing once food and death are taken out (seven states, six steps)
    path7 = walk_states(6) + [((1, 5), DOWN)]
    wall = make_traj(path7, scores=[0, 0, 1, 1, 1, 1, 1], rewards=[0.0, 1.0, 0.0, 0.0, 0.0, -1.0], cause="wall")
    assert shaping(wall, r0) == pytest.approx(0.0, abs=1e-12)
    assert shaping(make_traj(cycle_states(7)), r0) == 0.0  # starved without food: six zero rewards
    # R3 asymmetric: +closer on every second step of a back-and-forth tail
    oscillation = make_traj(cycle_states(7), rewards=[0.01, 0.0, 0.01, 0.0, 0.01, 0.0])
    assert shaping(oscillation, r3) == pytest.approx(0.03, abs=1e-12)
    assert shaping(make_traj(cycle_states(21), rewards=[0.01, 0.0] * 10), r3) == pytest.approx(0.1, abs=1e-12)
    # R3 with a food (1 + closer) and a wall death (-1 + farther): the shaping is what remains
    r3_wall = make_traj(path7, scores=[0, 0, 1, 1, 1, 1, 1], rewards=[0.01, 1.01, 0.01, 0.0, 0.0, -1.0], cause="wall")
    assert shaping(r3_wall, r3) == pytest.approx(0.03, abs=1e-12)
    # the death penalty is taken out for wall and self, not for starve, a cut or a win
    rewards = [0.0, 0.0, -1.0]
    assert shaping(make_traj(walk_states(4), rewards=rewards, cause="wall"), r0) == pytest.approx(0.0, abs=1e-12)
    assert shaping(make_traj(walk_states(4), rewards=rewards, cause="self"), r0) == pytest.approx(0.0, abs=1e-12)
    assert shaping(make_traj(walk_states(4), rewards=rewards, cause="starve"), r0) == pytest.approx(-1.0, abs=1e-12)
    assert shaping(make_traj(walk_states(4), rewards=rewards, cause="none", cut=True), r0) == pytest.approx(-1.0, abs=1e-12)
    assert shaping(make_traj(walk_states(2), scores=[0, 1], rewards=[1.0], cause="win"), r0) == pytest.approx(0.0, abs=1e-12)
    # the food and death values of the reward argument are the ones used
    custom = RewardConfig(variant="R0", food=2.0, death=-3.0)
    assert shaping(make_traj(walk_states(4), scores=[0, 1, 1, 1], rewards=[2.0, 0.0, -3.0], cause="wall"), custom) == pytest.approx(0.0, abs=1e-12)
    assert shaping(wall, custom) == pytest.approx(0.0 - 2.0 + 3.0, abs=1e-12)
    assert type(episode_table([wall], r0)["shaping_reward"].iloc[0]) in (float, np.float64)


# ------------------------------------------- episode_table, summarise, hacking
def test_episode_table_columns_in_order_and_values_by_hand():
    a, b, c, q = three_trajectories()
    table = episode_table([a, b, c], RewardConfig())
    assert isinstance(table, pd.DataFrame) and tuple(table.columns) == TABLE_COLUMNS and len(table) == 3
    assert table["seed"].tolist() == [3, 1, 2]  # input order
    assert table["score"].tolist() == [0, 0, 0] and table["steps"].tolist() == [20, 5, 12]
    assert table["length"].tolist() == [3, 3, 3] and table["cause"].tolist() == ["starve", "wall", "none"]
    assert table["truncated"].tolist() == [True, False, False] and table["cut"].tolist() == [False, False, True]
    assert table["tail_without_food"].tolist() == [20, 5, 12]
    assert table["period"].tolist() == [4, 0, 4] and table["loop_steps"].tolist() == [21, 0, 13]
    assert table["shaping_reward"].tolist() == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
    assert table["border_fraction"].tolist() == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)  # the square is interior
    for traj, value in zip((a, b, c), table["border_fraction"]):
        assert value == pytest.approx(border_fraction(visit_heatmap([traj], 6)), abs=1e-12)
    assert table["q_max_mean"].iloc[0] == pytest.approx(q.max(axis=1).mean(), rel=1e-12)
    assert math.isnan(table["q_max_mean"].iloc[1]) and math.isnan(table["q_max_mean"].iloc[2])
    assert episode_table((a,), RewardConfig()).shape == (1, len(TABLE_COLUMNS))
    # W comes from the trajectory's env: on a 4x4 board only (2, 2) of the square is interior, and the nine
    # states A B C D A B C D A visit A three times, B C D twice -> 7 of 9 counts on the border
    on_four = make_traj(cycle_states(9), env=ENV4, seed=4)
    assert episode_table([on_four], RewardConfig())["border_fraction"].iloc[0] == pytest.approx(7 / 9, abs=1e-12)


def test_q_max_mean_is_nan_without_q_values_or_without_steps():
    with_q = make_traj(walk_states(4), q_values=np.array([[1.0, 2.0, 0.0], [3.0, -1.0, 0.5], [0.0, 0.0, 0.25]]))
    empty_q = make_traj([((1, 1), UP)], q_values=np.zeros((0, 3)))
    none_q = make_traj([((1, 1), UP)])
    table = episode_table([with_q, empty_q, none_q], RewardConfig())
    assert table["steps"].tolist() == [3, 0, 0]
    assert table["q_max_mean"].iloc[0] == pytest.approx((2.0 + 3.0 + 0.25) / 3, rel=1e-12)
    assert table["q_max_mean"].iloc[1:].isna().all()


def test_summarise_keys_and_values_by_hand():
    a, b, c, q = three_trajectories()
    s = summarise(episode_table([a, b, c], RewardConfig()))
    assert set(s) == SUMMARY_KEYS
    assert s["n_episodes"] == 3 and type(s["n_episodes"]) is int
    assert all(type(v) is float for k, v in s.items() if k != "n_episodes"), {k: type(v) for k, v in s.items()}
    assert s["score_mean"] == 0.0 and s["steps_mean"] == pytest.approx(37 / 3, rel=1e-12) and s["length_mean"] == 3.0
    assert (s["frac_wall"], s["frac_self"], s["frac_starve"], s["frac_win"]) == pytest.approx((1 / 3, 0.0, 1 / 3, 0.0))
    assert s["frac_truncated"] == pytest.approx(1 / 3) and s["frac_cut"] == pytest.approx(1 / 3)
    assert s["tail_without_food_mean"] == pytest.approx(37 / 3, rel=1e-12)
    assert s["loop_fraction"] == pytest.approx(1 / 3)  # a: 21 >= 20; c: 13 < 20; b: no period
    assert s["loop_steps_mean"] == pytest.approx(34 / 3, rel=1e-12)  # over ALL rows, loops or not
    assert s["shaping_reward_mean"] == pytest.approx(0.0, abs=1e-12)
    assert s["border_fraction_mean"] == pytest.approx(1 / 3, rel=1e-12)
    assert s["q_max_mean"] == pytest.approx(q.max(axis=1).mean(), rel=1e-12)  # nanmean: the one row that has values
    assert math.isnan(summarise(episode_table([b, c], RewardConfig()))["q_max_mean"])  # all NaN -> NaN
    one = summarise(episode_table([b], RewardConfig()))
    assert one["n_episodes"] == 1 and one["frac_wall"] == 1.0 and one["loop_fraction"] == 0.0 and one["steps_mean"] == 5.0
    with pytest.raises(ValueError):
        summarise(pd.DataFrame(columns=list(TABLE_COLUMNS)))


def test_loop_fraction_counts_loops_of_at_least_min_loop_steps():
    short = make_traj(cycle_states(19), seed=1)  # loop_steps 19
    long = make_traj(cycle_states(20), seed=2)  # loop_steps 20 = MIN_LOOP_STEPS
    table = episode_table([short, long], RewardConfig())
    assert table["period"].tolist() == [4, 4] and table["loop_steps"].tolist() == [19, 20]
    assert summarise(table)["loop_fraction"] == 0.5
    assert summarise(episode_table([short], RewardConfig()))["loop_fraction"] == 0.0
    assert summarise(episode_table([long], RewardConfig()))["loop_fraction"] == 1.0


def test_hacking_candidates_threshold_and_order():
    t5 = make_traj(cycle_states(21), seed=5)
    t2 = make_traj(cycle_states(25), seed=2)
    t9 = make_traj(cycle_states(21), seed=9)
    t1 = make_traj(cycle_states(13), seed=1)
    t7 = make_traj(walk_states(6), seed=7, cause="wall")
    trajs = [t5, t1, t7, t2, t9]
    found = hacking_candidates(trajs)
    assert [t.seed for t in found] == [2, 5, 9]  # loop_steps 25, then 21 twice with the seeds ascending
    assert found[0] is t2 and found[1] is t5 and found[2] is t9
    assert [t.seed for t in hacking_candidates(trajs, min_loop_steps=13)] == [2, 5, 9, 1]
    assert [t.seed for t in hacking_candidates(trajs, 21)] == [2, 5, 9]
    assert hacking_candidates(trajs, min_loop_steps=22) == [t2]
    assert hacking_candidates(trajs, min_loop_steps=30) == []
    assert hacking_candidates([t7, t1]) == [] and hacking_candidates([]) == []
    assert hacking_candidates(tuple(trajs)) == found


# ------------------------------------------------------------- last-steps figure
def last_steps_trajectories() -> tuple[Trajectory, Trajectory]:
    straight = rollout(AlwaysStraight(), cfg6(), seed=10_001)  # three moves to the wall, nothing eaten
    bfs = rollout(make_policy("bfs"), cfg6(), seed=10_000)  # a long, eating episode
    assert (straight.steps, straight.cause, straight.score) == (3, "wall", 0) and bfs.steps > 10
    return straight, bfs


def test_last_steps_figure_panels_titles_limits_and_the_last_moves():
    straight, bfs = last_steps_trajectories()
    fig = last_steps_figure([straight, bfs], k=10)
    assert isinstance(fig.canvas, FigureCanvasAgg) and len(fig.axes) == 2
    for ax, traj in zip(fig.axes, (straight, bfs)):
        assert axes_title(ax) == f"seed {traj.seed}: {traj.cause}, score {traj.score}"
        assert_board_limits(ax, 6)
        assert_no_tick_labels(ax)
        assert polyline_at(ax, traj.heads[-11:]) is not None  # the last min(k, T) moves: k + 1 or T + 1 cells
        assert marker_at(ax, tuple(traj.foods[-1].tolist())), "the final food"
        assert ax.patches or ax.images or ax.collections  # the snake's cells are drawn
    assert axes_title(fig.axes[0]) == "seed 10001: wall, score 0"
    assert polyline_at(fig.axes[0], [[3, 3], [3, 4], [3, 5], [3, 5]]) is not None  # the whole short episode
    assert fig.get_suptitle() == ""
    assert last_steps_figure([straight], title="last moves").get_suptitle() == "last moves"
    fig = last_steps_figure([bfs], k=3)
    assert polyline_at(fig.axes[0], bfs.heads[-4:]) is not None and polyline_at(fig.axes[0], bfs.heads[-11:]) is None
    fig = last_steps_figure([bfs])  # k defaults to 10
    assert polyline_at(fig.axes[0], bfs.heads[-11:]) is not None


def test_last_steps_figure_errors_and_plot_last_steps(tmp_path):
    straight, bfs = last_steps_trajectories()
    for k in (0, -1):
        with pytest.raises(ValueError):
            last_steps_figure([straight], k=k)
    with pytest.raises(ValueError):
        last_steps_figure([])
    assert len(last_steps_figure([straight] * 8).axes) == 8
    with pytest.raises(ValueError):
        last_steps_figure([straight] * 9)
    with pytest.raises(ValueError):
        plot_last_steps([straight] * 9, tmp_path / "x.png")
    with pytest.raises(ValueError):
        plot_last_steps([straight], tmp_path / "x.png", k=0)
    assert not (tmp_path / "x.png").exists()
    out = tmp_path / "figures" / "nested" / "last.png"
    returned = plot_last_steps([straight, bfs], out, k=5, title="t")
    assert returned == out and isinstance(returned, Path) and out.is_file() and png_size(out)[0] > 0
    assert plot_last_steps([bfs], str(out)) == out


# ---------------------------------------------------------- checkpoint selection
def test_checkpoint_files_are_sorted_by_env_steps(tmp_path):
    run = write_dqn_run(tmp_path / "dqn_W6", 0, checkpoint_steps=(1000, 100, 500, 200, 800, 300, 600, 400, 700, 900))
    found = checkpoint_files(run)
    assert [step for step, _ in found] == list(range(100, 1001, 100))
    assert [path for _, path in found] == [run / "checkpoints" / f"step_{s:08d}.pt" for s in range(100, 1001, 100)]
    assert all(type(step) is int and isinstance(path, Path) for step, path in found)
    assert checkpoint_files(str(run)) == found
    assert checkpoint_files(write_dqn_run(tmp_path / "plain", 0)) == []  # keep_checkpoints: false leaves none


def test_resolve_checkpoint_aliases_pick_the_nearest_with_ties_to_the_smaller_step(tmp_path):
    run = write_dqn_run(tmp_path / "ten", 0, checkpoint_steps=range(100, 1001, 100))  # 10 %, 20 %, ..., 100 %
    assert resolve_checkpoint(run, "early") == (100, run / "checkpoints" / "step_00000100.pt")
    assert resolve_checkpoint(run, "mid") == (500, run / "checkpoints" / "step_00000500.pt")
    assert resolve_checkpoint(run, "late") == (1000, run / "checkpoints" / "step_00001000.pt")
    assert resolve_checkpoint(str(run), "early") == (100, run / "checkpoints" / "step_00000100.pt")
    ties = write_dqn_run(tmp_path / "ties", 0, checkpoint_steps=(0, 200, 400, 600, 800, 1000))
    assert resolve_checkpoint(ties, "early") == (0, ties / "checkpoints" / "step_00000000.pt")  # 100: 0 or 200
    assert resolve_checkpoint(ties, "mid") == (400, ties / "checkpoints" / "step_00000400.pt")  # 500: 400 or 600
    assert resolve_checkpoint(ties, "late") == (1000, ties / "checkpoints" / "step_00001000.pt")
    # the fractions are of the largest checkpoint step, not of the budget
    short = write_dqn_run(tmp_path / "short", 0, checkpoint_steps=(0, 100, 200, 300, 400))
    assert resolve_checkpoint(short, "early")[0] == 0  # 40: 0 is nearer than 100
    assert resolve_checkpoint(short, "mid")[0] == 200 and resolve_checkpoint(short, "late")[0] == 400


def test_resolve_checkpoint_final_decimal_and_errors(tmp_path):
    run = write_dqn_run(tmp_path / "ten", 0, checkpoint_steps=range(100, 1001, 100))
    assert resolve_checkpoint(run, "final") == (1000, run / "model.pt")
    assert load_config(run / "config.yaml").train.total_env_steps == 1000
    assert resolve_checkpoint(run, "300") == (300, run / "checkpoints" / "step_00000300.pt")
    assert resolve_checkpoint(run, "1000") == (1000, run / "checkpoints" / "step_00001000.pt")
    for bad in ("301", "0", "foo", "", "-100", "1e2", "latest"):
        with pytest.raises(ValueError):
            resolve_checkpoint(run, bad)
    plain = write_dqn_run(tmp_path / "plain", 0, total_env_steps=2000, eval_every=200)
    assert resolve_checkpoint(plain, "final") == (2000, plain / "model.pt")  # no checkpoints needed
    for alias in ("early", "mid", "late"):
        with pytest.raises(ValueError):
            resolve_checkpoint(plain, alias)
    with pytest.raises(ValueError):
        resolve_checkpoint(plain, "200")


# -------------------------------------------------------------- analyse_player
def test_analyse_player_writes_the_files_and_the_summary(tmp_path):
    cfg = cfg6(starve_limit=20)
    out = tmp_path / "analysis" / "left"
    assert not out.exists()
    summary = analyse_player(LeftTurner(), cfg, out, n_episodes=4, seed=10_000, record_top=3,
                             meta={"checkpoint": None, "source": "left"})
    names = sorted(p.name for p in out.iterdir())
    assert names == ["episodes.csv", "hack_1_10000.json", "hack_2_10001.json", "hack_3_10002.json", "heatmap.npy",
                     "heatmap.png", "summary.json"]  # no last_steps.png: every episode starves
    # the summary: the statistics of the table plus the bookkeeping keys and the meta keys
    assert set(summary) == SUMMARY_KEYS | {"player", "env", "seed", "n_episodes", "max_steps", "git", "hacking_cases",
                                           "checkpoint", "source"}
    assert summary["player"] == "left" and summary["env"] == env_config_to_dict(cfg)
    assert summary["seed"] == 10_000 and summary["n_episodes"] == 4 and summary["max_steps"] is None
    assert summary["git"] == git_hash() and isinstance(summary["git"], str)
    assert summary["hacking_cases"] == ["hack_1_10000.json", "hack_2_10001.json", "hack_3_10002.json"]
    assert summary["checkpoint"] is None and summary["source"] == "left"
    assert summary["frac_starve"] == 1.0 and summary["loop_fraction"] == 1.0 and summary["loop_steps_mean"] == 21.0
    assert summary["tail_without_food_mean"] == 20.0 and summary["border_fraction_mean"] == 0.0
    written = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    np.testing.assert_equal(written, summary)  # nan == nan under assert_equal
    # episodes.csv is the episode table
    trajs = rollouts(LeftTurner(), cfg, 4, 10_000)
    table = episode_table(trajs, cfg.reward)
    csv = pd.read_csv(out / "episodes.csv")
    assert list(csv.columns) == list(TABLE_COLUMNS) and len(csv) == 4
    for column in ("seed", "score", "steps", "length", "cause", "truncated", "cut", "tail_without_food", "period", "loop_steps"):
        assert csv[column].tolist() == table[column].tolist(), column
    for column in ("shaping_reward", "border_fraction"):
        assert csv[column].tolist() == pytest.approx(table[column].tolist(), abs=1e-9), column
    assert csv["q_max_mean"].isna().all()
    assert summary["n_episodes"] == summarise(table)["n_episodes"] and summary["steps_mean"] == summarise(table)["steps_mean"]
    # the heat-map of all episodes
    heat = np.load(out / "heatmap.npy")
    assert heat.shape == (6, 6) and abs(heat.sum() - 1.0) < 1e-12
    np.testing.assert_allclose(heat, visit_heatmap(trajs, 6), rtol=1e-12)
    assert png_size(out / "heatmap.png")[0] > 0
    # the recordings of the hacking cases, ranked
    candidates = hacking_candidates(trajs)
    assert [t.seed for t in candidates] == [10_000, 10_001, 10_002, 10_003]  # equal loops: seeds ascending
    for rank, name in enumerate(summary["hacking_cases"], start=1):
        rec = Recording.load(out / name)
        assert rec == candidates[rank - 1].to_recording()
        assert rec.seed == 10_000 + rank - 1 and rec.player == "left" and rec.cause == "starve" and rec.steps == 20


def test_analyse_player_record_top_zero_and_more_than_the_candidates(tmp_path):
    cfg = cfg6(starve_limit=20)
    summary = analyse_player(LeftTurner(), cfg, tmp_path / "none", n_episodes=3, seed=10_000, record_top=0)
    assert summary["hacking_cases"] == [] and not list((tmp_path / "none").glob("hack_*"))
    assert summary["loop_fraction"] == 1.0  # candidates exist, none is written
    assert "checkpoint" not in summary and "source" not in summary  # no meta, no extra keys
    summary = analyse_player(LeftTurner(), cfg, tmp_path / "all", n_episodes=3, seed=10_000, record_top=10)
    assert summary["hacking_cases"] == ["hack_1_10000.json", "hack_2_10001.json", "hack_3_10002.json"]
    assert sorted(p.name for p in (tmp_path / "all").glob("hack_*")) == summary["hacking_cases"]
    assert summary["n_episodes"] == 3


def test_analyse_player_last_steps_only_with_wall_or_self_deaths(tmp_path):
    cfg = cfg6()
    out = tmp_path / "straight"
    summary = analyse_player(AlwaysStraight(), cfg, out, n_episodes=3, seed=10_001, record_top=3, k_last=2)
    assert (out / "last_steps.png").is_file() and png_size(out / "last_steps.png")[0] > 0
    assert summary["frac_wall"] == 1.0 and summary["hacking_cases"] == [] and summary["loop_fraction"] == 0.0
    assert summary["steps_mean"] == 3.0 and summary["n_episodes"] == 3
    assert sorted(p.name for p in out.iterdir()) == ["episodes.csv", "heatmap.npy", "heatmap.png", "last_steps.png", "summary.json"]
    # a mixed player: bfs on 6x6 dies on the wall or on itself
    out = tmp_path / "bfs"
    summary = analyse_player(make_policy("bfs"), cfg, out, n_episodes=2, seed=10_000, max_steps=None)
    assert (out / "last_steps.png").is_file() and summary["frac_wall"] + summary["frac_self"] == 1.0
    assert summary["player"] == "bfs" and summary["max_steps"] is None
    # cut by max_steps: no death, so no last-steps figure, and the cap is recorded
    out = tmp_path / "cut"
    summary = analyse_player(make_policy("bfs"), cfg, out, n_episodes=2, seed=10_000, max_steps=4)
    assert not (out / "last_steps.png").exists() and summary["frac_cut"] == 1.0 and summary["max_steps"] == 4
    assert summary["steps_mean"] == 4.0

"""Behaviour analysis (M8): what a trained policy does, not only how much it
scores. Everything is recomputed from the scalar environment and a policy, so
every figure can be regenerated from a checkpoint (project rule: no number
typed by hand).

Three questions of ``docs/Snake项目方案-设计依据.md`` section 5 and Part B
("behaviour analysis: visit heat-maps, last-steps trajectory replays,
reward-hacking cases"):

* **Where does the agent go?** :func:`visit_heatmap` counts the head position
  over the states of greedy evaluation episodes (the same seeds as
  ``snake_rl.evaluate``), normalised to a distribution over the ``W x W``
  cells; :func:`border_fraction` is the share of that mass on the outer ring
  ("hugs the walls" against "crosses the middle").
* **How does it die?** :func:`last_steps_figure` draws the body at the end of
  an episode and the last ``k`` head positions. The evolution of the death
  causes over training is a curve of ``metrics.csv`` and lives in
  ``aggregate.death_cause_curve`` / ``plots.death_cause_figure``.
* **Does it exploit the reward?** :func:`loop_stats` looks at the *tail* of an
  episode, the states after the last food. A policy that farms a shaping
  reward (R3: +0.01 for moving closer) instead of eating moves in a cycle
  there; the smallest period of the ``(row, col, heading)`` state sequence at
  the end of the tail and the length of the periodic suffix are the two
  numbers, ``shaping_reward`` (the reward collected that is neither food nor
  death) is the third. Episodes whose periodic suffix is at least
  ``MIN_LOOP_STEPS`` states long are *hacking candidates*; they are written as
  GUI recordings (``snake_rl.gui.app.Recording``) so that the case can be
  replayed in the demo. Starvation truncation ends every such episode after
  ``H`` food-less steps (PLAN.md section 3), so a loop never runs for ever.

Trajectories are :class:`Trajectory` objects from :func:`rollout`, which plays
one episode exactly as ``snake_rl.evaluate.evaluate`` does (same reset seed,
``policy.reset()``, ``policy.act(obs, env)``) and additionally records the
head, heading, food, score, action, reward and - for a policy with
``q_values`` - the action values before every step.

CLI::

    python -m snake_rl.analysis.behaviour --player SPEC [--checkpoint early|mid|late|final|STEP] [--config FILE]
                                          --out-dir DIR [--episodes N] [--seed S] [--max-steps N]
                                          [--record-top K] [--k-last K] [--device cpu|cuda]
    python -m snake_rl.analysis.behaviour --exp runs/<exp> [--checkpoint ...] --out-dir DIR [...]

``--player`` is a baseline name (``random`` / ``bfs`` / ``hamiltonian``; then
``--config`` names the experiment YAML whose ``env`` block is the board), a
run directory (``--checkpoint`` picks the network: an alias of the training
budget or an exact env-steps count), a ``*.pt`` checkpoint or a ``*.npz``
table. ``--exp`` analyses every seed of an experiment and adds the seed-mean
heat-map and a summary with Student-t intervals over seeds.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from snake_rl.analysis.aggregate import mean_ci95
from snake_rl.analysis.plots import DPI, FONT_SIZE, INK, INK_SECONDARY, SURFACE, _save_png
from snake_rl.env.core import EnvConfig, SnakeEnv
from snake_rl.env.rewards import RewardConfig
from snake_rl.evaluate import DEFAULT_EVAL_SEED, env_config_to_dict
from snake_rl.utils import git_hash

DEFAULT_EPISODES = 100
DEFAULT_SEED = DEFAULT_EVAL_SEED
CHECKPOINT_ALIASES = ("early", "mid", "late", "final")
#: An episode whose periodic suffix (``loop_stats``) is at least this many
#: states long counts as a reward-hacking candidate.
MIN_LOOP_STEPS = 20
#: Causes that end an episode by a collision (the last-steps figure shows those).
COLLISION_CAUSES = ("wall", "self")

EPISODE_COLUMNS = (
    "seed", "score", "steps", "length", "cause", "truncated", "cut", "tail_without_food", "period", "loop_steps",
    "shaping_reward", "border_fraction", "q_max_mean",
)
SUMMARY_KEYS = (
    "n_episodes", "score_mean", "steps_mean", "length_mean", "frac_wall", "frac_self", "frac_starve", "frac_win",
    "frac_truncated", "frac_cut", "tail_without_food_mean", "loop_fraction", "loop_steps_mean", "shaping_reward_mean",
    "border_fraction_mean", "q_max_mean",
)

HEAD_COLOR = "#1f4e8c"
BODY_COLOR = "#7aa6d8"
PATH_COLOR = "#e34948"
FOOD_COLOR = "#1baf7a"
GRID_COLOR = "#e1e0d9"


# ------------------------------------------------------------------ rollouts
@dataclass
class Trajectory:
    """One greedy episode, state by state (module docstring).

    ``heads``, ``headings``, ``foods`` and ``scores`` describe the states
    ``s_0 .. s_T`` (``T + 1`` entries); ``actions``, ``rewards`` and
    ``q_values`` the ``T`` steps between them.
    """

    env: dict[str, Any]
    seed: int
    player: str
    heads: np.ndarray
    headings: np.ndarray
    foods: np.ndarray
    scores: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    q_values: np.ndarray | None
    final_cells: list[tuple[int, int]]
    score: int
    steps: int
    length: int
    cause: str
    truncated: bool
    cut: bool

    def to_recording(self):
        """The episode as a GUI recording (``snake_rl.gui.app.Recording``):
        the environment, the seed and the actions re-simulate it exactly."""
        from snake_rl.gui.app import Recording  # pygame, only when needed

        q_values = None if self.q_values is None else [[float(v) for v in row] for row in self.q_values]
        return Recording(
            env=dict(self.env), seed=int(self.seed), player=str(self.player), actions=[int(a) for a in self.actions],
            q_values=q_values, score=int(self.score), steps=int(self.steps), cause=str(self.cause),
        )


def rollout(policy, cfg: EnvConfig, seed: int, max_steps: int | None = None) -> Trajectory:
    """Play one greedy episode of ``policy`` on ``SnakeEnv(cfg)`` from
    ``reset(seed=seed)`` and record it (module docstring).

    Stops at the end of the episode or after ``max_steps`` steps (``cut``);
    with ``max_steps=None`` the starvation rule ends every episode.
    """
    if max_steps is not None and max_steps < 1:
        raise ValueError(f"max_steps must be None or >= 1, got {max_steps!r}")
    env = SnakeEnv(cfg)
    obs, info = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()
    has_q = hasattr(policy, "q_values")
    heads = [tuple(int(v) for v in env.head)]
    headings = [int(env.heading)]
    foods = [tuple(int(v) for v in env.food)]
    scores = [int(env.score)]
    actions: list[int] = []
    rewards: list[float] = []
    q_rows: list[np.ndarray] = []
    truncated = cut = False
    while True:
        if has_q:
            q_rows.append(np.asarray(policy.q_values(obs[None])[0], dtype=np.float64))
        action = int(policy.act(obs, env))
        obs, reward, terminated, trunc, info = env.step(action)
        actions.append(action)
        rewards.append(float(reward))
        heads.append(tuple(int(v) for v in env.head))
        headings.append(int(env.heading))
        foods.append(tuple(int(v) for v in env.food))
        scores.append(int(env.score))
        if terminated or trunc:
            truncated = bool(trunc)
            break
        if max_steps is not None and len(actions) >= max_steps:
            cut = True
            break
    return Trajectory(
        env=env_config_to_dict(cfg),
        seed=int(seed),
        player=str(getattr(policy, "name", type(policy).__name__)),
        heads=np.array(heads, dtype=np.int64).reshape(-1, 2),
        headings=np.array(headings, dtype=np.int64),
        foods=np.array(foods, dtype=np.int64).reshape(-1, 2),
        scores=np.array(scores, dtype=np.int64),
        actions=np.array(actions, dtype=np.int64),
        rewards=np.array(rewards, dtype=np.float64),
        q_values=np.array(q_rows, dtype=np.float64) if has_q else None,
        # snake_cells() lists tail -> head; the trajectory keeps the head first.
        final_cells=[tuple(int(v) for v in cell) for cell in reversed(env.snake_cells())],
        score=int(info["score"]),
        steps=int(info["steps"]),
        length=int(info["length"]),
        cause=str(info["cause"]),
        truncated=truncated,
        cut=cut,
    )


def rollouts(
    policy, cfg: EnvConfig, n_episodes: int = DEFAULT_EPISODES, seed: int = DEFAULT_SEED, max_steps: int | None = None
) -> list[Trajectory]:
    """``n_episodes`` episodes, episode ``i`` from ``reset(seed=seed + i)``:
    the protocol of ``snake_rl.evaluate.evaluate``, so the scores, steps,
    lengths and causes equal its records episode by episode."""
    return [rollout(policy, cfg, seed + i, max_steps=max_steps) for i in range(n_episodes)]


# ------------------------------------------------------------------ heat-maps
def visit_heatmap(trajectories: Sequence[Trajectory], W: int) -> np.ndarray:
    """Distribution of the head position over all states of ``trajectories``
    (``T + 1`` states per episode, the initial one included): a ``(W, W)``
    array that sums to 1, indexed ``[row, col]`` like ``body_age``."""
    if not trajectories:
        raise ValueError("visit_heatmap needs at least one trajectory")
    heat = np.zeros((W, W), dtype=np.float64)
    for traj in trajectories:
        heads = np.asarray(traj.heads, dtype=np.int64).reshape(-1, 2)
        if heads.size and (heads.min() < 0 or heads.max() >= W):
            raise ValueError(f"episode seed {traj.seed}: a head position lies outside the {W}x{W} board")
        np.add.at(heat, (heads[:, 0], heads[:, 1]), 1.0)
    return heat / heat.sum()


def border_fraction(heat: np.ndarray) -> float:
    """Share of ``heat`` on the outer ring of cells (row or column 0 or W-1)."""
    heat = np.asarray(heat, dtype=np.float64)
    if heat.ndim != 2 or heat.shape[0] != heat.shape[1]:
        raise ValueError(f"heat must be a square 2-D array, got shape {heat.shape}")
    W = heat.shape[0]
    mask = np.zeros((W, W), dtype=bool)
    mask[0, :] = mask[W - 1, :] = mask[:, 0] = mask[:, W - 1] = True
    return float(heat[mask].sum())


def heatmap_figure(heats: Sequence[tuple[str, np.ndarray]], title: str | None = None) -> Figure:
    """One panel per ``(label, heat)``, all on the same colour scale (0 to
    the largest cell of any panel), titled with the label and the border
    share; a colour bar axes follows the panels."""
    heats = [(str(label), np.asarray(h, dtype=np.float64)) for label, h in heats]
    if not heats:
        raise ValueError("heatmap_figure needs at least one heat-map")
    shape = heats[0][1].shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(f"a heat-map must be a square 2-D array, got shape {shape}")
    if any(h.shape != shape for _, h in heats):
        raise ValueError("all heat-maps of one figure must have the same shape")
    vmax = max(float(h.max()) for _, h in heats)
    n = len(heats)
    fig = Figure(figsize=(2.6 * n + 1.2, 3.0), dpi=DPI, facecolor=SURFACE, layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, n, squeeze=False)[0]
    image = None
    for ax, (label, heat) in zip(axes, heats):
        image = ax.imshow(heat, origin="upper", vmin=0.0, vmax=vmax if vmax > 0 else 1.0, cmap="viridis",
                          interpolation="nearest")
        ax.set_title(f"{label} (border {100 * border_fraction(heat):.0f}%)", fontsize=FONT_SIZE, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
    fig.colorbar(image, ax=axes.tolist(), shrink=0.8, pad=0.02).ax.tick_params(labelsize=FONT_SIZE - 1)
    if title is not None:
        fig.suptitle(title, x=0.015, ha="left", fontsize=FONT_SIZE + 2, color=INK)
    return fig


def plot_heatmaps(heats: Sequence[tuple[str, np.ndarray]], out: str | Path, title: str | None = None) -> Path:
    """Save :func:`heatmap_figure` as a PNG (parents created, overwritten)."""
    return _save_png(heatmap_figure(heats, title=title), Path(out))


# ------------------------------------------------------- loops / hacking cases
def _food_steps(traj: Trajectory) -> list[int]:
    scores = np.asarray(traj.scores)
    return [int(t) for t in range(1, traj.steps + 1) if scores[t] > scores[t - 1]]


def _tail_states(traj: Trajectory, t_last: int) -> list[tuple[int, int, int]]:
    heads, headings = np.asarray(traj.heads), np.asarray(traj.headings)
    return [(int(heads[t, 0]), int(heads[t, 1]), int(headings[t])) for t in range(t_last, traj.steps + 1)]


def _period(states: Sequence[tuple[int, int, int]]) -> int:
    n = len(states)
    for p in range(1, n // 2 + 1):
        if list(states[n - p:]) == list(states[n - 2 * p:n - p]):
            return p
    return 0


def _loop_steps(states: Sequence[tuple[int, int, int]], period: int) -> int:
    if period == 0:
        return 0
    n = len(states)
    size = 2 * period
    while size < n and states[n - size - 1] == states[n - size - 1 + period]:
        size += 1
    return size


def loop_stats(traj: Trajectory) -> dict[str, Any]:
    """Loop statistics of the tail of an episode (module docstring):
    ``tail_without_food`` (steps since the last food at the end), ``period``
    (smallest period of the ``(row, col, heading)`` sequence at the end of the
    tail, 0 if none), ``loop_steps`` (states in the longest periodic suffix,
    0 without a period), ``food_steps`` / ``n_food``, and the ``truncated`` /
    ``cut`` flags.

    A collision leaves the head where it was, so an episode ending by
    ``wall`` / ``self`` has two equal final states: ``period 1, loop_steps 2``
    is the signature of a collision, not of a loop. ``MIN_LOOP_STEPS`` keeps
    such episodes out of the hacking candidates.
    """
    food_steps = _food_steps(traj)
    t_last = food_steps[-1] if food_steps else 0
    states = _tail_states(traj, t_last)
    period = _period(states)
    return {
        "tail_without_food": int(traj.steps - t_last),
        "period": int(period),
        "loop_steps": int(_loop_steps(states, period)),
        "food_steps": food_steps,
        "n_food": len(food_steps),
        "truncated": bool(traj.truncated),
        "cut": bool(traj.cut),
    }


def shaping_reward(traj: Trajectory, reward: RewardConfig) -> float:
    """Reward of the episode that is neither food nor the death penalty."""
    death = float(reward.death) if traj.cause in COLLISION_CAUSES else 0.0
    return float(np.asarray(traj.rewards, dtype=np.float64).sum() - traj.score * float(reward.food) - death)


def episode_table(trajectories: Sequence[Trajectory], reward: RewardConfig) -> pd.DataFrame:
    """One row per episode, columns ``EPISODE_COLUMNS``."""
    rows = []
    for traj in trajectories:
        stats = loop_stats(traj)
        W = int(traj.env["W"])
        q = traj.q_values
        q_max_mean = float(np.asarray(q, dtype=np.float64).max(axis=1).mean()) if q is not None and len(q) else np.nan
        rows.append(
            {
                "seed": int(traj.seed),
                "score": int(traj.score),
                "steps": int(traj.steps),
                "length": int(traj.length),
                "cause": str(traj.cause),
                "truncated": bool(traj.truncated),
                "cut": bool(traj.cut),
                "tail_without_food": stats["tail_without_food"],
                "period": stats["period"],
                "loop_steps": stats["loop_steps"],
                "shaping_reward": shaping_reward(traj, reward),
                "border_fraction": border_fraction(visit_heatmap([traj], W)),
                "q_max_mean": q_max_mean,
            }
        )
    return pd.DataFrame(rows, columns=list(EPISODE_COLUMNS))


def summarise(table: pd.DataFrame) -> dict[str, Any]:
    """Episode-level means and fractions of an :func:`episode_table`
    (keys ``SUMMARY_KEYS``); ``loop_fraction`` is the share of episodes with
    a period and at least ``MIN_LOOP_STEPS`` periodic states."""
    if len(table) == 0:
        raise ValueError("summarise needs at least one episode")
    n = len(table)
    looping = (table["period"] > 0) & (table["loop_steps"] >= MIN_LOOP_STEPS)
    q = table["q_max_mean"].to_numpy(dtype=np.float64)
    return {
        "n_episodes": int(n),
        "score_mean": float(table["score"].mean()),
        "steps_mean": float(table["steps"].mean()),
        "length_mean": float(table["length"].mean()),
        "frac_wall": float((table["cause"] == "wall").mean()),
        "frac_self": float((table["cause"] == "self").mean()),
        "frac_starve": float((table["cause"] == "starve").mean()),
        "frac_win": float((table["cause"] == "win").mean()),
        "frac_truncated": float(table["truncated"].astype(bool).mean()),
        "frac_cut": float(table["cut"].astype(bool).mean()),
        "tail_without_food_mean": float(table["tail_without_food"].mean()),
        "loop_fraction": float(looping.mean()),
        "loop_steps_mean": float(table["loop_steps"].mean()),
        "shaping_reward_mean": float(table["shaping_reward"].mean()),
        "border_fraction_mean": float(table["border_fraction"].mean()),
        "q_max_mean": float(np.nanmean(q)) if np.isfinite(q).any() else float("nan"),
    }


def hacking_candidates(trajectories: Sequence[Trajectory], min_loop_steps: int = MIN_LOOP_STEPS) -> list[Trajectory]:
    """Episodes that end in a loop of at least ``min_loop_steps`` states,
    longest loop first (ties: smaller seed first)."""
    scored = []
    for traj in trajectories:
        stats = loop_stats(traj)
        if stats["period"] > 0 and stats["loop_steps"] >= min_loop_steps:
            scored.append((-stats["loop_steps"], int(traj.seed), traj))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [traj for _, _, traj in scored]


# ------------------------------------------------------------ last-steps figure
def last_steps_figure(trajectories: Sequence[Trajectory], k: int = 10, title: str | None = None) -> Figure:
    """One board per episode (at most 8): the body at the end (head darker),
    the last ``k`` head positions as a path and the final food."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k!r}")
    if not trajectories:
        raise ValueError("last_steps_figure needs at least one trajectory")
    if len(trajectories) > 8:
        raise ValueError(f"at most 8 episodes per figure, got {len(trajectories)}")
    n = len(trajectories)
    fig = Figure(figsize=(2.6 * n + 0.4, 3.0), dpi=DPI, facecolor=SURFACE, layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, n, squeeze=False)[0]
    for ax, traj in zip(axes, trajectories):
        W = int(traj.env["W"])
        ax.set_facecolor(SURFACE)
        for r in range(W + 1):
            ax.axhline(r - 0.5, color=GRID_COLOR, linewidth=0.5, zorder=1)
            ax.axvline(r - 0.5, color=GRID_COLOR, linewidth=0.5, zorder=1)
        for i, (row, col) in enumerate(traj.final_cells):
            ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1.0, 1.0, facecolor=HEAD_COLOR if i == 0 else BODY_COLOR,
                                   edgecolor="none", zorder=2))
        heads = np.asarray(traj.heads, dtype=np.int64).reshape(-1, 2)
        path = heads[-(k + 1):]
        ax.plot(path[:, 1], path[:, 0], color=PATH_COLOR, linewidth=1.5, marker=".", markersize=4, zorder=3)
        food = np.asarray(traj.foods, dtype=np.int64).reshape(-1, 2)[-1]
        ax.plot([food[1]], [food[0]], marker="o", markersize=6, color=FOOD_COLOR, linestyle="none", zorder=4)
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(W - 0.5, -0.5)  # row 0 at the top, as the board is printed
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_color(INK_SECONDARY)
        ax.set_title(f"seed {traj.seed}: {traj.cause}, score {traj.score}", fontsize=FONT_SIZE, color=INK)
    fig.legend(
        handles=[
            Line2D([], [], color=PATH_COLOR, marker=".", linewidth=1.5, label=f"last {k} head positions"),
            Line2D([], [], color=FOOD_COLOR, marker="o", linestyle="none", label="food"),
        ],
        loc="outside lower left", frameon=False, ncols=2, fontsize=FONT_SIZE, labelcolor=INK,
    )
    if title is not None:
        fig.suptitle(title, x=0.015, ha="left", fontsize=FONT_SIZE + 2, color=INK)
    return fig


def plot_last_steps(trajectories: Sequence[Trajectory], out: str | Path, k: int = 10, title: str | None = None) -> Path:
    """Save :func:`last_steps_figure` as a PNG."""
    return _save_png(last_steps_figure(trajectories, k=k, title=title), Path(out))


# ------------------------------------------------------------------ checkpoints
def checkpoint_files(run_dir: str | Path) -> list[tuple[int, Path]]:
    """``(env_steps, path)`` of every ``checkpoints/step_<digits>.pt`` of a run, in step order."""
    found = []
    for path in (Path(run_dir) / "checkpoints").glob("step_*.pt"):
        match = re.fullmatch(r"step_(\d+)", path.stem)
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found)


def resolve_checkpoint(run_dir: str | Path, which: str) -> tuple[int, Path]:
    """The network a ``--checkpoint`` word names: ``final`` is ``model.pt``
    at the run's training budget; ``early`` / ``mid`` / ``late`` the
    checkpoint nearest to 10 / 50 / 100 % of the last checkpoint's step (ties
    to the smaller step); a decimal string an exact env-steps count."""
    run_dir = Path(run_dir)
    which = str(which)
    if which == "final":
        from snake_rl.config import load_config

        cfg = load_config(run_dir / "config.yaml")
        return int(cfg.train.total_env_steps), run_dir / "model.pt"
    files = checkpoint_files(run_dir)
    if which in ("early", "mid", "late"):
        if not files:
            raise ValueError(f"{run_dir}: no checkpoint files under checkpoints/")
        target = {"early": 0.1, "mid": 0.5, "late": 1.0}[which] * files[-1][0]
        return min(files, key=lambda item: (abs(item[0] - target), item[0]))
    if which.isdigit():
        for step, path in files:
            if step == int(which):
                return step, path
        raise ValueError(f"{run_dir}: no checkpoint at {int(which)} env steps")
    raise ValueError(f"unknown checkpoint {which!r}: expected one of {CHECKPOINT_ALIASES} or an env-steps count")


# --------------------------------------------------------- per-player analysis
def analyse_player(
    policy,
    cfg: EnvConfig,
    out_dir: str | Path,
    *,
    n_episodes: int = DEFAULT_EPISODES,
    seed: int = DEFAULT_SEED,
    max_steps: int | None = None,
    record_top: int = 3,
    k_last: int = 10,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Roll out ``policy`` and write the behaviour files of one player into
    ``out_dir`` (module docstring): ``episodes.csv``, ``heatmap.npy`` /
    ``heatmap.png``, ``last_steps.png`` (collision deaths only; skipped when
    there is none), ``hack_<rank>_<seed>.json`` for the first ``record_top``
    hacking candidates and ``summary.json``. Returns the summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trajectories = rollouts(policy, cfg, n_episodes=n_episodes, seed=seed, max_steps=max_steps)
    table = episode_table(trajectories, cfg.reward)
    table.to_csv(out_dir / "episodes.csv", index=False)
    heat = visit_heatmap(trajectories, cfg.W)
    np.save(out_dir / "heatmap.npy", heat)
    label = str(getattr(policy, "name", type(policy).__name__))
    plot_heatmaps([(label, heat)], out_dir / "heatmap.png")
    collisions = [t for t in trajectories if t.cause in COLLISION_CAUSES][:8]
    if collisions:
        plot_last_steps(collisions, out_dir / "last_steps.png", k=k_last)
    written = []
    for rank, traj in enumerate(hacking_candidates(trajectories)[: max(0, int(record_top))], start=1):
        name = f"hack_{rank}_{traj.seed}.json"
        traj.to_recording().save(out_dir / name)
        written.append(name)
    summary: dict[str, Any] = summarise(table)
    summary.update(
        {
            "player": label,
            "env": env_config_to_dict(cfg),
            "seed": int(seed),
            "n_episodes": int(n_episodes),
            "max_steps": max_steps,
            "git": git_hash(),
            "hacking_cases": written,
        }
    )
    if meta:
        summary.update(meta)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


# ----------------------------------------------------------------------- CLI
def _seed_dirs(exp_dir: Path) -> list[Path]:
    dirs = [d for d in exp_dir.iterdir() if d.is_dir() and d.name.isdigit() and (d / "config.yaml").is_file()]
    return sorted(dirs, key=lambda d: int(d.name))


def _aggregate_summary(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in SUMMARY_KEYS:
        values = [s[key] for s in summaries if isinstance(s.get(key), (int, float)) and not isinstance(s.get(key), bool)]
        if len(values) != len(summaries):
            continue
        mean, half = mean_ci95(np.asarray(values, dtype=np.float64))
        out[f"{key}_mean"], out[f"{key}_ci95"] = mean, half
    return out


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Behaviour analysis of a player: heat-map, loops / hacking cases, last steps.")
    who = p.add_mutually_exclusive_group(required=True)
    who.add_argument("--player", metavar="SPEC", help="baseline name, run directory, *.pt checkpoint or *.npz table")
    who.add_argument("--exp", metavar="DIR", help="experiment directory: every <seed>/ run is analysed")
    p.add_argument("--checkpoint", default="final", metavar="WHICH",
                   help="with a run directory: early | mid | late | final | <env steps> (default: final)")
    p.add_argument("--config", default=None, metavar="FILE",
                   help="experiment YAML whose env block is the board (required for a baseline player)")
    p.add_argument("--episodes", type=int, default=DEFAULT_EPISODES, metavar="N")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, metavar="S", help="episode i uses seed S + i")
    p.add_argument("--max-steps", type=int, default=None, metavar="N", help="cut an episode after N steps")
    p.add_argument("--record-top", type=int, default=3, metavar="K", help="GUI recordings of the K longest loops")
    p.add_argument("--k-last", type=int, default=10, metavar="K", help="head positions drawn in last_steps.png")
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--out-dir", required=True, metavar="DIR")
    args = p.parse_args(argv)

    from snake_rl.config import load_config
    from snake_rl.gui.app import load_player

    common = dict(n_episodes=args.episodes, seed=args.seed, max_steps=args.max_steps, record_top=args.record_top,
                  k_last=args.k_last)

    if args.exp is not None:
        exp_dir = Path(args.exp)
        if not exp_dir.is_dir():
            p.error(f"{exp_dir} is not a directory")
        if args.config is not None:
            p.error("--config is not used with --exp: every run carries its own configuration")
        seed_dirs = _seed_dirs(exp_dir)
        if not seed_dirs:
            p.error(f"{exp_dir}: no <seed>/ run directory with a config.yaml")
        out_dir = Path(args.out_dir)
        summaries, heats = [], []
        for run_dir in seed_dirs:
            step, path = resolve_checkpoint(run_dir, args.checkpoint)
            loaded = load_player(path, device=args.device)
            cfg = load_config(run_dir / "config.yaml").env
            meta = {"checkpoint": step, "source": str(path), "run": str(run_dir)}
            summaries.append(analyse_player(loaded.player, cfg, out_dir / run_dir.name, meta=meta, **common))
            heats.append((f"seed {run_dir.name}", np.load(out_dir / run_dir.name / "heatmap.npy")))
        mean_heat = np.mean([h for _, h in heats], axis=0)
        np.save(out_dir / "heatmap_mean.npy", mean_heat)
        plot_heatmaps(heats + [("mean", mean_heat)], out_dir / "heatmap_mean.png", title=exp_dir.name)
        aggregate = _aggregate_summary(summaries)
        aggregate.update(
            {"n_seeds": len(summaries), "seeds": [int(d.name) for d in seed_dirs], "exp": exp_dir.name,
             "checkpoint": args.checkpoint}
        )
        out = out_dir / "summary.json"
        out.write_text(json.dumps(aggregate, indent=1), encoding="utf-8")
        print(out)
        return 0

    spec = Path(args.player)
    if spec.is_dir():
        if args.config is not None:
            p.error("--config is not used with a run directory: the run carries its own configuration")
        step, path = resolve_checkpoint(spec, args.checkpoint)
        loaded = load_player(path, device=args.device)
        cfg = load_config(spec / "config.yaml").env
    else:
        if args.checkpoint != "final":
            p.error("--checkpoint only applies to a run directory")
        step = None
        loaded = load_player(args.player, device=args.device)
        if loaded.env_cfg is not None:
            if args.config is not None:
                p.error("--config is not used with a player that comes from a run")
            cfg = loaded.env_cfg
        else:
            if args.config is None:
                p.error("--config FILE is required for a player without a run configuration (a baseline)")
            cfg = load_config(args.config).env
    meta = {"checkpoint": step, "source": str(args.player)}
    analyse_player(loaded.player, cfg, args.out_dir, meta=meta, **common)
    print(Path(args.out_dir) / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

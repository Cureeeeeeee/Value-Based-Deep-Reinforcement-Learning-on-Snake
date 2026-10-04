"""Throughput benchmark of the vectorised Snake environment (PLAN.md section 4, M3).

The env-steps per second measured here are the environment-only upper bound on
data collection and evaluation speed. They are NOT the training budget: in a
DQN loop the gradient updates dominate the wall-clock (docs/PROGRESS.md, M3).
The protocol is fixed and stated in full:

* A cell is ``(observation, policy, n_envs)`` on ``VecSnakeEnv``. The scalar
  reference ``SnakeEnv`` runs the same observation and policy as the baseline
  of the speed-up column.
* Observations: ``A`` (11 features), ``B8`` (8 planes), ``B3`` (head / body /
  food planes only). Reward R0, relative actions, default starvation limit.
* The policy decides which regime is measured:

  - ``random``: uniformly random actions. Short episodes, reset-heavy; the
    early-training regime.
  - ``safe``: uniformly random among the actions whose target cell is not
    dangerous (``obs.cell_is_danger``), any action if there is none. Longer
    episodes, few resets; closer to mid-training.

* Only the environment is inside the timer. Every ``env.step`` call is timed
  on its own with ``time.perf_counter``; drawing random numbers, choosing
  actions and the bookkeeping happen between the timed calls. The vectorised
  step contains its auto-reset, so for the scalar environment the ``reset()``
  that follows a finished episode is timed as well.
* A measurement builds a fresh environment, runs ``--warmup`` untimed
  vec-steps (default ``2 H``, long enough for the boards to drift out of
  phase) and then times vec-steps until its share of ``--seconds`` has
  accumulated.
* The whole grid is measured ``--passes`` times, one pass after the other, so
  every cell is sampled at different moments and in different allocator
  states. This matters: every step returns freshly allocated observation
  arrays, and once such an array reaches roughly 0.5 MB the same step can
  cost twice as much in one pass as in the next (the system allocator
  sometimes, and above about 1 MB always, pages the block in anew). One
  window would hide that.
* Throughput is the mean over all passes (total env-steps / total timed
  seconds), not a best-of: a training run pays the mean. The slowest and the
  fastest pass are printed next to it. The garbage collector stays on.

Reported per cell: env-steps/s (``n_envs * vec_steps / elapsed``) with its
range over the passes, microseconds per vec-step, the speed-up over the
scalar environment and - so that the reader can see which regime was
measured - the mean length of the episodes that ended inside the timed
windows and the food eaten per 1000 env-steps.

Usage (from the repo root)::

    python scripts/bench_vec_env.py            # full grid, about 80 seconds
    python scripts/bench_vec_env.py --quick    # smoke run, a few seconds

Prints a markdown table and writes all rows plus metadata (machine, versions,
git hash, timestamp) to ``runs/bench/vec_env.json``. An existing file is never
overwritten: the new one gets a numeric suffix (``vec_env_1.json``, ...).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from snake_rl.env import EnvConfig, ObsConfig, SnakeEnv
from snake_rl.env.common import DELTAS, ActionMode, head_positions, n_actions, next_heading
from snake_rl.env.obs import cell_is_danger
from snake_rl.env.vec_env import VecSnakeEnv
from snake_rl.utils import git_hash

OBS_CONFIGS = {
    "A": ObsConfig(kind="A"),
    "B8": ObsConfig(kind="B"),
    "B3": ObsConfig(kind="B", age_plane=False, heading_planes=False),
}
POLICIES = ("random", "safe")

# Grid of the full run and of the ``--quick`` smoke run.
FULL_N_ENVS = (1, 16, 64, 128, 256, 512, 1024)
FULL_OBS = ("A", "B8", "B3")
FULL_SECONDS = 1.0
FULL_PASSES = 5
FULL_OUT = "runs/bench/vec_env.json"
QUICK_N_ENVS = (1, 16, 64)
QUICK_OBS = ("A", "B8")
QUICK_SECONDS = 0.2
QUICK_PASSES = 2
QUICK_OUT = "runs/bench/vec_env_quick.json"

#: A window is timed for at least this many vec-steps, however short it is.
MIN_TIMED_STEPS = 20

#: What one timed step reports: seconds spent inside the environment, food
#: eaten, episodes ended, summed length of the episodes that ended.
StepResult = tuple[float, int, int, int]


@dataclass(frozen=True)
class Cell:
    """One line of the table."""

    env: str  # "scalar" (SnakeEnv) or "vec" (VecSnakeEnv)
    obs: str
    policy: str
    n_envs: int


@dataclass
class Window:
    """Raw counts of one timed window of one cell."""

    vec_steps: int = 0
    elapsed: float = 0.0  # seconds inside the environment
    food: int = 0
    episodes: int = 0  # episodes that ended inside the window
    episode_steps: int = 0  # summed length of those episodes


# ------------------------------------------------------------------ policies
def choose_actions(
    policy: str,
    rng: np.random.Generator,
    body_age: np.ndarray,
    heading: np.ndarray,
    action_mode: ActionMode,
) -> np.ndarray:
    """One action per board, ``(N,)`` int64, from the batched state.

    A uniform key is drawn per (board, action) and the argmax is played, i.e.
    a uniformly random action. ``safe`` adds 1 to the key of every action
    whose target cell is not dangerous, so the argmax is uniform over the safe
    actions if there is one and uniform over all actions otherwise.
    """
    n = n_actions(action_mode)
    keys = rng.random((body_age.shape[0], n))
    if policy == "safe":
        head = head_positions(body_age)
        for a in range(n):
            target = head + DELTAS[next_heading(heading, a, action_mode)]
            keys[:, a] += ~cell_is_danger(body_age, target)
    return keys.argmax(axis=1)


def action_rng(seed: int) -> np.random.Generator:
    """Action stream; seeded apart from every board's food stream (``seed + i``)."""
    return np.random.default_rng([seed, 1])


# ------------------------------------------------------------------ steppers
def vec_stepper(cfg: EnvConfig, n_envs: int, policy: str, seed: int) -> Callable[[], StepResult]:
    """Function that advances a fresh ``VecSnakeEnv`` by one vec-step per call."""
    env = VecSnakeEnv(cfg, n_envs=n_envs, seed=seed)
    env.reset()
    rng = action_rng(seed)

    def advance() -> StepResult:
        actions = choose_actions(policy, rng, env.body_age, env.heading, cfg.action_mode)
        score_before = int(env.score.sum())
        t0 = time.perf_counter()
        obs, reward, terminated, truncated, info = env.step(actions)
        dt = time.perf_counter() - t0
        done = info["done"]  # info holds the counters as they stood before the auto-reset
        food = int(info["score"].sum()) - score_before
        return dt, food, int(done.sum()), int(info["steps"][done].sum())

    return advance


def scalar_stepper(cfg: EnvConfig, policy: str, seed: int) -> Callable[[], StepResult]:
    """Function that advances a fresh ``SnakeEnv`` by one step per call.

    The timer covers the ``reset()`` after a finished episode, because the
    vectorised step it is compared with performs that reset itself.
    """
    env = SnakeEnv(cfg)
    env.reset(seed=seed)
    rng = action_rng(seed)

    def advance() -> StepResult:
        heading = np.array([env.heading])
        action = int(choose_actions(policy, rng, env.body_age[None], heading, cfg.action_mode)[0])
        score_before = env.score
        t0 = time.perf_counter()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            obs, _ = env.reset()  # no seed: the food stream continues
        dt = time.perf_counter() - t0
        return dt, info["score"] - score_before, int(done), info["steps"] if done else 0

    return advance


def make_stepper(cell: Cell, W: int, seed: int) -> Callable[[], StepResult]:
    cfg = EnvConfig(W=W, obs=OBS_CONFIGS[cell.obs])
    if cell.env == "scalar":
        return scalar_stepper(cfg, cell.policy, seed)
    return vec_stepper(cfg, cell.n_envs, cell.policy, seed)


# --------------------------------------------------------------- measurement
def measure(advance: Callable[[], StepResult], warmup: int, seconds: float) -> Window:
    """Run ``warmup`` untimed steps, then time steps until ``seconds`` have accumulated."""
    for _ in range(warmup):
        advance()
    win = Window()
    while win.elapsed < seconds or win.vec_steps < MIN_TIMED_STEPS:
        dt, food, episodes, episode_steps = advance()
        win.vec_steps += 1
        win.elapsed += dt
        win.food += food
        win.episodes += episodes
        win.episode_steps += episode_steps
    return win


def make_grid(obs_names: list[str], policies: list[str], n_envs_list: list[int]) -> list[Cell]:
    """Cells in table order: per (obs, policy) the scalar baseline, then every n_envs."""
    grid: list[Cell] = []
    for obs_name in obs_names:
        for policy in policies:
            grid.append(Cell("scalar", obs_name, policy, 1))
            grid.extend(Cell("vec", obs_name, policy, n) for n in n_envs_list)
    return grid


def run_passes(
    grid: list[Cell], W: int, passes: int, warmup: int, seconds: float, seed: int
) -> dict[Cell, list[Window]]:
    """Measure the whole grid ``passes`` times; each window lasts ``seconds / passes``."""
    windows: dict[Cell, list[Window]] = {cell: [] for cell in grid}
    for k in range(passes):
        t0 = time.perf_counter()
        for cell in grid:
            windows[cell].append(measure(make_stepper(cell, W, seed + k), warmup, seconds / passes))
        print(f"pass {k + 1}/{passes}: {time.perf_counter() - t0:.1f} s", file=sys.stderr, flush=True)
    return windows


def summarise(cell: Cell, windows: list[Window], W: int) -> dict[str, Any]:
    """Pool the windows of one cell into its table / JSON row."""
    vec_steps = sum(w.vec_steps for w in windows)
    elapsed = sum(w.elapsed for w in windows)
    env_steps = cell.n_envs * vec_steps
    episodes = sum(w.episodes for w in windows)
    return {
        **asdict(cell),
        "obs_shape": list(OBS_CONFIGS[cell.obs].shape(W)),
        "vec_steps": vec_steps,
        "elapsed_s": elapsed,
        "env_steps_per_s": env_steps / elapsed,
        "env_steps_per_s_passes": [cell.n_envs * w.vec_steps / w.elapsed for w in windows],
        "us_per_vec_step": 1e6 * elapsed / vec_steps,
        "episodes": episodes,
        "mean_episode_length": sum(w.episode_steps for w in windows) / episodes if episodes else None,
        "food_per_1000_steps": 1000.0 * sum(w.food for w in windows) / env_steps,
    }


def make_rows(windows: dict[Cell, list[Window]], W: int) -> list[dict[str, Any]]:
    """One row per cell, with the speed-up over the scalar row of the same obs and policy."""
    rows = [summarise(cell, wins, W) for cell, wins in windows.items()]
    scalar = {(r["obs"], r["policy"]): r["env_steps_per_s"] for r in rows if r["env"] == "scalar"}
    for r in rows:
        r["speedup"] = r["env_steps_per_s"] / scalar[r["obs"], r["policy"]]
    return rows


# -------------------------------------------------------------------- output
def detail_table(rows: list[dict[str, Any]]) -> str:
    out = [
        "| env | obs | policy | n_envs | env-steps/s | slowest - fastest pass | us/vec-step | ep len "
        "| food/1k steps | speed-up |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        ep_len = "-" if r["mean_episode_length"] is None else f"{r['mean_episode_length']:.1f}"
        lo, hi = min(r["env_steps_per_s_passes"]), max(r["env_steps_per_s_passes"])
        out.append(
            f"| {r['env']} | {r['obs']} | {r['policy']} | {r['n_envs']} | {r['env_steps_per_s']:,.0f} | "
            f"{lo:,.0f} - {hi:,.0f} | {r['us_per_vec_step']:.1f} | {ep_len} | "
            f"{r['food_per_1000_steps']:.1f} | {r['speedup']:.1f}x |"
        )
    return "\n".join(out)


def summary_table(rows: list[dict[str, Any]]) -> str:
    """Compact view of the same rows: thousand env-steps/s, one column per n_envs."""
    columns: list[tuple[str, int]] = []  # (env, n_envs) in order of first appearance
    lines: dict[tuple[str, str], dict[tuple[str, int], float]] = {}
    for r in rows:
        column = (r["env"], r["n_envs"])
        if column not in columns:
            columns.append(column)
        lines.setdefault((r["obs"], r["policy"]), {})[column] = r["env_steps_per_s"]
    names = ["scalar" if env == "scalar" else f"N={n}" for env, n in columns]
    out = ["| obs | policy | " + " | ".join(names) + " |", "|---|---|" + "---:|" * len(columns)]
    for (obs_name, policy), cells in lines.items():
        values = " | ".join(f"{cells[column] / 1e3:,.1f}" for column in columns)
        out.append(f"| {obs_name} | {policy} | {values} |")
    return "\n".join(out)


def fresh_path(path: Path) -> Path:
    """``path`` if it is free, else the first free ``<stem>_<k><suffix>``, k = 1, 2, ..."""
    k = 0
    candidate = path
    while candidate.exists():
        k += 1
        candidate = path.with_name(f"{path.stem}_{k}{path.suffix}")
    return candidate


def save(path: Path, meta: dict[str, Any], rows: list[dict[str, Any]]) -> Path:
    """Write ``{meta, rows}`` as JSON without ever replacing an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path = fresh_path(path)
    with path.open("x", encoding="utf-8") as f:  # "x": fail rather than overwrite
        json.dump({"meta": meta, "rows": rows}, f, indent=1)
    return path


# ----------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Throughput benchmark of VecSnakeEnv against the scalar SnakeEnv.")
    p.add_argument("--W", type=int, default=10, help="grid side (default 10)")
    p.add_argument("--n-envs", type=int, nargs="+", help=f"default {' '.join(map(str, FULL_N_ENVS))}")
    p.add_argument("--obs", nargs="+", choices=list(OBS_CONFIGS), help=f"default {' '.join(FULL_OBS)}")
    p.add_argument("--policy", nargs="+", choices=POLICIES, help=f"default {' '.join(POLICIES)}")
    p.add_argument(
        "--seconds", type=float, help=f"timed environment seconds per cell, all passes together (default {FULL_SECONDS})"
    )
    p.add_argument("--passes", type=int, help=f"times the whole grid is measured (default {FULL_PASSES})")
    p.add_argument("--warmup", type=int, help="untimed vec-steps before each window (default 2 * starve limit)")
    p.add_argument("--seed", type=int, default=0, help="pass k seeds board i with seed + k + i (default 0)")
    p.add_argument("--out", help=f"JSON file (default {FULL_OUT}; {QUICK_OUT} with --quick)")
    p.add_argument("--quick", action="store_true", help="smoke run: small grid, short windows")
    args = p.parse_args(argv)

    # Options left out take the default of the chosen grid; repeated values count once.
    quick = args.quick
    n_envs_list = list(dict.fromkeys(args.n_envs or (QUICK_N_ENVS if quick else FULL_N_ENVS)))
    obs_names = list(dict.fromkeys(args.obs or (QUICK_OBS if quick else FULL_OBS)))
    policies = list(dict.fromkeys(args.policy or POLICIES))
    seconds = args.seconds if args.seconds is not None else (QUICK_SECONDS if quick else FULL_SECONDS)
    passes = args.passes if args.passes is not None else (QUICK_PASSES if quick else FULL_PASSES)
    out = Path(args.out or (QUICK_OUT if quick else FULL_OUT))
    cfg = EnvConfig(W=args.W)
    warmup = args.warmup if args.warmup is not None else 2 * cfg.H
    if passes < 1:
        p.error("--passes must be at least 1")

    meta = {
        "W": args.W,
        "starve_limit": cfg.H,
        "action_mode": cfg.action_mode,
        "reward": cfg.reward.variant,
        "seconds_per_cell": seconds,
        "passes": passes,
        "warmup_vec_steps": warmup,
        "seed": args.seed,
        "quick": quick,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "git": git_hash(),
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "argv": list(sys.argv[1:] if argv is None else argv),
    }
    print(
        f"W={meta['W']}  H={meta['starve_limit']}  reward={meta['reward']}  actions={meta['action_mode']}  "
        f"{seconds:g} s/cell in {passes} passes, {warmup} warm-up vec-steps each  seed={args.seed}"
    )
    print(
        f"numpy {meta['numpy']} | python {meta['python']} | {meta['platform']} | {meta['processor']} | "
        f"{meta['cpu_count']} logical CPUs | git {meta['git']} | {meta['timestamp']}"
    )
    grid = make_grid(obs_names, policies, n_envs_list)
    rows = make_rows(run_passes(grid, args.W, passes, warmup, seconds, args.seed), args.W)
    print()
    print(detail_table(rows))
    print()
    print("Thousand env-steps/s:")
    print()
    print(summary_table(rows))
    print()
    print(f"-> {save(out, meta, rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

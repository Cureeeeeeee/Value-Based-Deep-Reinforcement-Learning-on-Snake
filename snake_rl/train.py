"""Config-driven training entry point (PLAN.md milestone M4: tabular Q-learning).

Command line::

    python -m snake_rl.train --config configs/tabular_q_8x8.yaml --seed 0
        [--exp NAME] [--set key=value ...] [--runs-dir runs] [--quiet]

``--set`` applies ``dotted.key=value`` overrides to the configuration file
(``snake_rl.config.load_config``); ``--exp NAME`` is shorthand for
``--set exp=NAME``. One progress line is printed per evaluation unless
``--quiet``. Exit code 0 on success, 2 for an invalid configuration, 1 if the
run directory already holds a run.

Run directory
-------------
``train(cfg, seed, runs_dir)`` writes ``<runs_dir>/<exp>/<seed>/``; the
directory is named by the bare seed number, e.g. ``runs/tabular_q_W10/0/``::

    config.yaml      the full configuration (every field, defaults included)
                     plus a ``run`` block: seed, env_seed, git hash, start
                     time, python / numpy / platform versions, argv. Written
                     BEFORE training starts. ``load_config`` reads it back
                     (the ``run`` block is ignored), so the file of a finished
                     run is the recipe to repeat it.
    metrics.csv      the evaluation curve: one row per evaluation, flushed
                     after every row (columns below).
    q_table.npz      ``TabularQAgent.save`` after the last update.
    final_eval.json  ``EvalResult`` of the last evaluation, the one at
                     ``total_env_steps``: the per-episode records behind the
                     results table. Written last and in one piece; its
                     presence is what marks a run as finished.

A run is never overwritten: ``train`` raises ``FileExistsError`` if the
directory already holds any of these files, finished or not.

Tabular training loop
---------------------
::

    env = VecSnakeEnv(cfg.env, n_envs, seed=env_seed_for(seed, n_envs)); obs = env.reset()
    explore_rng = default_rng([seed, 1])
    evaluate                                             # row 0: before any update
    for k = 0, 1, ...:                                   # env_steps_before = k * n_envs
        epsilon = linear_schedule(env_steps_before, eps_start, eps_end,
                                  eps_decay_fraction * total_env_steps)
        actions = agent.act_epsilon(obs, epsilon, explore_rng)
        next_obs, reward, terminated, truncated, info = env.step(actions)
        agent.update(obs, actions, reward, info["final_obs"], terminated)
        obs = next_obs
        if (k + 1) * n_envs is a multiple of eval_every: evaluate

Two details carry the semantics. ``env.step`` auto-resets, so ``next_obs``
already belongs to the next episode wherever one ended; the update takes the
true successor ``info["final_obs"]``. And the update bootstraps unless the
transition ``terminated`` (wall, self collision, win): a starvation
truncation is a time limit the agent cannot observe, not a terminal state, so
``truncated`` is deliberately unused.

An evaluation is ``evaluate_batched(agent, cfg.env, eval_episodes, eval_seed)``:
greedy (epsilon = 0), always on the same boards, which are the boards every
baseline is judged on. ``metrics.csv`` therefore has
``total_env_steps / eval_every + 1`` rows.

Seeds
-----
The run seed fixes everything. The training boards use the generators
``env_seed + i`` with ``env_seed = env_seed_for(seed, n_envs)`` (disjoint
between runs and from the evaluation boards, see ``snake_rl.config``);
exploration draws from its own generator ``default_rng([seed, 1])``. (numpy
seeds ``[seed, 1]`` like the integer ``seed + 2**32``, which no food stream
reaches for run seeds below 42,948.) The same config and seed give identical
``q_table.npz`` arrays, ``final_eval.json`` episodes and ``metrics.csv``
(except ``wall_time_s``), in the same process or in a fresh one.

metrics.csv (tabular Q-learning)
--------------------------------
``env_steps``, ``vec_steps``: position of the row (``env_steps = vec_steps *
n_envs``). ``epsilon``: the value used for the most recent vec-step
(``eps_start`` in row 0). ``eval_*``: ``EvalResult.summary()`` of that
evaluation; ``eval_ci95`` is the episode-level normal-approximation
half-width. ``train_episodes`` / ``train_mean_score``: number and mean score
of the *training* (epsilon-greedy) episodes that finished since the previous
row. ``mean_abs_td``: mean of the ``update()`` return values since the
previous row. ``cells_visited``: (state, action) cells updated at least once.
``wall_time_s``: seconds since training started. Cells without a value (the
training statistics of row 0, the mean score when no episode finished) are
empty strings; numbers are written with ``repr``, so nothing is rounded.

Adding an agent kind (M5): write a trainer with the signature of
``_train_tabular_q`` that owns its metrics columns and checkpoint files and
register it in ``_TRAINERS``; ``train``, the run-directory bookkeeping,
``MetricsWriter`` and ``eval_columns`` are shared.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from snake_rl.agents.schedules import linear_schedule
from snake_rl.agents.tabular_q import TabularQAgent
from snake_rl.config import ExperimentConfig, config_from_dict, config_to_dict, env_seed_for, load_config
from snake_rl.env.vec_env import VecSnakeEnv
from snake_rl.evaluate import EvalResult, evaluate_batched
from snake_rl.utils import git_hash

CONFIG_FILE = "config.yaml"
METRICS_FILE = "metrics.csv"
Q_TABLE_FILE = "q_table.npz"
FINAL_EVAL_FILE = "final_eval.json"
#: Everything a tabular run writes. A directory holding any of them is a run.
RUN_FILES = (CONFIG_FILE, METRICS_FILE, Q_TABLE_FILE, FINAL_EVAL_FILE)

#: Evaluation columns shared by every agent kind.
EVAL_COLUMNS = (
    "eval_mean_score",
    "eval_ci95",
    "eval_median_score",
    "eval_max_score",
    "eval_mean_steps",
    "eval_frac_wall",
    "eval_frac_self",
    "eval_frac_starve",
    "eval_frac_win",
)
#: Header of ``metrics.csv`` for tabular Q-learning, in file order.
TABULAR_COLUMNS = (
    "env_steps",
    "vec_steps",
    "epsilon",
    *EVAL_COLUMNS,
    "train_episodes",
    "train_mean_score",
    "mean_abs_td",
    "cells_visited",
    "wall_time_s",
)


# ---------------------------------------------------------------- the loop
def _train_tabular_q(cfg: ExperimentConfig, seed: int, run_dir: Path, quiet: bool) -> None:
    """Tabular Q-learning on observation A (the loop of the module docstring)."""
    ac, tc = cfg.agent, cfg.train
    env = VecSnakeEnv(cfg.env, tc.n_envs, seed=env_seed_for(seed, tc.n_envs))
    agent = TabularQAgent(env.n_actions, ac)
    explore_rng = np.random.default_rng([seed, 1])  # exploration stream, apart from every food stream
    decay_steps = ac.eps_decay_fraction * tc.total_env_steps

    with MetricsWriter(run_dir / METRICS_FILE, TABULAR_COLUMNS) as metrics:
        evaluate_now = partial(
            _evaluate_tabular, agent, cfg, metrics, started=time.perf_counter(), label=f"{cfg.exp}/{seed}", quiet=quiet
        )
        obs = env.reset()
        last_eval = evaluate_now(0, ac.eps_start, TrainWindow())  # before any update
        window = TrainWindow()
        for k in range(tc.total_env_steps // tc.n_envs):
            epsilon = linear_schedule(k * tc.n_envs, ac.eps_start, ac.eps_end, decay_steps)
            actions = agent.act_epsilon(obs, epsilon, explore_rng)
            next_obs, reward, terminated, truncated, info = env.step(actions)
            # True successor (not the post-reset observation); bootstrap unless
            # terminated, so a starvation truncation keeps its future value.
            abs_td = agent.update(obs, actions, reward, info["final_obs"], terminated)
            obs = next_obs
            window.add(info, abs_td)

            env_steps = (k + 1) * tc.n_envs
            if env_steps % tc.eval_every == 0:
                last_eval = evaluate_now(env_steps, epsilon, window)
                window = TrainWindow()

    agent.save(run_dir / Q_TABLE_FILE)
    _save_final_eval(last_eval, run_dir)


@dataclass
class TrainWindow:
    """Training statistics accumulated between two metrics rows."""

    episodes: int = 0
    score_sum: int = 0
    updates: int = 0
    abs_td_sum: float = 0.0

    def add(self, info: Mapping[str, np.ndarray], abs_td: float) -> None:
        """One vec-step: the episodes it finished and the mean |TD error| of its update."""
        done = info["done"]
        self.episodes += int(done.sum())
        self.score_sum += int(info["score"][done].sum())
        self.updates += 1
        self.abs_td_sum += abs_td

    @property
    def mean_score(self) -> float | None:
        return self.score_sum / self.episodes if self.episodes else None

    @property
    def mean_abs_td(self) -> float | None:
        return self.abs_td_sum / self.updates if self.updates else None


def _evaluate_tabular(
    agent: TabularQAgent,
    cfg: ExperimentConfig,
    metrics: "MetricsWriter",
    env_steps: int,
    epsilon: float,
    window: TrainWindow,
    *,
    started: float,
    label: str,
    quiet: bool,
) -> EvalResult:
    """Greedy evaluation on the fixed evaluation boards; appends one metrics row."""
    tc = cfg.train
    result = evaluate_batched(agent, cfg.env, tc.eval_episodes, tc.eval_seed)
    row = {
        "env_steps": env_steps,
        "vec_steps": env_steps // tc.n_envs,
        "epsilon": epsilon,
        **eval_columns(result),
        "train_episodes": window.episodes,
        "train_mean_score": window.mean_score,
        "mean_abs_td": window.mean_abs_td,
        "cells_visited": int(np.count_nonzero(agent.visits)),
        "wall_time_s": round(time.perf_counter() - started, 3),
    }
    metrics.write(row)
    if not quiet:
        print(_progress_line(label, tc.total_env_steps, row), flush=True)
    return result


#: ``agent.kind`` -> trainer ``(cfg, seed, run_dir, quiet)``. A trainer writes
#: its metrics and checkpoints into ``run_dir`` and ``final_eval.json`` last.
_TRAINERS: dict[str, Callable[[ExperimentConfig, int, Path, bool], None]] = {
    "tabular_q": _train_tabular_q,
}


# ------------------------------------------------------------- entry point
def train(
    cfg: ExperimentConfig,
    seed: int,
    runs_dir: str | Path = "runs",
    argv: Sequence[str] | None = None,
    quiet: bool = False,
) -> Path:
    """Train one run and return its directory ``<runs_dir>/<exp>/<seed>/``.

    ``argv`` is recorded in ``config.yaml`` (the command-line arguments when
    called from :func:`main`). ``ValueError`` for an invalid config or seed or
    an agent kind without a trainer; ``FileExistsError`` if the run exists.
    """
    cfg = config_from_dict(config_to_dict(cfg))  # a hand-built config gets the checks of a file
    trainer = _TRAINERS.get(cfg.agent_kind)
    if trainer is None:
        raise ValueError(f"no trainer for agent.kind {cfg.agent_kind!r} (known: {', '.join(_TRAINERS)})")
    env_seed = env_seed_for(seed, cfg.train.n_envs)
    run_dir = Path(runs_dir) / cfg.exp / str(seed)
    _start_run(cfg, seed, env_seed, run_dir, argv)
    if not quiet:
        print(f"[{cfg.exp}/{seed}] {cfg.agent_kind} -> {run_dir}  (env_seed {env_seed})", flush=True)
    trainer(cfg, seed, run_dir, quiet)
    if not quiet:
        print(f"[{cfg.exp}/{seed}] finished -> {run_dir}", flush=True)
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m snake_rl.train",
        description="Train one run from a YAML experiment file; writes <runs-dir>/<exp>/<seed>/.",
        allow_abbrev=False,
    )
    p.add_argument("--config", required=True, help="YAML experiment file (see snake_rl/config.py)")
    p.add_argument("--seed", type=int, required=True, help="run seed (>= 0); also the name of the run directory")
    p.add_argument("--exp", help="experiment name; shorthand for --set exp=NAME")
    p.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="config overrides, e.g. --set agent.alpha=0.2 env.reward.variant=R1",
    )
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--quiet", action="store_true", help="no progress lines")
    args = p.parse_args(argv)
    if args.seed < 0:
        p.error("--seed must be >= 0")

    overrides = [item for group in args.overrides for item in group]
    if args.exp is not None:
        overrides.append(f"exp={json.dumps(args.exp)}")  # quoted: a name like 2026 stays a string
    try:
        cfg = load_config(args.config, overrides)
    except (OSError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    try:
        train(cfg, args.seed, args.runs_dir, argv=sys.argv[1:] if argv is None else argv, quiet=args.quiet)
    except FileExistsError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    return 0


# ------------------------------------------------------ run-directory files
def _start_run(cfg: ExperimentConfig, seed: int, env_seed: int, run_dir: Path, argv: Sequence[str] | None) -> None:
    """Claim the run directory by writing ``config.yaml``; refuse an existing run."""
    existing = [name for name in RUN_FILES if (run_dir / name).exists()]
    if existing:
        raise FileExistsError(
            f"{run_dir} already holds a run ({', '.join(existing)}); runs are never overwritten - "
            f"use another --exp or --seed"
        )
    record = config_to_dict(cfg)
    record["run"] = {
        "seed": seed,
        "env_seed": env_seed,
        "git": git_hash(),
        "started": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "argv": None if argv is None else [str(arg) for arg in argv],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    # Mode "x" fails if the file exists: two processes cannot both claim the run.
    with open(run_dir / CONFIG_FILE, "x", encoding="utf-8") as f:
        f.write("# Written by snake_rl.train before training started. To repeat this run:\n")
        f.write("#   python -m snake_rl.train --config <this file> --seed <run.seed> --exp <new name>\n")
        yaml.safe_dump(record, f, sort_keys=False)


def _save_final_eval(result: EvalResult, run_dir: Path) -> None:
    """``final_eval.json`` marks the run as finished (the analysis and the
    batch runner look for it), so it must never exist half-written: write
    next to it, then rename."""
    partial_file = run_dir / (FINAL_EVAL_FILE + ".tmp")
    result.save(partial_file)
    os.replace(partial_file, run_dir / FINAL_EVAL_FILE)


class MetricsWriter:
    """Append-only CSV with a fixed header. The header is written on creation
    and every row is flushed, so an interrupted run keeps its curve up to the
    last evaluation. Refuses to overwrite an existing file."""

    def __init__(self, path: str | Path, columns: Sequence[str]) -> None:
        self.columns = tuple(columns)
        self._file = open(path, "x", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file, lineterminator="\n")
        self._writer.writerow(self.columns)
        self._file.flush()

    def write(self, row: Mapping[str, Any]) -> None:
        """Append one row; ``row`` must have exactly the columns of the header."""
        if set(row) != set(self.columns):
            raise KeyError(f"metrics row keys {sorted(row)} do not match the header {sorted(self.columns)}")
        self._writer.writerow([_cell(row[name]) for name in self.columns])
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "MetricsWriter":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _cell(value: Any) -> str:
    """CSV text of one value: empty for ``None``, ``repr`` for floats (exact
    round trip), plain digits for integers."""
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("booleans are not metrics values")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return repr(float(value))


def eval_columns(result: EvalResult) -> dict[str, Any]:
    """The ``eval_*`` metrics columns of one evaluation (``EvalResult.summary()``)."""
    s = result.summary()
    frac = s["cause_fractions"]
    return {
        "eval_mean_score": s["mean_score"],
        "eval_ci95": s["ci95"],
        "eval_median_score": s["median_score"],
        "eval_max_score": s["max_score"],
        "eval_mean_steps": s["mean_steps"],
        "eval_frac_wall": frac["wall"],
        "eval_frac_self": frac["self"],
        "eval_frac_starve": frac["starve"],
        "eval_frac_win": frac["win"],
    }


def _progress_line(label: str, total_env_steps: int, row: Mapping[str, Any]) -> str:
    def optional(value: float | None, spec: str) -> str:
        return "-" if value is None else format(value, spec)

    width = len(f"{total_env_steps:,}")
    return (
        f"[{label}] {row['env_steps']:>{width},}/{total_env_steps:,} steps  eps {row['epsilon']:.3f}  "
        f"eval {row['eval_mean_score']:.2f} +- {row['eval_ci95']:.2f}  "
        f"wall {row['eval_frac_wall']:.2f} self {row['eval_frac_self']:.2f} "
        f"starve {row['eval_frac_starve']:.2f} win {row['eval_frac_win']:.2f}  "
        f"train {optional(row['train_mean_score'], '.2f')} ({row['train_episodes']} ep)  "
        f"|td| {optional(row['mean_abs_td'], '.4f')}  {row['wall_time_s']:.1f}s"
    )


if __name__ == "__main__":
    raise SystemExit(main())

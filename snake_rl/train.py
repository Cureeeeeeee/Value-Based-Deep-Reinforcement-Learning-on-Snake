"""Config-driven training entry point (PLAN.md milestones M4: tabular
Q-learning, M5: DQN, M6: DQN without replay, M7: prioritised replay and the
value-estimate columns). ``agent.kind`` of the configuration selects the
trainer.

Command line::

    python -m snake_rl.train --config configs/tabular_q_8x8.yaml --seed 0
        [--exp NAME] [--set key=value ...] [--runs-dir runs] [--quiet]

``--set`` applies ``dotted.key=value`` overrides to the configuration file
(``snake_rl.config.load_config``); ``--exp NAME`` is shorthand for
``--set exp=NAME``. One progress line is printed per evaluation unless
``--quiet``. Exit code 0 on success, 2 for an invalid configuration, 1 if the
run directory already holds a run, 3 if a DQN update produced a non-finite
loss (the run is aborted, see below).

Run directory
-------------
``train(cfg, seed, runs_dir)`` writes ``<runs_dir>/<exp>/<seed>/``; the
directory is named by the bare seed number, e.g. ``runs/tabular_q_W10/0/``::

    config.yaml      the full configuration (every field, defaults included)
                     plus a ``run`` block: seed, env_seed, git hash, start
                     time, python / numpy / platform versions, argv; for dqn
                     also the torch version, the device in use (``cpu`` /
                     ``cuda``), the CUDA device name (null on the CPU) and the
                     number of torch threads. Written BEFORE training starts.
                     ``load_config`` reads it back (the ``run`` block is
                     ignored), so the file of a finished run is the recipe to
                     repeat it.
    metrics.csv      the evaluation curve: one row per evaluation, flushed
                     after every row (columns below, per agent kind).
    q_table.npz      tabular_q: ``TabularQAgent.save`` after the last update.
    checkpoints/     dqn, unless ``agent.keep_checkpoints`` is false: one
                     weights-only file ``step_<env_steps, 8 digits>.pt`` per
                     evaluation (``DQNAgent.save(include_optimizer=False)``),
                     the network that produced that metrics row.
    model.pt         dqn: ``DQNAgent.save`` (with the optimiser state) after
                     the last update.
    final_eval.json  ``EvalResult`` of the last evaluation, the one at
                     ``total_env_steps``: the per-episode records behind the
                     results table. Written last and in one piece; its
                     presence is what marks a run as finished.
    aborted.json     dqn, only when an update produced a non-finite loss:
                     ``error`` (the message), ``env_steps`` and ``vec_steps``
                     of the vec-step whose update failed, ``grad_steps``
                     completed. Distinguishes an aborted run from one that was
                     merely interrupted; ``snake_rl.analysis.aggregate``
                     reports the two separately.

A run is never overwritten: ``train`` raises ``FileExistsError`` if the
directory already holds any of these files (of either agent kind), finished
or not. To repeat an interrupted run under the same experiment name, move
its directory out of the experiment directory first (for example to
``runs/_interrupted/``) and run again.

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

DQN training loop
-----------------
The same loop with a replay buffer between acting and learning::

    env = VecSnakeEnv(cfg.env, n_envs, seed=env_seed_for(seed, n_envs)); obs = env.reset()
    agent = DQNAgent(env.obs_shape, env.n_actions, cfg.agent, seed=seed)
    buffer = ReplayBuffer(buffer_size, env.obs_shape)
    explore_rng = default_rng([seed, 1]); sample_rng = default_rng([seed, 2])
    G = grad_steps_per_vec_step or max(1, n_envs // 4)
    evaluate                                             # row 0: before any update
    for k = 0, 1, ...:                                   # env_steps_before = k * n_envs
        epsilon = linear_schedule(env_steps_before, eps_start, eps_end,
                                  eps_decay_fraction * total_env_steps)
        actions = agent.act_epsilon(obs, epsilon, explore_rng)
        next_obs, reward, terminated, truncated, info = env.step(actions)
        buffer.add(obs, actions, reward, info["final_obs"], terminated)
        obs = next_obs
        env_steps = (k + 1) * n_envs
        if env_steps >= learning_starts:
            G times: agent.update(buffer.sample(batch_size, sample_rng))
        if env_steps is a multiple of eval_every: evaluate

The two details of the tabular loop carry over to what is *stored*: the
buffer keeps the true successor ``info["final_obs"]`` and the ``terminated``
flag only, so a starvation-truncated transition is replayed as an ordinary
one and bootstraps (``DQNAgent.update``). A vec-step adds ``n_envs``
transitions and is followed by ``G`` gradient steps on uniform batches
(default: one gradient step per four environment steps); nothing is learned
before ``learning_starts`` environment steps are in the buffer. Each
evaluation also writes its checkpoint. ``DQNAgent.update`` raises
``FloatingPointError`` on a non-finite loss (Part B: automatic abort on NaN):
the metrics file is closed with the rows written so far, ``aborted.json`` is
written, neither ``model.pt`` nor ``final_eval.json`` is written (the run
stays recognisable as unfinished) and the error leaves ``train``; the command
line reports it with exit code 3. torch is imported by the DQN trainer only:
a tabular run and ``--help`` do not pay for it.

DQN without replay (``agent.replay: false``, the M6 ablation)
-------------------------------------------------------------
The same loop with no buffer: every vec-step after ``learning_starts`` is
followed by exactly ONE gradient step on that vec-step's own ``n_envs``
transitions, each used once (``batch_size``, ``buffer_size`` and
``grad_steps_per_vec_step`` are unused)::

    for k = 0, 1, ...:
        epsilon = ...; actions = agent.act_epsilon(obs, epsilon, explore_rng)
        next_obs, reward, terminated, truncated, info = env.step(actions)
        if (k + 1) * n_envs >= learning_starts:
            agent.update(Batch(obs, actions, reward, info["final_obs"], terminated))
        obs = next_obs

The transition semantics are the same (true successor, ``terminated`` only).
``default_rng([seed, 2])`` is still created, so the seed layout is the same,
but never drawn from. ``buffer_size`` is 0 in every metrics row. Against the
replay loop this changes two things at once, both of which the buffer
provides: the mixing of a batch across time and policies, and the number of
gradient steps per transition (64 in the shipped configurations, 1 here).

DQN with prioritised replay (``agent.per: true``, M7)
-----------------------------------------------------
The replay loop with a ``PrioritizedReplayBuffer`` (Schaul et al. 2016) in
place of the uniform one::

    buffer = PrioritizedReplayBuffer(buffer_size, obs_shape, per_alpha, per_eps)
    for k = 0, 1, ...:
        epsilon = ...
        beta = linear_schedule(k * n_envs, per_beta_start, 1.0, total_env_steps)
        ... act, step, buffer.add(...), obs = next_obs ...
        if env_steps >= learning_starts:
            G times:
                batch = buffer.sample(batch_size, sample_rng, beta)
                stats = agent.update(batch, batch.weights)
                buffer.update_priorities(batch.indices, stats["td_abs"])

``beta``, the importance-sampling exponent, is a function of the env steps
before the vec-step like epsilon (the same for the ``G`` gradient steps of
one vec-step) and reaches 1.0 at ``total_env_steps``. The sampling stream
``default_rng([seed, 2])`` is drawn once per sample as before. Everything
else - transitions stored, evaluations, files - is the replay loop's.

Value estimates (every DQN run, M7)
-----------------------------------
Each evaluation is ``evaluate_batched(agent, cfg.env, eval_episodes,
eval_seed, gamma=agent.gamma)``: besides the score it records, for the states
the greedy policy visits on the evaluation boards, the network's value
estimate ``max_a Q(s, a)`` and the discounted return the policy then collects
from the state (``snake_rl.evaluate``, module docstring). The four
``eval_value_* / eval_return_*`` columns are the measurement of van Hasselt
et al. (2016) for the Double DQN study; their difference is the
overestimation. Runs trained before M7 get the same columns from their
checkpoints with ``scripts/backfill_values.py`` (``values.csv``).

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

DQN adds two streams: replay sampling draws from ``default_rng([seed, 2])``
and the initial network weights are a function of the run seed
(``DQNAgent(..., seed=seed)``, which leaves the global torch generator
untouched). On the CPU the same config and seed give, on one machine and
torch build, identical ``model.pt`` tensors, ``final_eval.json`` episodes and
``metrics.csv`` (except the ``time_*`` and ``wall_time_s`` columns), in the
same process or a fresh one. Another CPU or BLAS code path changes the
rounding and, through it, the whole run: a re-run elsewhere gives other
weights and per-seed scores from the same distribution. Nothing is promised
bit for bit on cuda.

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

metrics.csv (DQN)
-----------------
``env_steps``, ``vec_steps``, ``epsilon``, ``eval_*``, ``train_episodes``,
``train_mean_score``, ``wall_time_s``: as above. ``eval_value_start``,
``eval_return_start``, ``eval_value_visited``, ``eval_return_visited``:
``EvalResult.value_summary()`` of that evaluation (the value estimates
against the realised returns, above). ``grad_steps``: gradient
steps taken so far (``agent.grad_steps``). ``train_truncated``: training
episodes since the previous row that ended by starvation truncation.
``loss``, ``mean_q``, ``mean_target``, ``grad_norm``: means of the
``DQNAgent.update`` statistics since the previous row (Huber loss,
importance-weighted with prioritised replay, mean ``Q(s, a)`` of the batch,
mean target, global gradient norm before clipping); empty while no update
has happened. ``buffer_size``: transitions in the replay buffer (0 without
replay). ``time_env_s``, ``time_act_s``, ``time_update_s``,
``time_eval_s``: cumulative seconds since the start spent in ``env.step``, in
``act_epsilon``, in sampling + gradient steps, and in evaluations; what is
left of ``wall_time_s`` is the buffer writes, checkpoints and bookkeeping.

Agent kinds: a trainer has the signature of ``_train_tabular_q``, owns its
metrics columns and checkpoint files and is registered in ``_TRAINERS``;
``train``, the run-directory bookkeeping, ``MetricsWriter`` and
``eval_columns`` are shared. ``_RUN_INFO`` holds what a kind adds to the
``run`` block of ``config.yaml``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import yaml

from snake_rl.agents.dqn import (  # numpy only; the torch part is imported by the DQN trainer
    Batch,
    PrioritizedReplayBuffer,
    ReplayBuffer,
)
from snake_rl.agents.schedules import linear_schedule
from snake_rl.agents.tabular_q import TabularQAgent
from snake_rl.config import ExperimentConfig, config_from_dict, config_to_dict, env_seed_for, load_config
from snake_rl.env.vec_env import VecSnakeEnv
from snake_rl.evaluate import EvalResult, evaluate_batched
from snake_rl.utils import git_hash

if TYPE_CHECKING:  # annotations only: torch must not be imported for a tabular run
    from snake_rl.agents.dqn.agent import DQNAgent

CONFIG_FILE = "config.yaml"
METRICS_FILE = "metrics.csv"
Q_TABLE_FILE = "q_table.npz"
MODEL_FILE = "model.pt"
CHECKPOINT_DIR = "checkpoints"
FINAL_EVAL_FILE = "final_eval.json"
ABORTED_FILE = "aborted.json"
#: Everything a run writes, of either agent kind (``q_table.npz``: tabular_q;
#: ``model.pt``, ``checkpoints/`` and ``aborted.json``: dqn). A directory
#: holding any of them is a run.
RUN_FILES = (CONFIG_FILE, METRICS_FILE, Q_TABLE_FILE, MODEL_FILE, CHECKPOINT_DIR, FINAL_EVAL_FILE, ABORTED_FILE)

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
#: The value estimates against the realised returns of an evaluation (M7;
#: ``EvalResult.value_summary()``), DQN only.
VALUE_COLUMNS = ("eval_value_start", "eval_return_start", "eval_value_visited", "eval_return_visited")
#: The statistics ``DQNAgent.update`` returns, averaged per metrics row.
UPDATE_STATS = ("loss", "mean_q", "mean_target", "grad_norm")
#: The phases of the DQN loop that are timed (``time_<phase>_s``).
PHASES = ("env", "act", "update", "eval")
#: Header of ``metrics.csv`` for DQN, in file order.
DQN_COLUMNS = (
    "env_steps",
    "vec_steps",
    "grad_steps",
    "epsilon",
    *EVAL_COLUMNS,
    *VALUE_COLUMNS,
    "train_episodes",
    "train_mean_score",
    "train_truncated",
    *UPDATE_STATS,
    "buffer_size",
    *(f"time_{phase}_s" for phase in PHASES),
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


def _train_dqn(cfg: ExperimentConfig, seed: int, run_dir: Path, quiet: bool) -> None:
    """DQN with a target network, with uniform replay (``agent.replay``) or
    without it (the two DQN loops of the module docstring)."""
    import torch  # here, not at module level: tabular runs and --help never import torch

    from snake_rl.agents.dqn.agent import DQNAgent

    ac, tc = cfg.agent, cfg.train
    if ac.torch_threads is not None:
        torch.set_num_threads(ac.torch_threads)
    env = VecSnakeEnv(cfg.env, tc.n_envs, seed=env_seed_for(seed, tc.n_envs))
    agent = DQNAgent(env.obs_shape, env.n_actions, ac, seed=seed)  # initial weights: a function of the run seed
    buffer: ReplayBuffer | None = None
    if ac.replay:
        buffer = (
            PrioritizedReplayBuffer(ac.buffer_size, env.obs_shape, ac.per_alpha, ac.per_eps)
            if ac.per
            else ReplayBuffer(ac.buffer_size, env.obs_shape)
        )
    explore_rng = np.random.default_rng([seed, 1])  # exploration stream, as in the tabular loop
    sample_rng = np.random.default_rng([seed, 2])  # replay sampling stream (created, never drawn without replay)
    grad_steps_per_vec_step = (ac.grad_steps_per_vec_step or max(1, tc.n_envs // 4)) if ac.replay else 1
    decay_steps = ac.eps_decay_fraction * tc.total_env_steps

    def learn_from_buffer(env_steps_before: int) -> None:
        """The gradient steps of one vec-step with replay: uniform, or prioritised (module docstring)."""
        if isinstance(buffer, PrioritizedReplayBuffer):
            beta = linear_schedule(env_steps_before, ac.per_beta_start, 1.0, tc.total_env_steps)
            for _ in range(grad_steps_per_vec_step):
                batch = buffer.sample(ac.batch_size, sample_rng, beta)
                stats = agent.update(batch, batch.weights)
                buffer.update_priorities(batch.indices, stats["td_abs"])
                window.add_update(stats)
        else:
            for _ in range(grad_steps_per_vec_step):
                window.add_update(agent.update(buffer.sample(ac.batch_size, sample_rng)))
    # Queued GPU work is waited for at row boundaries only, never per update.
    timer = PhaseTimer(synchronize=torch.cuda.synchronize if agent.device.type == "cuda" else None)

    env_steps = 0
    try:
        with MetricsWriter(run_dir / METRICS_FILE, DQN_COLUMNS) as metrics:
            evaluate_now = partial(
                _evaluate_dqn, agent, buffer, cfg, metrics, run_dir, timer, label=f"{cfg.exp}/{seed}", quiet=quiet
            )
            obs = env.reset()
            last_eval = evaluate_now(0, ac.eps_start, DQNWindow())  # before any update
            window = DQNWindow()
            for k in range(tc.total_env_steps // tc.n_envs):
                epsilon = linear_schedule(k * tc.n_envs, ac.eps_start, ac.eps_end, decay_steps)
                with timer.phase("act"):
                    actions = agent.act_epsilon(obs, epsilon, explore_rng)
                with timer.phase("env"):
                    next_obs, reward, terminated, truncated, info = env.step(actions)
                # True successor (not the post-reset observation) and the terminated
                # flag only: a starvation truncation is learned from as an ordinary
                # transition, so its target bootstraps.
                if buffer is not None:
                    buffer.add(obs, actions, reward, info["final_obs"], terminated)
                window.add_step(info, truncated)

                env_steps = (k + 1) * tc.n_envs
                if env_steps >= ac.learning_starts:
                    with timer.phase("update"):
                        if buffer is not None:
                            learn_from_buffer(k * tc.n_envs)
                        else:  # no replay: this vec-step's own transitions, once, one gradient step
                            own = Batch(obs=obs, actions=actions, rewards=reward, next_obs=info["final_obs"],
                                        terminated=terminated)
                            window.add_update(agent.update(own))
                obs = next_obs
                if env_steps % tc.eval_every == 0:
                    last_eval = evaluate_now(env_steps, epsilon, window)
                    window = DQNWindow()
    except FloatingPointError as err:
        # Mark the run as aborted (not merely interrupted) for the analysis;
        # the metrics file is already closed with the rows written so far.
        _write_json(run_dir / ABORTED_FILE, {
            "error": str(err),
            "env_steps": env_steps,
            "vec_steps": env_steps // tc.n_envs,
            "grad_steps": agent.grad_steps,
        })
        raise

    agent.save(run_dir / MODEL_FILE)  # with the optimiser state
    _save_final_eval(last_eval, run_dir)


@dataclass
class DQNWindow:
    """Training statistics of the DQN loop accumulated between two metrics rows."""

    episodes: int = 0
    score_sum: int = 0
    truncated: int = 0
    updates: int = 0
    stat_sums: dict[str, float] = field(default_factory=lambda: dict.fromkeys(UPDATE_STATS, 0.0))

    def add_step(self, info: Mapping[str, np.ndarray], truncated: np.ndarray) -> None:
        """One vec-step: the training episodes it finished, and how many of them by starvation."""
        done = info["done"]
        self.episodes += int(done.sum())
        self.score_sum += int(info["score"][done].sum())
        self.truncated += int(truncated.sum())

    def add_update(self, stats: Mapping[str, float]) -> None:
        """One gradient step: the statistics ``DQNAgent.update`` returned."""
        self.updates += 1
        for name in UPDATE_STATS:
            self.stat_sums[name] += stats[name]

    @property
    def mean_score(self) -> float | None:
        return self.score_sum / self.episodes if self.episodes else None

    def mean(self, name: str) -> float | None:
        """Mean of one update statistic over the window; ``None`` if there was no update."""
        return self.stat_sums[name] / self.updates if self.updates else None


class PhaseTimer:
    """Cumulative wall-clock seconds per phase of the DQN loop (the ``time_*``
    columns) and since its creation (``wall_time_s``).

    ``synchronize`` (``torch.cuda.synchronize`` on cuda, ``None`` on the CPU)
    waits for queued GPU work. It is called by :meth:`flush` only, at a
    metrics row, so the per-update cost is not inflated by a wait after every
    step; the time it takes is booked on the phase that queued the work.
    """

    def __init__(self, synchronize: Callable[[], None] | None = None) -> None:
        self.seconds = dict.fromkeys(PHASES, 0.0)
        self._synchronize = synchronize
        self._started = time.perf_counter()

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Add the duration of the ``with`` block to phase ``name``."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] += time.perf_counter() - t0

    def flush(self, name: str) -> None:
        """Wait for queued GPU work (if any) and book the wait on phase ``name``."""
        if self._synchronize is not None:
            with self.phase(name):
                self._synchronize()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._started


def _evaluate_dqn(
    agent: "DQNAgent",
    buffer: ReplayBuffer | None,
    cfg: ExperimentConfig,
    metrics: "MetricsWriter",
    run_dir: Path,
    timer: PhaseTimer,
    env_steps: int,
    epsilon: float,
    window: DQNWindow,
    *,
    label: str,
    quiet: bool,
) -> EvalResult:
    """Greedy evaluation on the fixed evaluation boards; appends one metrics
    row and, with ``keep_checkpoints``, saves the evaluated weights."""
    ac, tc = cfg.agent, cfg.train
    timer.flush("update")  # the gradient steps since the previous row are complete before the clock is read
    with timer.phase("eval"):  # ends with the action values on the host, so nothing is left queued
        result = evaluate_batched(agent, cfg.env, tc.eval_episodes, tc.eval_seed, gamma=ac.gamma)
    if ac.keep_checkpoints:
        agent.save(run_dir / CHECKPOINT_DIR / f"step_{env_steps:08d}.pt", include_optimizer=False)
    row = {
        "env_steps": env_steps,
        "vec_steps": env_steps // tc.n_envs,
        "grad_steps": agent.grad_steps,
        "epsilon": epsilon,
        **eval_columns(result),
        **value_columns(result),
        "train_episodes": window.episodes,
        "train_mean_score": window.mean_score,
        "train_truncated": window.truncated,
        **{name: window.mean(name) for name in UPDATE_STATS},
        "buffer_size": len(buffer) if buffer is not None else 0,
        **{f"time_{phase}_s": round(timer.seconds[phase], 3) for phase in PHASES},
        "wall_time_s": round(timer.elapsed, 3),
    }
    metrics.write(row)
    if not quiet:
        print(_dqn_progress_line(label, tc.total_env_steps, row), flush=True)
    return result


def _dqn_run_info(cfg: ExperimentConfig) -> dict[str, Any]:
    """What a DQN run adds to the ``run`` block of ``config.yaml``: the torch
    version and the hardware the numbers were produced on."""
    import torch

    from snake_rl.agents.dqn.agent import resolve_device

    device = resolve_device(cfg.agent.device)  # the device DQNAgent will resolve to
    threads = cfg.agent.torch_threads if cfg.agent.torch_threads is not None else torch.get_num_threads()
    return {
        "torch": str(torch.__version__),  # a str subclass that YAML cannot represent
        "device": device.type,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_threads": threads,
    }


#: ``agent.kind`` -> trainer ``(cfg, seed, run_dir, quiet)``. A trainer writes
#: its metrics and checkpoints into ``run_dir`` and ``final_eval.json`` last.
_TRAINERS: dict[str, Callable[[ExperimentConfig, int, Path, bool], None]] = {
    "tabular_q": _train_tabular_q,
    "dqn": _train_dqn,
}
#: ``agent.kind`` -> extra keys of the ``run`` block of ``config.yaml``
#: (kinds without an entry add nothing).
_RUN_INFO: dict[str, Callable[[ExperimentConfig], dict[str, Any]]] = {
    "dqn": _dqn_run_info,
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
    an agent kind without a trainer; ``FileExistsError`` if the run exists;
    ``FloatingPointError`` if a DQN update produced a non-finite loss (the
    run directory is left without ``final_eval.json``).
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
    except FloatingPointError as err:  # DQN: non-finite loss, the run was aborted
        print(f"error: {err}", file=sys.stderr)
        return 3
    return 0


# ------------------------------------------------------ run-directory files
def _start_run(cfg: ExperimentConfig, seed: int, env_seed: int, run_dir: Path, argv: Sequence[str] | None) -> None:
    """Claim the run directory by writing ``config.yaml``; refuse an existing run."""
    existing = [name for name in RUN_FILES if (run_dir / name).exists()]
    if existing:
        raise FileExistsError(
            f"{run_dir} already holds a run ({', '.join(existing)}); runs are never overwritten - "
            f"use another --exp or --seed, or, to repeat an interrupted run under the same name, move that "
            f"directory out of its experiment directory first (for example to runs/_interrupted/)"
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
        **_RUN_INFO.get(cfg.agent_kind, lambda _: {})(cfg),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    # Mode "x" fails if the file exists: two processes cannot both claim the run.
    with open(run_dir / CONFIG_FILE, "x", encoding="utf-8") as f:
        f.write("# Written by snake_rl.train before training started. To repeat this run:\n")
        f.write("#   python -m snake_rl.train --config <this file> --seed <run.seed> --exp <new name>\n")
        yaml.safe_dump(record, f, sort_keys=False)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2)
        f.write("\n")


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


def value_columns(result: EvalResult) -> dict[str, float]:
    """The ``eval_value_* / eval_return_*`` columns (``EvalResult.value_summary()``)."""
    values = result.value_summary()
    if values is None:
        raise ValueError("the evaluation carries no value estimates: evaluate with gamma")
    return {f"eval_{name}": values[name] for name in ("value_start", "return_start", "value_visited", "return_visited")}


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


def _dqn_progress_line(label: str, total_env_steps: int, row: Mapping[str, Any]) -> str:
    def optional(value: float | None, spec: str) -> str:
        return "-" if value is None else format(value, spec)

    width = len(f"{total_env_steps:,}")
    return (
        f"[{label}] {row['env_steps']:>{width},}/{total_env_steps:,} steps  eps {row['epsilon']:.3f}  "
        f"eval {row['eval_mean_score']:.2f} +- {row['eval_ci95']:.2f}  "
        f"wall {row['eval_frac_wall']:.2f} self {row['eval_frac_self']:.2f} "
        f"starve {row['eval_frac_starve']:.2f} win {row['eval_frac_win']:.2f}  "
        f"train {optional(row['train_mean_score'], '.2f')} ({row['train_episodes']} ep)  "
        f"loss {optional(row['loss'], '.4f')}  q {optional(row['mean_q'], '.3f')}  "
        f"{row['grad_steps']:,} updates  {row['wall_time_s']:.1f}s"
    )


if __name__ == "__main__":
    raise SystemExit(main())

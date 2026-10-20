"""The DQN training path (PLAN.md milestones M5, M6 and M7): the ``dqn`` agent
kind of ``snake_rl.config`` (M6: observation B accepted, ``replay`` and
``conv_channels``; M7: ``dueling``, ``per``, ``per_alpha``, ``per_beta_start``,
``per_eps``), the trainer ``_train_dqn`` of ``snake_rl.train`` (M6: the
no-replay loop, ``aborted.json``; M7: the four value columns of the
overestimation measurement, van Hasselt et al. 2016), the six shipped DQN
configuration files and ``scripts/backfill_values.py`` (M7). The PER loop of
M7 has its own file, ``tests/test_m7_train_per.py``.

Every run of this file goes to a pytest temporary directory, on the CPU with
one torch thread and tiny networks and budgets. The subprocess jobs (a fresh
training process, ``scripts/run_matrix.py`` on the smoke config and the
shipped observation-B smoke config) are started when the module starts, so
they run while the in-process tests do.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

import snake_rl.config as config_module
import snake_rl.evaluate as evaluate_module
import snake_rl.train as train_module
from snake_rl.agents.baselines import make_policy
from snake_rl.agents.dqn import DQNConfig, ReplayBuffer
from snake_rl.agents.dqn.agent import DQNAgent
from snake_rl.agents.dqn.networks import CNN
from snake_rl.agents.dqn.replay import Batch
from snake_rl.agents.schedules import linear_schedule
from snake_rl.agents.tabular_q import TabularQConfig
from snake_rl.config import ExperimentConfig, TrainConfig, config_from_dict, config_to_dict, load_config
from snake_rl.env import EnvConfig, ObsConfig, RewardConfig, VecSnakeEnv
from snake_rl.evaluate import EvalResult, evaluate, evaluate_batched
from snake_rl.train import main, train
from snake_rl.utils import git_hash

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"
SMOKE = CONFIGS / "smoke_8x8.yaml"
SMOKE_B = CONFIGS / "smoke_B_8x8.yaml"
SMOKE_TABULAR = CONFIGS / "smoke_tabular.yaml"
ABORTED_FILE = "aborted.json"
BATCH_FIELDS = ("obs", "actions", "rewards", "next_obs", "terminated")

#: M7 (D6): the four value columns sit right after ``eval_frac_win``.
HEADER = (
    "env_steps,vec_steps,grad_steps,epsilon,eval_mean_score,eval_ci95,eval_median_score,eval_max_score,"
    "eval_mean_steps,eval_frac_wall,eval_frac_self,eval_frac_starve,eval_frac_win,eval_value_start,eval_return_start,"
    "eval_value_visited,eval_return_visited,train_episodes,train_mean_score,"
    "train_truncated,loss,mean_q,mean_target,grad_norm,buffer_size,time_env_s,time_act_s,time_update_s,time_eval_s,"
    "wall_time_s"
)
VALUE_COLUMNS = ("eval_value_start", "eval_return_start", "eval_value_visited", "eval_return_visited")
#: The four ``EpisodeRecord`` fields behind the value columns (M7, D5), and the five fields of M5.
VALUE_FIELDS = ("value_start", "return_start", "value_visited", "return_visited")
OLD_RECORD_FIELDS = ("seed", "score", "steps", "length", "cause")
#: Unchanged by M7 (D6): the header of a tabular ``metrics.csv`` (``tests/test_train.py``).
TABULAR_HEADER = (
    "env_steps,vec_steps,epsilon,eval_mean_score,eval_ci95,eval_median_score,eval_max_score,eval_mean_steps,"
    "eval_frac_wall,eval_frac_self,eval_frac_starve,eval_frac_win,train_episodes,train_mean_score,mean_abs_td,"
    "cells_visited,wall_time_s"
)
#: M7 (D7): the columns of ``values.csv`` written by ``scripts/backfill_values.py``.
VALUES_HEADER = "env_steps,eval_mean_score,eval_value_start,eval_return_start,eval_value_visited,eval_return_visited"
BACKFILL = REPO_ROOT / "scripts" / "backfill_values.py"
#: M7 (D0 / D10): the five new agent keys, in dataclass order, with their defaults.
M7_KEYS = ("dueling", "per", "per_alpha", "per_beta_start", "per_eps")
M7_DEFAULTS = {"dueling": False, "per": False, "per_alpha": 0.6, "per_beta_start": 0.4, "per_eps": 1e-6}
#: The four shipped batch files that D10 changes (the smoke files are unchanged).
BATCH_FILES = ("dqn_A_8x8.yaml", "dqn_A_10x10.yaml", "dqn_B_8x8.yaml", "dqn_B_10x10.yaml")
INT_COLUMNS = ("env_steps", "vec_steps", "grad_steps", "eval_max_score", "train_episodes", "train_truncated",
               "buffer_size")
TIME_COLUMNS = ("time_env_s", "time_act_s", "time_update_s", "time_eval_s", "wall_time_s")
UPDATE_COLUMNS = ("loss", "mean_q", "mean_target", "grad_norm")
RUN_ENTRIES = ["checkpoints", "config.yaml", "final_eval.json", "metrics.csv", "model.pt"]
FAKE_GIT = "testhash"

#: The smoke config shrunk to a fraction of a second: 300 vec-steps of 8 boards,
#: 4 evaluations of 6 episodes, a one-layer network. Learning starts after the
#: second evaluation row (1000 > 800) and the buffer (1000) is smaller than the
#: budget, so both "no update yet" and "buffer full" occur in the metrics.
TINY = ("exp=tiny_dqn", "env.starve_limit=20", "train.n_envs=8", "train.total_env_steps=2400",
        "train.eval_every=800", "train.eval_episodes=6", "agent.hidden=[16]", "agent.batch_size=16",
        "agent.buffer_size=1000", "agent.learning_starts=1000", "agent.target_update_every=50")
TINY_SEED = 3
#: A second shape of run for the loop tests: a 6x6 board with a starvation
#: limit of 8 (truncations are frequent), an explicit number of gradient steps
#: per vec-step, a buffer that wraps, the Double DQN target, a target sync
#: every 7 gradient steps and an epsilon schedule that is still decaying at
#: the first evaluations.
VARIANT = ("exp=variant_dqn", "env.W=6", "env.starve_limit=8", "train.n_envs=6", "train.total_env_steps=1800",
           "train.eval_every=600", "train.eval_episodes=8", "train.eval_seed=777", "agent.hidden=[12, 12]",
           "agent.batch_size=8", "agent.buffer_size=500", "agent.learning_starts=300",
           "agent.grad_steps_per_vec_step=3", "agent.target_update_every=7", "agent.double=true",
           "agent.gamma=0.8", "agent.lr=1.0e-3", "agent.eps_end=0.1", "agent.eps_decay_fraction=0.5")
VARIANT_SEED = 2
#: M6: the VARIANT run without a replay buffer (one gradient step per vec-step
#: on that vec-step's own six transitions, from vec-step 50 on).
NOREPLAY = (*VARIANT, "exp=noreplay_dqn", "agent.replay=false", "agent.grad_steps_per_vec_step=1")
NOREPLAY_SEED = 4
#: M6: the observation-B smoke config shrunk like TINY: a CNN of two 4-channel
#: layers and a 16-unit hidden layer on the 8-plane 8x8 grid, 300 vec-steps of
#: 8 boards, learning from 1000 env steps on.
TINY_B = ("exp=tiny_dqn_B", "env.starve_limit=20", "train.n_envs=8", "train.total_env_steps=2400",
          "train.eval_every=800", "train.eval_episodes=6", "agent.conv_channels=[4, 4]", "agent.hidden=[16]",
          "agent.batch_size=16", "agent.buffer_size=1000", "agent.learning_starts=1000",
          "agent.target_update_every=50")


# ----------------------------------------------------------------- fixtures
@pytest.fixture(scope="module", autouse=True)
def background(tmp_path_factory):
    """The subprocess jobs of this file, started now and collected by the tests
    that need them (``background(name)`` waits for the job and returns its
    result). Each costs a torch import, so they overlap with the in-process tests."""
    root = tmp_path_factory.mktemp("subprocess_runs")
    arguments = {
        # One fresh training process of the tiny configuration (determinism across processes).
        "fresh": ["-m", "snake_rl.train", "--config", str(SMOKE), "--seed", str(TINY_SEED), "--runs-dir",
                  str(root / "fresh"), "--quiet", "--set", *TINY],
        # The batch runner on the shipped smoke config, unchanged: two seeds, two processes.
        "matrix": [str(REPO_ROOT / "scripts" / "run_matrix.py"), "--config", str(SMOKE), "--seeds", "0", "1",
                   "--runs-dir", str(root / "matrix"), "--jobs", "2"],
        # M6: the shipped observation-B smoke config, as shipped, in a fresh process.
        "smoke_b": ["-m", "snake_rl.train", "--config", str(SMOKE_B), "--seed", "0", "--runs-dir",
                    str(root / "smoke_b"), "--quiet"],
    }
    processes = {
        name: subprocess.Popen([sys.executable, *args], cwd=REPO_ROOT, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
        for name, args in arguments.items()
    }
    finished: dict[str, SimpleNamespace] = {}

    def finish(name: str) -> SimpleNamespace:
        if name not in finished:
            out, err = processes[name].communicate(timeout=300)
            finished[name] = SimpleNamespace(returncode=processes[name].returncode, stdout=out, stderr=err,
                                             args=arguments[name], runs=root / name)
        return finished[name]

    yield finish
    for name in processes:  # never leave a process (or its children) behind
        finish(name)


@pytest.fixture(scope="module", autouse=True)
def fast_git_hash():
    """``git_hash()`` starts git processes and runs once per evaluation; a
    constant keeps the file fast. The fresh-process test checks the real hash."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train_module, "git_hash", lambda: FAKE_GIT)
        patch.setattr(evaluate_module, "git_hash", lambda: FAKE_GIT)
        yield


@pytest.fixture(scope="module", autouse=True)
def restore_torch_threads():
    """The trainer applies ``agent.torch_threads`` with ``torch.set_num_threads``
    (process-wide); other test files get the previous setting back."""
    before = torch.get_num_threads()
    yield
    torch.set_num_threads(before)


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path, monkeypatch):
    """A relative ``runs`` directory can only ever land in ``tmp_path``."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    runs = tmp_path_factory.mktemp("tiny_runs")
    cfg = load_config(SMOKE, TINY)
    return SimpleNamespace(cfg=cfg, seed=TINY_SEED, runs=runs, dir=train(cfg, TINY_SEED, runs, quiet=True))


@pytest.fixture(scope="module")
def variant_run(tmp_path_factory):
    """The VARIANT run, with a spy on the training environment and on
    ``ReplayBuffer.add`` (neither changes what the run does)."""
    runs = tmp_path_factory.mktemp("variant_runs")
    cfg = load_config(SMOKE, VARIANT)
    envs, observed, steps, adds = [], [], [], []

    class SpyEnv(VecSnakeEnv):
        def __init__(self, env_cfg, n_envs, seed):
            super().__init__(env_cfg, n_envs, seed=seed)
            envs.append((env_cfg, n_envs, seed))

        def reset(self, seed=None):
            obs = super().reset(seed)
            observed.append(obs.copy())
            return obs

        def step(self, actions):
            obs, reward, terminated, truncated, info = super().step(actions)
            steps.append(SimpleNamespace(actions=np.array(actions), reward=reward.copy(), terminated=terminated.copy(),
                                         truncated=truncated.copy(), final_obs=info["final_obs"].copy()))
            observed.append(obs.copy())
            return obs, reward, terminated, truncated, info

    real_add = ReplayBuffer.add

    def spy_add(self, obs, actions, rewards, next_obs, terminated):
        adds.append(SimpleNamespace(obs=np.array(obs), actions=np.array(actions), rewards=np.array(rewards),
                                    next_obs=np.array(next_obs), terminated=np.array(terminated)))
        return real_add(self, obs, actions, rewards, next_obs, terminated)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train_module, "VecSnakeEnv", SpyEnv)  # the training env only; evaluations build their own
        patch.setattr(ReplayBuffer, "add", spy_add)
        run_dir = train(cfg, VARIANT_SEED, runs, quiet=True)
    return SimpleNamespace(cfg=cfg, seed=VARIANT_SEED, runs=runs, dir=run_dir, envs=envs, observed=observed,
                           steps=steps, adds=adds)


@pytest.fixture(scope="module")
def noreplay_run(tmp_path_factory):
    """M6: the NOREPLAY run, with a spy on the training environment, on every
    ``DQNAgent.update`` call (the batch it received and ``grad_steps`` before
    it) and on ``ReplayBuffer`` construction (none changes what the run does)."""
    runs = tmp_path_factory.mktemp("noreplay_runs")
    cfg = load_config(SMOKE, NOREPLAY)
    envs, observed, steps, updates, buffers = [], [], [], [], []

    class SpyEnv(VecSnakeEnv):
        def __init__(self, env_cfg, n_envs, seed):
            super().__init__(env_cfg, n_envs, seed=seed)
            envs.append((env_cfg, n_envs, seed))

        def reset(self, seed=None):
            obs = super().reset(seed)
            observed.append(obs.copy())
            return obs

        def step(self, actions):
            obs, reward, terminated, truncated, info = super().step(actions)
            steps.append(SimpleNamespace(actions=np.array(actions), reward=reward.copy(), terminated=terminated.copy(),
                                         truncated=truncated.copy(), final_obs=info["final_obs"].copy()))
            observed.append(obs.copy())
            return obs, reward, terminated, truncated, info

    real_update = DQNAgent.update

    def spy_update(self, batch):
        assert isinstance(batch, Batch)
        updates.append(SimpleNamespace(grad_steps=self.grad_steps,
                                       batch=Batch(*(np.array(getattr(batch, name)) for name in BATCH_FIELDS))))
        return real_update(self, batch)

    real_init = ReplayBuffer.__init__

    def spy_init(self, *args, **kwargs):
        buffers.append((args, kwargs))
        return real_init(self, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train_module, "VecSnakeEnv", SpyEnv)
        patch.setattr(DQNAgent, "update", spy_update)
        patch.setattr(ReplayBuffer, "__init__", spy_init)
        run_dir = train(cfg, NOREPLAY_SEED, runs, quiet=True)
    return SimpleNamespace(cfg=cfg, seed=NOREPLAY_SEED, runs=runs, dir=run_dir, envs=envs, observed=observed,
                           steps=steps, updates=updates, buffers=buffers)


@pytest.fixture(scope="module")
def tiny_b_run(tmp_path_factory):
    """M6: the TINY_B run (the CNN on the 8-plane 8x8 grid, with replay)."""
    runs = tmp_path_factory.mktemp("tiny_b_runs")
    cfg = load_config(SMOKE_B, TINY_B)
    return SimpleNamespace(cfg=cfg, seed=TINY_SEED, runs=runs, dir=train(cfg, TINY_SEED, runs, quiet=True))


# ------------------------------------------------------------------ helpers
def dqn_dict() -> dict:
    """A complete, valid DQN configuration written out by hand (YAML layout)."""
    return {
        "exp": "unit_dqn",
        "env": {
            "W": 8,
            "action_mode": "relative",
            "starve_limit": None,
            "obs": {"kind": "A", "age_plane": True, "heading_planes": True},
            "reward": {"variant": "R0", "food": 1.0, "death": -1.0, "step_penalty": 0.01, "gamma": 0.99,
                       "closer": 0.01, "farther": 0.0},
        },
        "agent": {"kind": "dqn", "gamma": 0.9, "lr": 2.5e-4, "batch_size": 16, "buffer_size": 1000, "replay": True,
                  "learning_starts": 400, "grad_steps_per_vec_step": None, "target_update_every": 50,
                  "grad_clip": 10.0, "eps_start": 1.0, "eps_end": 0.05, "eps_decay_fraction": 0.2,
                  "hidden": [16, 16], "conv_channels": [32, 64, 64], "double": False, "dueling": False, "per": False,
                  "per_alpha": 0.6, "per_beta_start": 0.4, "per_eps": 1.0e-6, "keep_checkpoints": True,
                  "device": "cpu", "torch_threads": 1},
        "train": {"n_envs": 8, "total_env_steps": 2400, "eval_every": 800, "eval_episodes": 6, "eval_seed": 10000},
    }


def edited(changes: dict) -> dict:
    """``dqn_dict()`` with ``{dotted.key: value}`` applied."""
    d = dqn_dict()
    for dotted, value in changes.items():
        *parents, leaf = dotted.split(".")
        node = d
        for name in parents:
            node = node[name]
        node[leaf] = value
    return d


def write_yaml(path: Path, data) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def config_error(path: Path, overrides=()) -> str:
    """Message of the ``ValueError`` of ``load_config``, without the path prefix."""
    with pytest.raises(ValueError) as excinfo:
        load_config(path, overrides)
    message = str(excinfo.value)
    assert message.startswith(f"{path}: "), message
    return message[len(f"{path}: "):]


def names_key(message: str, key: str) -> bool:
    """``key`` occurs in ``message`` as a whole dotted name."""
    return re.search(rf"(?<![\w.]){re.escape(key)}(?![\w.])", message) is not None


def read_run_config(run_dir: Path) -> dict:
    return yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))


def read_metrics(run_dir: Path) -> list[dict]:
    """``metrics.csv`` as typed rows: ``None`` for an empty cell, ``int`` for
    the integer columns (which must be written without a decimal point)."""
    with open(run_dir / "metrics.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    typed = []
    for row in rows:
        out = {}
        for name, text in row.items():
            if text == "":
                out[name] = None
            elif name in INT_COLUMNS:
                assert re.fullmatch(r"\d+", text), (name, text)
                out[name] = int(text)
            else:
                out[name] = float(text)
        typed.append(out)
    return typed


def metrics_without_times(run_dir: Path) -> list[dict]:
    """The rows of ``metrics.csv`` as written (text), without the time columns."""
    with open(run_dir / "metrics.csv", newline="", encoding="utf-8") as f:
        return [{name: text for name, text in row.items() if name not in TIME_COLUMNS} for row in csv.DictReader(f)]


def flat(path: Path) -> dict:
    """Everything in a checkpoint file, flattened: tensors as (shape, dtype, bytes)."""
    out: dict = {}

    def walk(prefix: str, node) -> None:
        if isinstance(node, torch.Tensor):
            out[prefix] = (tuple(node.shape), str(node.dtype), node.numpy().tobytes())
        elif isinstance(node, dict):
            for key, value in node.items():
                walk(f"{prefix}/{key}", value)
        elif isinstance(node, (list, tuple)):
            for i, value in enumerate(node):
                walk(f"{prefix}/{i}", value)
        else:
            out[prefix] = node

    walk("", torch.load(path, map_location="cpu", weights_only=True))  # plain tensors and primitives only
    return out


def final_episodes(run_dir: Path) -> list:
    return EvalResult.load(run_dir / "final_eval.json").episodes


def old_fields(records) -> list[tuple]:
    """The five M5 fields of episode records (an evaluation without ``gamma``
    leaves the four M7 value fields ``None``, a DQN run's file fills them)."""
    return [tuple(getattr(e, name) for name in OLD_RECORD_FIELDS) for e in records]


def checkpoint_names(run_dir: Path) -> list[str]:
    return sorted(p.name for p in (run_dir / "checkpoints").iterdir())


def assert_same_run(a: Path, b: Path) -> None:
    """Identical model and checkpoint tensors, final-evaluation episodes and
    metrics (except the time columns)."""
    assert flat(a / "model.pt") == flat(b / "model.pt")
    assert checkpoint_names(a) == checkpoint_names(b)
    for name in checkpoint_names(a):
        assert flat(a / "checkpoints" / name) == flat(b / "checkpoints" / name)
    assert final_episodes(a) == final_episodes(b)
    assert metrics_without_times(a) == metrics_without_times(b)


def tree(directory: Path) -> dict:
    """Relative path -> file bytes (``None`` for a directory) of everything below ``directory``."""
    return {
        p.relative_to(directory).as_posix(): p.read_bytes() if p.is_file() else None
        for p in sorted(directory.rglob("*"))
    }


def expected_grad_steps(cfg: ExperimentConfig, vec_steps: int) -> int:
    """G gradient steps after every vec-step that ends at or after ``learning_starts``
    (with replay); exactly one per such vec-step without replay (M6, D4)."""
    ac, n = cfg.agent, cfg.train.n_envs
    per_vec_step = (ac.grad_steps_per_vec_step or max(1, n // 4)) if ac.replay else 1
    return per_vec_step * sum(1 for v in range(1, vec_steps + 1) if v * n >= ac.learning_starts)


# =================================================================== config
OBS_B8 = ObsConfig(kind="B", age_plane=True, heading_planes=True)


def expected_config(
    exp: str, W: int, agent: DQNConfig, train_cfg: TrainConfig, obs: ObsConfig = ObsConfig(kind="A")
) -> ExperimentConfig:
    return ExperimentConfig(
        exp=exp,
        env=EnvConfig(W=W, action_mode="relative", starve_limit=None, obs=obs, reward=RewardConfig(variant="R0")),
        agent_kind="dqn",
        agent=agent,
        train=train_cfg,
    )


SHIPPED = {
    # gamma 0.9: the value of the tabular reference (M4 sweep). cpu with 2 threads: measured about three
    # times faster per gradient step than cuda on this MLP, and bit-for-bit reproducible. eval_every
    # 12,800: the greedy score plateaus within the first 50,000 steps. The rest is the DQNConfig default.
    # M7 (D10): the four batch files write the five new fields out with their defaults (M7_DEFAULTS) and
    # load to the same configurations as before.
    "dqn_A_10x10.yaml": expected_config("dqn_A_W10", 10, DQNConfig(gamma=0.9, device="cpu", torch_threads=2,
                                                                   **M7_DEFAULTS),
                                        TrainConfig(128, 1_024_000, 12_800, 100, 10_000)),
    "dqn_A_8x8.yaml": expected_config("dqn_A_W8", 8, DQNConfig(gamma=0.9, device="cpu", torch_threads=2,
                                                               **M7_DEFAULTS),
                                      TrainConfig(128, 512_000, 12_800, 100, 10_000)),
    "smoke_8x8.yaml": expected_config(
        "smoke_dqn_W8", 8,
        DQNConfig(gamma=0.9, batch_size=64, buffer_size=10_000, learning_starts=1024, target_update_every=250,
                  hidden=(64, 64), device="cpu", torch_threads=1),
        TrainConfig(16, 16_000, 4_000, 20, 10_000),
    ),
    # M6 (D6): the CNN files on the 8-plane observation B. The two experiment files have the board, reward,
    # evaluation protocol and budget of their A counterparts; the smoke file a small CNN (16 and 32
    # channels, one hidden layer of 128) at three times the budget of smoke_8x8.yaml (the M6 measurement:
    # an 8/8-channel, 32-unit CNN does not leave the random level within 96,000 steps; this one scores
    # 9.7 / 13.7 on seeds 0 / 1 at 48,000 steps, about 45 s on one CPU thread).
    "smoke_B_8x8.yaml": expected_config(
        "smoke_dqn_B8_W8", 8,
        DQNConfig(gamma=0.9, batch_size=64, buffer_size=10_000, learning_starts=1024, target_update_every=250,
                  hidden=(128,), conv_channels=(16, 32), device="cpu", torch_threads=1),
        TrainConfig(16, 48_000, 12_000, 20, 10_000), obs=OBS_B8,
    ),
    "dqn_B_8x8.yaml": expected_config(
        "dqn_B8_W8", 8, DQNConfig(gamma=0.9, hidden=(512,), conv_channels=(32, 64, 64), device="cpu", **M7_DEFAULTS),
        TrainConfig(128, 512_000, 12_800, 100, 10_000), obs=OBS_B8,
    ),
    "dqn_B_10x10.yaml": expected_config(
        "dqn_B8_W10", 10, DQNConfig(gamma=0.9, hidden=(512,), conv_channels=(32, 64, 64), device="cpu", **M7_DEFAULTS),
        TrainConfig(128, 1_024_000, 12_800, 100, 10_000), obs=OBS_B8,
    ),
}
#: Agent fields of a shipped file that D6 leaves open (decided by the D7 benchmark): compared loosely.
FREE_FIELDS = {"dqn_B_8x8.yaml": ("device", "torch_threads"), "dqn_B_10x10.yaml": ("device", "torch_threads")}


def test_dqn_config_defaults():
    assert DQNConfig() == DQNConfig(
        gamma=0.99, lr=2.5e-4, batch_size=256, buffer_size=100_000, replay=True, learning_starts=10_000,
        grad_steps_per_vec_step=None, target_update_every=1_000, grad_clip=10.0, eps_start=1.0, eps_end=0.05,
        eps_decay_fraction=0.2, hidden=(256, 256), conv_channels=(32, 64, 64), double=False, dueling=False, per=False,
        per_alpha=0.6, per_beta_start=0.4, per_eps=1e-6, keep_checkpoints=True, device="auto", torch_threads=None,
    )
    assert [f.name for f in dataclasses.fields(DQNConfig)] == [
        "gamma", "lr", "batch_size", "buffer_size", "replay", "learning_starts", "grad_steps_per_vec_step",
        "target_update_every", "grad_clip", "eps_start", "eps_end", "eps_decay_fraction", "hidden", "conv_channels",
        "double", *M7_KEYS, "keep_checkpoints", "device", "torch_threads",
    ]
    assert {k: getattr(DQNConfig(), k) for k in M7_KEYS} == M7_DEFAULTS
    # The defaults pass the validation (with a budget that reaches learning_starts), on either observation.
    cfg = ExperimentConfig("d", EnvConfig(), "dqn", DQNConfig(), TrainConfig())
    assert config_from_dict(config_to_dict(cfg)) == cfg
    grid = ExperimentConfig("d", EnvConfig(obs=OBS_B8), "dqn", DQNConfig(), TrainConfig())
    assert config_from_dict(config_to_dict(grid)) == grid


@pytest.mark.parametrize("name", sorted(SHIPPED))
def test_shipped_dqn_config_files_load(name):
    cfg = load_config(CONFIGS / name)
    expected = SHIPPED[name]
    free = FREE_FIELDS.get(name, ())
    if free:  # device cpu or cuda, torch_threads null or >= 1: the benchmark of D7 decides
        assert cfg.agent.device in ("cpu", "cuda")
        assert cfg.agent.torch_threads is None or cfg.agent.torch_threads >= 1
        cfg = dataclasses.replace(cfg, agent=dataclasses.replace(cfg.agent, **{f: getattr(expected.agent, f) for f in free}))
    assert cfg == expected
    assert type(cfg.agent) is DQNConfig and type(cfg.agent.hidden) is tuple and type(cfg.agent.conv_channels) is tuple
    assert cfg.agent.replay is True
    written = yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))
    every_key = {"kind", *(f.name for f in dataclasses.fields(DQNConfig))}
    # M7 (D10): the four batch files write the five new keys out; the smoke files are unchanged and rely
    # on the defaults, so for them "every key" means every key up to M6.
    required = every_key if name in BATCH_FILES else every_key - set(M7_KEYS)
    if cfg.env.obs.kind == "B":
        # Every field is written out in the file, none is left to a default.
        assert set(written["agent"]) == required
        assert written["env"]["obs"] == {"kind": "B", "age_plane": True, "heading_planes": True}
    else:
        # conv_channels is unused on observation A; D6 adds the line ``replay: true`` to "the two shipped
        # A configs" (the experiment files, whose comment says every DQNConfig field is written out).
        assert required - {"conv_channels", "replay"} <= set(written["agent"]) <= required
        if name.startswith("dqn_A"):
            assert written["agent"]["replay"] is True
    assert set(written["train"]) == {f.name for f in dataclasses.fields(TrainConfig)}


@pytest.mark.parametrize("name", BATCH_FILES)
def test_shipped_batch_files_write_the_m7_fields_after_double_and_list_the_m7_variants(name):
    """M7 (D10): ``dueling``, ``per``, ``per_alpha``, ``per_beta_start``, ``per_eps`` follow ``double: false``
    with their defaults, and the head comment names the three M7 variants run with ``--set``."""
    text = (CONFIGS / name).read_text(encoding="utf-8")
    written = yaml.safe_load(text)
    keys = list(written["agent"])
    i = keys.index("double")
    assert keys[i + 1 : i + 6] == list(M7_KEYS)
    assert written["agent"]["double"] is False
    assert {k: written["agent"][k] for k in M7_KEYS} == M7_DEFAULTS
    assert type(written["agent"]["per_eps"]) is float and type(written["agent"]["per_alpha"]) is float
    exp = written["exp"]
    comment = "".join(line for line in text.splitlines(keepends=True) if line.startswith("#"))
    for variant, setting in (("double", "agent.double=true"), ("dueling", "agent.dueling=true"), ("per", "agent.per=true")):
        assert f"--exp {exp}_{variant}" in comment, variant
        assert setting in comment, variant
    # The file still loads to the configuration of SHIPPED (checked above) and the variants load too.
    for setting in ("agent.double=true", "agent.dueling=true", "agent.per=true"):
        cfg = load_config(CONFIGS / name, [setting, f"exp={exp}_x"])
        key = setting.split("=")[0].split(".")[1]
        assert getattr(cfg.agent, key) is True


@pytest.mark.parametrize("name", ["smoke_8x8.yaml", "smoke_B_8x8.yaml"])
def test_smoke_files_are_unchanged_and_rely_on_the_m7_defaults(name):
    written = yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))
    assert not set(M7_KEYS) & set(written["agent"])
    cfg = load_config(CONFIGS / name)
    assert {k: getattr(cfg.agent, k) for k in M7_KEYS} == M7_DEFAULTS
    assert cfg == SHIPPED[name]


@pytest.mark.parametrize("name", sorted(SHIPPED))
def test_shipped_dqn_config_files_start_with_a_comment_saying_how_to_run_them(name):
    text = (CONFIGS / name).read_text(encoding="utf-8")
    comment = "".join(line for line in text.splitlines(keepends=True) if line.startswith("#"))
    assert text.startswith("# ")
    assert f"python -m snake_rl.train --config configs/{name} --seed 0" in comment
    assert f"python scripts/run_matrix.py --config configs/{name} --seeds 0 1 2 3 4" in comment
    assert "--set needs a new --exp" in comment


TABULAR = {
    "tabular_q_8x8.yaml": ("tabular_q_W8", 8, 0.9, TrainConfig(50, 500_000, 25_000, 100, 10_000)),
    "tabular_q_10x10.yaml": ("tabular_q_W10", 10, 0.9, TrainConfig(50, 1_000_000, 50_000, 100, 10_000)),
    "smoke_tabular.yaml": ("smoke_tabular", 8, 0.99, TrainConfig(16, 32_000, 8_000, 20, 10_000)),
}


@pytest.mark.parametrize("name", sorted(TABULAR))
def test_tabular_config_files_still_load_to_the_same_objects(name):
    exp, W, gamma, train_cfg = TABULAR[name]
    cfg = load_config(CONFIGS / name)
    assert cfg == ExperimentConfig(
        exp=exp,
        env=EnvConfig(W=W, action_mode="relative", starve_limit=None, obs=ObsConfig(kind="A"),
                      reward=RewardConfig(variant="R0")),
        agent_kind="tabular_q",
        agent=TabularQConfig(alpha=0.1, gamma=gamma, eps_start=1.0, eps_end=0.05, eps_decay_fraction=0.2, q_init=0.0),
        train=train_cfg,
    )
    assert list(config_to_dict(cfg)["agent"]) == ["kind", "alpha", "gamma", "eps_start", "eps_end",
                                                  "eps_decay_fraction", "q_init"]


def test_only_exp_and_agent_kind_are_required_for_dqn(tmp_path):
    cfg = load_config(write_yaml(tmp_path / "c.yaml", {"exp": "minimal", "agent": {"kind": "dqn"}}))
    assert cfg == ExperimentConfig("minimal", EnvConfig(), "dqn", DQNConfig(), TrainConfig())


def test_the_hand_written_dqn_dict_loads(tmp_path):
    cfg = load_config(write_yaml(tmp_path / "c.yaml", dqn_dict()))
    assert cfg.agent == DQNConfig(gamma=0.9, batch_size=16, buffer_size=1000, learning_starts=400,
                                  target_update_every=50, hidden=(16, 16), device="cpu", torch_threads=1)
    assert config_to_dict(cfg) == dqn_dict()


INT_FIELDS = ("agent.batch_size", "agent.buffer_size", "agent.learning_starts", "agent.grad_steps_per_vec_step",
              "agent.target_update_every", "agent.torch_threads")
NULLABLE = ("agent.grad_steps_per_vec_step", "agent.torch_threads")
NUMBER_FIELDS = ("agent.gamma", "agent.lr", "agent.grad_clip", "agent.eps_start", "agent.eps_end",
                 "agent.eps_decay_fraction", "agent.per_alpha", "agent.per_beta_start", "agent.per_eps")  # M7: per_*
FLAGS = ("agent.double", "agent.keep_checkpoints", "agent.replay", "agent.dueling", "agent.per")  # M7: dueling, per
#: Two list-of-positive-integers fields with the same checks (M6: conv_channels).
LIST_FIELDS = ("agent.hidden", "agent.conv_channels")

#: (changes to the valid dict, key the error must name)
INVALID = [
    # ---- unknown keys: a typo, and the keys of the other agent kind
    ({"agent.learning_rate": 1.0e-3}, "agent.learning_rate"),
    ({"agent.alpha": 0.1}, "agent.alpha"),
    ({"agent.q_init": 0.0}, "agent.q_init"),
    ({"agent.alpha": 0.1}, "agent.kind"),  # the message says which kind the keys belong to
    # ---- types
    *[({key: 64.0}, key) for key in INT_FIELDS],
    *[({key: True}, key) for key in INT_FIELDS],
    *[({key: "64"}, key) for key in INT_FIELDS],
    *[({key: None}, key) for key in INT_FIELDS if key not in NULLABLE],
    *[({key: "0.5"}, key) for key in NUMBER_FIELDS],
    *[({key: True}, key) for key in NUMBER_FIELDS],
    *[({key: None}, key) for key in NUMBER_FIELDS],
    ({"agent.lr": float("inf")}, "agent.lr"),
    ({"agent.lr": float("nan")}, "agent.lr"),
    ({"agent.grad_clip": float("inf")}, "agent.grad_clip"),
    *[({key: 1}, key) for key in FLAGS],
    *[({key: "true"}, key) for key in FLAGS],
    *[({key: None}, key) for key in FLAGS],
    *[({key: 64}, key) for key in LIST_FIELDS],
    *[({key: "64"}, key) for key in LIST_FIELDS],
    *[({key: None}, key) for key in LIST_FIELDS],
    *[({key: [64, 0]}, key) for key in LIST_FIELDS],
    *[({key: [-8]}, key) for key in LIST_FIELDS],
    *[({key: [64.0]}, key) for key in LIST_FIELDS],
    *[({key: [True]}, key) for key in LIST_FIELDS],
    *[({key: [[64]]}, key) for key in LIST_FIELDS],
    *[({key: {"a": 64}}, key) for key in LIST_FIELDS],
    ({"agent.device": "gpu"}, "agent.device"),
    ({"agent.device": "cuda:0"}, "agent.device"),
    ({"agent.device": 0}, "agent.device"),
    ({"agent.device": None}, "agent.device"),
    # ---- ranges
    ({"agent.gamma": -0.01}, "agent.gamma"),
    ({"agent.gamma": 1.01}, "agent.gamma"),
    ({"agent.lr": 0}, "agent.lr"),
    ({"agent.lr": -2.5e-4}, "agent.lr"),
    ({"agent.batch_size": 0}, "agent.batch_size"),
    ({"agent.batch_size": -16}, "agent.batch_size"),
    ({"agent.buffer_size": 15}, "agent.buffer_size"),  # below batch_size = 16
    ({"agent.buffer_size": 0}, "agent.buffer_size"),
    ({"agent.learning_starts": 15}, "agent.learning_starts"),  # below batch_size = 16
    ({"agent.learning_starts": 0}, "agent.learning_starts"),
    ({"agent.learning_starts": 2401}, "agent.learning_starts"),  # above train.total_env_steps = 2400
    ({"train.total_env_steps": 800, "agent.learning_starts": 1000}, "agent.learning_starts"),
    ({"agent.grad_steps_per_vec_step": 0}, "agent.grad_steps_per_vec_step"),
    ({"agent.grad_steps_per_vec_step": -1}, "agent.grad_steps_per_vec_step"),
    ({"agent.target_update_every": 0}, "agent.target_update_every"),
    ({"agent.target_update_every": -1000}, "agent.target_update_every"),
    ({"agent.grad_clip": 0}, "agent.grad_clip"),
    ({"agent.grad_clip": -10.0}, "agent.grad_clip"),
    ({"agent.eps_start": 0.2, "agent.eps_end": 0.5}, "agent.eps_end"),
    ({"agent.eps_start": 1.5}, "agent.eps_start"),
    ({"agent.eps_end": -0.1}, "agent.eps_end"),
    ({"agent.eps_decay_fraction": 1.5}, "agent.eps_decay_fraction"),
    ({"agent.eps_decay_fraction": -0.1}, "agent.eps_decay_fraction"),
    ({"agent.torch_threads": 0}, "agent.torch_threads"),
    ({"agent.torch_threads": -2}, "agent.torch_threads"),
    # M7 (D4): per_alpha and per_beta_start in [0, 1], per_eps > 0
    ({"agent.per_alpha": -0.01}, "agent.per_alpha"),
    ({"agent.per_alpha": 1.01}, "agent.per_alpha"),
    ({"agent.per_beta_start": -0.01}, "agent.per_beta_start"),
    ({"agent.per_beta_start": 1.01}, "agent.per_beta_start"),
    ({"agent.per_eps": 0}, "agent.per_eps"),
    ({"agent.per_eps": 0.0}, "agent.per_eps"),
    ({"agent.per_eps": -1.0e-6}, "agent.per_eps"),
    ({"agent.per_eps": float("inf")}, "agent.per_eps"),
    ({"agent.per_eps": float("nan")}, "agent.per_eps"),
    # ---- combinations
    ({"env.reward.variant": "R2"}, "env.reward.gamma"),  # 0.99 against agent.gamma 0.9
    ({"env.reward.variant": "R2"}, "agent.gamma"),
    # M6: without replay there is exactly one gradient step per vec-step
    ({"agent.replay": False, "agent.grad_steps_per_vec_step": 2}, "agent.grad_steps_per_vec_step"),
    ({"agent.replay": False, "agent.grad_steps_per_vec_step": 3}, "agent.grad_steps_per_vec_step"),
    # M6: with replay the buffer must hold at least one vec-step (n_envs = 8; batch_size 4 <= 7 is fine)
    ({"agent.buffer_size": 7, "agent.batch_size": 4, "agent.learning_starts": 8}, "agent.buffer_size"),
    ({"agent.replay": True, "agent.buffer_size": 4, "agent.batch_size": 4, "agent.learning_starts": 8},
     "agent.buffer_size"),
    # M7 (D4): prioritised replay needs the replay buffer
    ({"agent.per": True, "agent.replay": False}, "agent.per"),
    ({"agent.per": True, "agent.replay": False, "agent.grad_steps_per_vec_step": 1}, "agent.per"),
    # ---- the shared sections are checked as for every kind
    ({"train.n_envs": 7}, "train.eval_every"),
    ({"env.W": 3}, "env.W"),
]


def _case_id(case) -> str:
    changes, key = case
    return ",".join(f"{k}={v!r}" for k, v in changes.items()) + f"->{key}"


@pytest.mark.parametrize("case", INVALID, ids=_case_id)
def test_invalid_dqn_config_is_a_value_error_naming_the_key(case, tmp_path):
    changes, key = case
    message = config_error(write_yaml(tmp_path / "c.yaml", edited(changes)))
    assert names_key(message, key), message


def test_no_replay_with_several_gradient_steps_per_vec_step_is_explained(tmp_path):
    """M6 (D3): the message names the key and says it is about replay."""
    message = config_error(write_yaml(tmp_path / "c.yaml",
                                      edited({"agent.replay": False, "agent.grad_steps_per_vec_step": 2})))
    assert names_key(message, "agent.grad_steps_per_vec_step") and "replay" in message, message
    assert "replay" in config_error(SMOKE, ["agent.replay=false", "agent.grad_steps_per_vec_step=4"])


def test_per_without_replay_is_explained(tmp_path):
    """M7 (D4): ``agent.per: true`` with ``agent.replay: false`` names ``agent.per`` and says it is about replay."""
    message = config_error(write_yaml(tmp_path / "c.yaml", edited({"agent.per": True, "agent.replay": False})))
    assert names_key(message, "agent.per") and "replay" in message, message
    message = config_error(SMOKE, ["agent.per=true", "agent.replay=false"])
    assert names_key(message, "agent.per") and "replay" in message, message
    # The same switch with replay is fine, and the plain default is fine without replay.
    assert load_config(SMOKE, ["agent.per=true"]).agent.per is True
    assert load_config(SMOKE, ["agent.replay=false"]).agent.per is False


def test_config_docstring_lists_the_m7_keys_with_their_defaults():
    """D4: the ``agent`` block of the module docstring lists the five keys after ``double``."""
    doc = config_module.__doc__
    for text in ("dueling: false", "per: false", "per_alpha: 0.6", "per_beta_start: 0.4", "per_eps: 1.0e-6"):
        assert text in doc, text
    assert doc.index("double: false") < doc.index("dueling: false") < doc.index("per: false") < doc.index("per_eps: 1.0e-6")


@pytest.mark.parametrize("age_plane, heading_planes", [(True, True), (False, True), (True, False), (False, False)])
def test_observation_b_is_accepted_for_dqn_and_still_refused_for_tabular_q(age_plane, heading_planes, tmp_path):
    """M6 (D3): the M5 refusal of observation B for dqn is gone, with any plane set."""
    changes = {"env.obs.kind": "B", "env.obs.age_plane": age_plane, "env.obs.heading_planes": heading_planes}
    cfg = load_config(write_yaml(tmp_path / "c.yaml", edited(changes)))
    assert cfg.agent_kind == "dqn" and type(cfg.agent) is DQNConfig
    assert cfg.env.obs == ObsConfig(kind="B", age_plane=age_plane, heading_planes=heading_planes)
    assert cfg.env.obs.shape(cfg.env.W) == (3 + int(age_plane) + 4 * int(heading_planes), 8, 8)
    assert config_from_dict(config_to_dict(cfg)) == cfg
    # The M6 ablations are run with --set on the shipped files.
    overrides = [f"env.obs.age_plane={str(age_plane).lower()}", f"env.obs.heading_planes={str(heading_planes).lower()}"]
    assert load_config(SMOKE, ["env.obs.kind=B", "exp=smoke_B", *overrides]).env.obs == cfg.env.obs
    message = config_error(SMOKE_TABULAR, ["env.obs.kind=B", *overrides])
    assert names_key(message, "env.obs.kind") and "tabular_q" in message, message


def test_switching_the_kind_of_a_tabular_file_names_the_kind_and_the_stray_key():
    message = config_error(SMOKE_TABULAR, ["agent.kind=dqn"])
    assert names_key(message, "agent.kind") and names_key(message, "agent.alpha")
    message = config_error(SMOKE, ["agent.kind=tabular_q"])
    assert names_key(message, "agent.kind") and names_key(message, "agent.lr")


VALID = [
    ({"agent.hidden": []}, lambda c: c.agent.hidden == ()),
    ({"agent.hidden": [8]}, lambda c: c.agent.hidden == (8,)),
    ({"agent.hidden": [32, 16, 8]}, lambda c: c.agent.hidden == (32, 16, 8)),
    ({"agent.gamma": 0}, lambda c: c.agent.gamma == 0.0),
    ({"agent.gamma": 1}, lambda c: c.agent.gamma == 1.0 and type(c.agent.gamma) is float),
    ({"agent.lr": 1}, lambda c: c.agent.lr == 1.0 and type(c.agent.lr) is float),
    ({"agent.lr": 1.0e-5}, lambda c: c.agent.lr == 1e-5),
    ({"agent.batch_size": 1, "agent.learning_starts": 8}, lambda c: c.agent.batch_size == 1),
    ({"agent.buffer_size": 16}, lambda c: c.agent.buffer_size == c.agent.batch_size),
    ({"agent.learning_starts": 16}, lambda c: c.agent.learning_starts == c.agent.batch_size),
    ({"agent.learning_starts": 2400}, lambda c: c.agent.learning_starts == c.train.total_env_steps),
    ({"agent.buffer_size": 16, "agent.learning_starts": 2000}, lambda c: c.agent.learning_starts > c.agent.buffer_size),
    ({"agent.grad_steps_per_vec_step": 1}, lambda c: c.agent.grad_steps_per_vec_step == 1),
    ({"agent.grad_steps_per_vec_step": 64}, lambda c: c.agent.grad_steps_per_vec_step == 64),
    ({"agent.target_update_every": 1}, lambda c: c.agent.target_update_every == 1),
    ({"agent.grad_clip": 0.5}, lambda c: c.agent.grad_clip == 0.5),
    ({"agent.grad_clip": 1}, lambda c: c.agent.grad_clip == 1.0 and type(c.agent.grad_clip) is float),
    ({"agent.eps_start": 0.3, "agent.eps_end": 0.3}, lambda c: c.agent.eps_start == c.agent.eps_end == 0.3),
    ({"agent.eps_decay_fraction": 0}, lambda c: c.agent.eps_decay_fraction == 0.0),
    ({"agent.eps_decay_fraction": 1}, lambda c: c.agent.eps_decay_fraction == 1.0),
    ({"agent.double": True}, lambda c: c.agent.double is True),
    ({"agent.keep_checkpoints": False}, lambda c: c.agent.keep_checkpoints is False),
    ({"agent.device": "auto"}, lambda c: c.agent.device == "auto"),
    ({"agent.device": "cuda"}, lambda c: c.agent.device == "cuda"),  # availability is checked when the run starts
    ({"agent.torch_threads": None}, lambda c: c.agent.torch_threads is None),
    ({"agent.torch_threads": 8}, lambda c: c.agent.torch_threads == 8),
    ({"env.reward.variant": "R2", "env.reward.gamma": 0.9}, lambda c: c.env.reward.gamma == c.agent.gamma == 0.9),
    ({"env.reward.variant": "R1"}, lambda c: c.env.reward.gamma != c.agent.gamma),  # only R2 ties the two
    ({"env.action_mode": "absolute"}, lambda c: c.env.action_mode == "absolute"),
    # ---- M6
    ({"agent.replay": False}, lambda c: c.agent.replay is False),
    ({"agent.replay": False, "agent.grad_steps_per_vec_step": 1}, lambda c: c.agent.grad_steps_per_vec_step == 1),
    ({"agent.replay": False, "agent.grad_steps_per_vec_step": None}, lambda c: c.agent.grad_steps_per_vec_step is None),
    # without replay the buffer size is not held against n_envs (it is unused)
    ({"agent.replay": False, "agent.buffer_size": 1, "agent.batch_size": 1, "agent.learning_starts": 8},
     lambda c: c.agent.buffer_size == 1 < c.train.n_envs),
    # with replay: buffer_size == n_envs is the boundary
    ({"agent.buffer_size": 8, "agent.batch_size": 8, "agent.learning_starts": 8},
     lambda c: c.agent.buffer_size == c.train.n_envs == 8),
    ({"agent.conv_channels": []}, lambda c: c.agent.conv_channels == ()),
    ({"agent.conv_channels": [8]}, lambda c: c.agent.conv_channels == (8,)),
    ({"agent.conv_channels": [32, 64, 64]}, lambda c: c.agent.conv_channels == (32, 64, 64)),
    ({"agent.conv_channels": [4], "env.obs.kind": "B"}, lambda c: c.env.obs.kind == "B" and c.agent.conv_channels == (4,)),
    ({"agent.conv_channels": [], "env.obs.kind": "B"}, lambda c: c.agent.conv_channels == ()),  # the MLP on B
    ({"agent.conv_channels": [4], "env.obs.kind": "A"}, lambda c: c.agent.conv_channels == (4,)),  # kept, unused
    # ---- M7 (D4)
    ({"agent.dueling": True}, lambda c: c.agent.dueling is True),
    ({"agent.per": True}, lambda c: c.agent.per is True and c.agent.replay is True),
    ({"agent.per": True, "agent.replay": True, "agent.grad_steps_per_vec_step": 3},
     lambda c: c.agent.per is True and c.agent.grad_steps_per_vec_step == 3),
    ({"agent.per": False, "agent.replay": False}, lambda c: c.agent.per is False and c.agent.replay is False),
    ({"agent.per_alpha": 0}, lambda c: c.agent.per_alpha == 0.0 and type(c.agent.per_alpha) is float),
    ({"agent.per_alpha": 1}, lambda c: c.agent.per_alpha == 1.0 and type(c.agent.per_alpha) is float),
    ({"agent.per_alpha": 0.5}, lambda c: c.agent.per_alpha == 0.5),
    ({"agent.per_beta_start": 0}, lambda c: c.agent.per_beta_start == 0.0 and type(c.agent.per_beta_start) is float),
    ({"agent.per_beta_start": 1}, lambda c: c.agent.per_beta_start == 1.0),
    ({"agent.per_eps": 1}, lambda c: c.agent.per_eps == 1.0 and type(c.agent.per_eps) is float),
    ({"agent.per_eps": 1.0e-9}, lambda c: c.agent.per_eps == 1e-9),
    ({"agent.dueling": True, "agent.double": True, "agent.per": True},
     lambda c: c.agent.dueling and c.agent.double and c.agent.per),
]


@pytest.mark.parametrize("case", VALID, ids=lambda case: ",".join(f"{k}={v!r}" for k, v in case[0].items()))
def test_dqn_boundary_values_are_accepted(case, tmp_path):
    changes, check = case
    cfg = load_config(write_yaml(tmp_path / "c.yaml", edited(changes)))
    assert check(cfg)
    assert type(cfg.agent.batch_size) is int and type(cfg.agent.hidden) is tuple
    assert type(cfg.agent.conv_channels) is tuple and type(cfg.agent.replay) is bool
    assert type(cfg.agent.dueling) is bool and type(cfg.agent.per) is bool  # M7
    assert all(type(getattr(cfg.agent, name)) is float for name in ("per_alpha", "per_beta_start", "per_eps"))


def test_dqn_overrides_and_frozen_config():
    cfg = load_config(SMOKE, ["agent.hidden=[32]", "agent.double=true", "agent.lr=1.0e-3", "agent.torch_threads=null",
                              "agent.grad_steps_per_vec_step=2", "exp=other"])
    base = load_config(SMOKE)
    assert cfg.agent == dataclasses.replace(base.agent, hidden=(32,), double=True, lr=1e-3, torch_threads=None,
                                            grad_steps_per_vec_step=2)
    assert cfg.exp == "other"
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.agent.lr = 1.0
    assert "1.0e-3" in config_error(SMOKE, ["agent.lr=1e-3"])  # the YAML trap is explained for dqn keys too
    # M6 overrides on the shipped files: the ablation switches and the CNN sizes.
    cfg = load_config(SMOKE_B, ["agent.replay=false", "agent.conv_channels=[]", "env.obs.age_plane=false",
                                "agent.target_update_every=1", "exp=ablation"])
    base = load_config(SMOKE_B)
    assert cfg.agent == dataclasses.replace(base.agent, replay=False, conv_channels=(), target_update_every=1)
    assert cfg.env.obs == ObsConfig(kind="B", age_plane=False, heading_planes=True) and cfg.exp == "ablation"
    # M7 overrides on the shipped files: the Dueling head and prioritised replay.
    cfg = load_config(SMOKE, ["agent.dueling=true", "agent.per=true", "agent.per_beta_start=0.5", "agent.per_alpha=0.7",
                              "agent.per_eps=1.0e-5", "exp=m7"])
    base = load_config(SMOKE)
    assert cfg.agent == dataclasses.replace(base.agent, dueling=True, per=True, per_beta_start=0.5, per_alpha=0.7,
                                            per_eps=1e-5)
    assert cfg.agent.replay is True and cfg.exp == "m7"
    assert "1.0e-5" in config_error(SMOKE, ["agent.per_eps=1e-5"]) or "1.0e-3" in config_error(SMOKE, ["agent.per_eps=1e-5"])


@pytest.mark.parametrize("override", ["agent.lr=1.0e9", "agent.lr=1e-3", "agent.lr=1e9"])
def test_set_number_hint_covers_an_exponent_without_a_sign(override):
    """M6 (D3): PyYAML reads ``1.0e9`` as a string too (the exponent needs a
    sign); the hint says to write a dot AND a sign, e.g. ``1.0e-3``, ``1.0e+9``."""
    message = config_error(SMOKE, [override])
    assert names_key(message, "agent.lr"), message
    assert "1.0e+9" in message or "1.0e-3" in message, message


def test_dqn_config_to_dict_spells_out_every_field_with_the_tuples_as_lists():
    d = config_to_dict(load_config(SMOKE))
    assert d["agent"] == {
        "kind": "dqn", "gamma": 0.9, "lr": 2.5e-4, "batch_size": 64, "buffer_size": 10000, "replay": True,
        "learning_starts": 1024, "grad_steps_per_vec_step": None, "target_update_every": 250, "grad_clip": 10.0,
        "eps_start": 1.0, "eps_end": 0.05, "eps_decay_fraction": 0.2, "hidden": [64, 64],
        "conv_channels": [32, 64, 64], "double": False, "dueling": False, "per": False, "per_alpha": 0.6,
        "per_beta_start": 0.4, "per_eps": 1e-6, "keep_checkpoints": True, "device": "cpu", "torch_threads": 1,
    }
    assert type(d["agent"]["hidden"]) is list and type(d["agent"]["conv_channels"]) is list
    assert type(d["agent"]["per_eps"]) is float and type(d["agent"]["per"]) is bool  # M7
    assert list(d) == ["exp", "env", "agent", "train"]
    assert list(d["agent"]) == ["kind", *(f.name for f in dataclasses.fields(DQNConfig))]
    b = config_to_dict(load_config(SMOKE_B))
    assert b["agent"]["conv_channels"] == [16, 32] and b["agent"]["hidden"] == [128] and b["agent"]["replay"] is True
    assert b["env"]["obs"] == {"kind": "B", "age_plane": True, "heading_planes": True}
    assert config_to_dict(load_config(SMOKE, ["agent.conv_channels=[]"]))["agent"]["conv_channels"] == []


ROUND_TRIP = [
    (SMOKE, ()),
    (CONFIGS / "dqn_A_8x8.yaml", ()),
    (CONFIGS / "dqn_A_10x10.yaml", ()),
    (SMOKE, TINY),
    (SMOKE, VARIANT),
    (SMOKE, ("agent.hidden=[]", "env.reward.variant=R2", "env.reward.gamma=0.9", "agent.lr=1.2345678901234567e-4")),
    # M6
    (SMOKE_B, ()),
    (CONFIGS / "dqn_B_8x8.yaml", ()),
    (CONFIGS / "dqn_B_10x10.yaml", ()),
    (SMOKE_B, TINY_B),
    (SMOKE, NOREPLAY),
    (SMOKE_B, ("agent.conv_channels=[]", "agent.replay=false", "env.obs.age_plane=false",
               "env.obs.heading_planes=false", "agent.target_update_every=1")),
    (SMOKE, ("env.obs.kind=B", "agent.conv_channels=[16]")),
    # M7
    (SMOKE, ("agent.per=true", "agent.per_beta_start=0.5", "agent.per_alpha=0.7", "agent.per_eps=1.0e-5", "exp=per")),
    (SMOKE_B, ("agent.dueling=true", "agent.double=true", "exp=dueling_double")),
    (CONFIGS / "dqn_A_8x8.yaml", ("agent.per=true", "exp=dqn_A_W8_per")),
    (CONFIGS / "dqn_B_10x10.yaml", ("agent.dueling=true", "exp=dqn_B8_W10_dueling")),
]


@pytest.mark.parametrize("path, overrides", ROUND_TRIP)
def test_dqn_config_round_trips_through_a_yaml_file(path, overrides, tmp_path):
    cfg = load_config(path, overrides)
    d = config_to_dict(cfg)
    assert json.loads(json.dumps(d)) == d  # plain Python types only (a tuple would come back as a list)
    assert config_from_dict(d) == cfg
    assert load_config(write_yaml(tmp_path / "copy.yaml", d)) == cfg
    assert config_to_dict(load_config(tmp_path / "copy.yaml")) == d


# ============================================================ torch stays out
def run_python(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)


def imported_modules(stderr: str) -> set[str]:
    """Module names from the ``-X importtime`` trace on stderr."""
    return {line.rsplit("|", 1)[1].strip() for line in stderr.splitlines() if line.startswith("import time:")}


def test_validating_a_dqn_config_does_not_import_torch():
    code = (
        "import sys\n"
        "import snake_rl.config, snake_rl.agents.dqn, snake_rl.agents.dqn.config, snake_rl.train\n"
        f"cfg = snake_rl.config.load_config(r'{SMOKE}')\n"
        "print(cfg.agent_kind, 'torch' in sys.modules)\n"
    )
    done = run_python("-c", code)
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["dqn", "False"]


def test_help_does_not_import_torch():
    done = run_python("-X", "importtime", "-m", "snake_rl.train", "--help")
    assert done.returncode == 0 and "--config" in done.stdout
    modules = imported_modules(done.stderr)
    assert {"numpy", "snake_rl.config", "snake_rl.agents.dqn.replay"} <= modules  # the trace is what it seems
    assert "torch" not in modules


def test_a_tabular_run_does_not_import_torch(tmp_path):
    done = run_python("-X", "importtime", "-m", "snake_rl.train", "--config", str(SMOKE_TABULAR), "--seed", "0",
                      "--runs-dir", str(tmp_path), "--quiet", "--set", "train.n_envs=8", "train.total_env_steps=800",
                      "train.eval_every=800", "train.eval_episodes=4", "env.starve_limit=20")
    assert done.returncode == 0, done.stderr
    assert "torch" not in imported_modules(done.stderr)
    run_dir = tmp_path / "smoke_tabular" / "0"
    assert sorted(p.name for p in run_dir.iterdir()) == ["config.yaml", "final_eval.json", "metrics.csv",
                                                         "q_table.npz"]
    # The run block of a tabular run has no torch entries.
    assert sorted(read_run_config(run_dir)["run"]) == sorted(["seed", "env_seed", "git", "started", "python", "numpy",
                                                              "platform", "argv"])


# ============================================================ run directory
def test_a_dqn_run_writes_exactly_the_specified_files(tiny_run):
    assert tiny_run.dir == tiny_run.runs / "tiny_dqn" / str(TINY_SEED)
    assert sorted(p.name for p in tiny_run.dir.iterdir()) == RUN_ENTRIES
    # One weights-only checkpoint per evaluation, named by the 8-digit step count.
    assert checkpoint_names(tiny_run.dir) == ["step_00000000.pt", "step_00000800.pt", "step_00001600.pt",
                                              "step_00002400.pt"]
    assert sorted(tree(tiny_run.runs)) == sorted(
        [f"tiny_dqn/{TINY_SEED}/{name}" for name in RUN_ENTRIES]
        + [f"tiny_dqn/{TINY_SEED}/checkpoints/{name}" for name in checkpoint_names(tiny_run.dir)]
        + ["tiny_dqn", f"tiny_dqn/{TINY_SEED}"]
    )


def test_run_block_records_torch_device_and_threads(tiny_run):
    recorded = read_run_config(tiny_run.dir)
    run = recorded.pop("run")
    assert recorded == config_to_dict(tiny_run.cfg)
    assert recorded["agent"]["hidden"] == [16]
    assert list(run) == ["seed", "env_seed", "git", "started", "python", "numpy", "platform", "argv", "torch",
                         "device", "cuda_device", "torch_threads"]
    assert run["seed"] == TINY_SEED and run["env_seed"] == 100_000 * (TINY_SEED + 1)
    assert run["torch"] == str(torch.__version__) and type(run["torch"]) is str
    assert run["device"] == "cpu"
    assert run["cuda_device"] is None
    assert run["torch_threads"] == 1
    assert load_config(tiny_run.dir / "config.yaml") == tiny_run.cfg  # the recipe to repeat the run


def test_run_info_resolves_auto_and_reports_the_threads_in_effect():
    info = train_module._dqn_run_info(load_config(SMOKE, ["agent.device=auto", "agent.torch_threads=null"]))
    assert info["device"] == ("cuda" if torch.cuda.is_available() else "cpu")
    assert (info["cuda_device"] is None) == (info["device"] == "cpu")
    assert info["torch_threads"] == torch.get_num_threads()
    assert train_module._dqn_run_info(load_config(SMOKE, ["agent.torch_threads=3"]))["torch_threads"] == 3


def test_the_trainer_applies_torch_threads_and_null_leaves_the_setting_alone(tmp_path):
    """``agent.torch_threads`` is how several seeds share one CPU (five
    processes of two threads each), and the run block records it as the number
    of threads in effect: the trainer has to apply it, and ``null`` has to
    leave torch's setting as it found it."""
    short = [*TINY, "train.total_env_steps=1600"]  # two evaluations, learning from 1000 env steps on
    before = torch.get_num_threads()
    try:
        torch.set_num_threads(3)
        run_dir = train(load_config(SMOKE, [*short, "agent.torch_threads=2"]), 0, tmp_path, quiet=True)
        assert torch.get_num_threads() == 2
        assert read_run_config(run_dir)["run"]["torch_threads"] == 2
        torch.set_num_threads(3)
        run_dir = train(load_config(SMOKE, [*short, "agent.torch_threads=null"]), 1, tmp_path, quiet=True)
        assert torch.get_num_threads() == 3
        assert read_run_config(run_dir)["run"]["torch_threads"] == 3
    finally:
        torch.set_num_threads(before)


def test_metrics_header_and_row_positions(tiny_run):
    raw = (tiny_run.dir / "metrics.csv").read_bytes()
    assert b"\r" not in raw
    lines = raw.decode("utf-8").splitlines()
    assert lines[0] == HEADER
    assert ",".join(train_module.DQN_COLUMNS) == HEADER
    rows = read_metrics(tiny_run.dir)
    assert len(lines) - 1 == len(rows) == 4
    assert [r["env_steps"] for r in rows] == [0, 800, 1_600, 2_400]
    assert [r["vec_steps"] for r in rows] == [0, 100, 200, 300]


def test_grad_steps_and_buffer_size_columns(tiny_run):
    cfg = tiny_run.cfg
    rows = read_metrics(tiny_run.dir)
    # n_envs 8 -> 2 gradient steps per vec-step, from vec-step 125 (1000 env steps) on.
    assert [r["grad_steps"] for r in rows] == [0, 0, 152, 352]
    assert [r["grad_steps"] for r in rows] == [expected_grad_steps(cfg, r["vec_steps"]) for r in rows]
    assert [r["buffer_size"] for r in rows] == [0, 800, 1_000, 1_000]
    assert [r["buffer_size"] for r in rows] == [min(r["env_steps"], cfg.agent.buffer_size) for r in rows]
    assert DQNAgent.load(tiny_run.dir / "model.pt").grad_steps == 352


def test_rows_before_learning_starts_have_empty_update_cells(tiny_run):
    first, second, *later = read_metrics(tiny_run.dir)
    assert first["epsilon"] == tiny_run.cfg.agent.eps_start == 1.0
    assert first["train_episodes"] == 0 and first["train_mean_score"] is None and first["train_truncated"] == 0
    for row in (first, second):  # 800 env steps < learning_starts = 1000: nothing learned yet
        assert all(row[name] is None for name in UPDATE_COLUMNS)
        assert row["grad_steps"] == 0
    assert second["train_episodes"] > 0 and second["train_mean_score"] is not None
    with open(tiny_run.dir / "metrics.csv", newline="", encoding="utf-8") as f:
        assert all(row[name] == "" for row in list(csv.DictReader(f))[:2] for name in UPDATE_COLUMNS)  # empty strings
    for row in later:
        assert all(value is not None and np.isfinite(value) for value in row.values())
        assert row["loss"] > 0 and row["grad_norm"] > 0
        assert sum(row[f"eval_frac_{c}"] for c in ("wall", "self", "starve", "win")) == pytest.approx(1.0)
    # The untrained and the not-yet-trained network are the same network.
    assert all(first[name] == second[name] for name in train_module.EVAL_COLUMNS)


def test_time_columns_are_cumulative_and_inside_the_wall_time(tiny_run):
    rows = read_metrics(tiny_run.dir)
    assert rows[0]["time_env_s"] == rows[0]["time_act_s"] == rows[0]["time_update_s"] == 0.0
    assert rows[1]["time_update_s"] == 0.0  # no gradient step before learning_starts
    for name in TIME_COLUMNS:
        values = [r[name] for r in rows]
        assert values == sorted(values) and values[0] >= 0
    last = rows[-1]
    assert last["time_env_s"] > 0 and last["time_act_s"] > 0 and last["time_update_s"] > 0 and last["time_eval_s"] > 0
    phases = sum(last[name] for name in TIME_COLUMNS if name != "wall_time_s")
    assert phases <= last["wall_time_s"] + 0.01  # each cell is rounded to a millisecond


def test_epsilon_column_is_the_schedule_value_of_the_last_vec_step(variant_run):
    ac, tc = variant_run.cfg.agent, variant_run.cfg.train
    decay_steps = ac.eps_decay_fraction * tc.total_env_steps
    rows = read_metrics(variant_run.dir)
    assert rows[0]["epsilon"] == ac.eps_start
    for row in rows[1:]:
        steps_before = row["env_steps"] - tc.n_envs
        assert row["epsilon"] == ac.eps_start + (ac.eps_end - ac.eps_start) * min(1.0, steps_before / decay_steps)
    assert ac.eps_end < rows[1]["epsilon"] < ac.eps_start  # the first evaluation falls into the decay phase


def test_last_metrics_row_is_the_final_evaluation(tiny_run):
    last = read_metrics(tiny_run.dir)[-1]
    summary = EvalResult.load(tiny_run.dir / "final_eval.json").summary()
    assert last["eval_mean_score"] == summary["mean_score"]
    assert last["eval_ci95"] == summary["ci95"]
    assert last["eval_median_score"] == summary["median_score"]
    assert last["eval_max_score"] == summary["max_score"]
    assert last["eval_mean_steps"] == summary["mean_steps"]
    for cause in ("wall", "self", "starve", "win"):
        assert last[f"eval_frac_{cause}"] == summary["cause_fractions"][cause]


@pytest.mark.parametrize("which", ["tiny_run", "variant_run", "tiny_b_run", "noreplay_run"])
def test_final_eval_is_the_scalar_greedy_evaluation_of_the_saved_model(which, request):
    run = request.getfixturevalue(which)
    cfg, tc = run.cfg, run.cfg.train
    agent = DQNAgent.load(run.dir / "model.pt")
    assert agent.cfg == cfg.agent and agent.obs_shape == cfg.env.obs.shape(cfg.env.W) and agent.n_actions == 3
    stored = json.loads((run.dir / "final_eval.json").read_text(encoding="utf-8"))
    final = EvalResult.load(run.dir / "final_eval.json")
    assert [e.seed for e in final.episodes] == list(range(tc.eval_seed, tc.eval_seed + tc.eval_episodes))
    # One episode at a time on SnakeEnv, with the discount of the run (M7: the file carries the value fields,
    # which the scalar protocol reproduces to the rounding of a batched forward pass).
    scalar = evaluate(agent, cfg.env, tc.eval_episodes, tc.eval_seed, gamma=cfg.agent.gamma)
    assert old_fields(final.episodes) == old_fields(scalar.episodes)
    for got, want in zip(final.episodes, scalar.episodes):
        for name in VALUE_FIELDS:
            assert getattr(got, name) == pytest.approx(getattr(want, name), abs=1e-5), name
    for key in ("policy", "env", "seed", "git", "summary"):  # everything but the wall time and the values
        assert stored[key] == scalar.to_dict()[key]
    assert [{k: e[k] for k in OLD_RECORD_FIELDS} for e in stored["episodes"]] == [
        {k: e[k] for k in OLD_RECORD_FIELDS} for e in scalar.to_dict()["episodes"]]
    assert stored["policy"] == "dqn" and stored["git"] == FAKE_GIT
    assert not list(run.dir.glob("*.tmp"))  # written next to it, then renamed


@pytest.mark.parametrize("which", ["tiny_run", "variant_run", "tiny_b_run", "noreplay_run"])
def test_each_checkpoint_reproduces_the_evaluation_of_its_metrics_row(which, request):
    run = request.getfixturevalue(which)
    cfg, tc = run.cfg, run.cfg.train
    rows = read_metrics(run.dir)
    names = checkpoint_names(run.dir)
    assert names == [f"step_{row['env_steps']:08d}.pt" for row in rows]
    for name, row in zip(names, rows):
        path = run.dir / "checkpoints" / name
        payload = torch.load(path, weights_only=True)
        assert "optimizer" not in payload  # weights only
        agent = DQNAgent.load(path)
        assert agent.grad_steps == row["grad_steps"] and agent.cfg == cfg.agent
        result = evaluate_batched(agent, cfg.env, tc.eval_episodes, tc.eval_seed)
        assert result.mean_score == row["eval_mean_score"]
        for name, value in train_module.eval_columns(result).items():
            assert row[name] == value, (name, row["env_steps"])
    # The checkpoints are different networks, not one file four times: their tensors differ wherever
    # a gradient step lies between them ...
    files = [flat(run.dir / "checkpoints" / name) for name in names]
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            if rows[i]["grad_steps"] != rows[j]["grad_steps"]:
                assert files[i] != files[j], (names[i], names[j])
            else:
                assert files[i] == files[j], (names[i], names[j])  # nothing learned in between: the same network
    # ... and they play differently (the MLP runs change their play at every row; the tiny CNN, which
    # starves at every evaluation of its 2400-step budget, at least once).
    play = {(row["eval_mean_score"], row["eval_mean_steps"], row["eval_frac_starve"]) for row in rows}
    assert len(play) >= (2 if which == "tiny_b_run" else 3), play


def test_model_file_is_the_last_checkpoint_plus_the_optimizer(tiny_run):
    model = flat(tiny_run.dir / "model.pt")
    last = flat(tiny_run.dir / "checkpoints" / "step_00002400.pt")
    assert any(key.startswith("/optimizer/") for key in model)
    assert {key: value for key, value in model.items() if not key.startswith("/optimizer/")} == last
    first = flat(tiny_run.dir / "checkpoints" / "step_00000000.pt")
    assert first["/grad_steps"] == 0 and first["/q_net/net.0.weight"] != last["/q_net/net.0.weight"]
    # Row 0 is the network as initialised from the run seed.
    fresh = DQNAgent((11,), 3, tiny_run.cfg.agent, seed=TINY_SEED)
    fresh.save(tiny_run.runs / "fresh_init.pt", include_optimizer=False)
    assert flat(tiny_run.runs / "fresh_init.pt") == first


# ====================================================== M7: the value columns (D5 / D6)
def test_value_columns_sit_after_eval_frac_win_and_the_other_headers_are_unchanged():
    assert train_module.VALUE_COLUMNS == VALUE_COLUMNS
    assert train_module.EVAL_COLUMNS == (
        "eval_mean_score", "eval_ci95", "eval_median_score", "eval_max_score", "eval_mean_steps", "eval_frac_wall",
        "eval_frac_self", "eval_frac_starve", "eval_frac_win",
    )
    assert train_module.TABULAR_COLUMNS == tuple(TABULAR_HEADER.split(","))
    columns = train_module.DQN_COLUMNS
    i = columns.index("eval_frac_win")
    assert columns[i + 1 : i + 5] == VALUE_COLUMNS and columns[i + 5] == "train_episodes"
    assert ",".join(columns) == HEADER
    assert train_module.UPDATE_STATS == ("loss", "mean_q", "mean_target", "grad_norm")


@pytest.mark.parametrize("which", ["tiny_run", "variant_run", "tiny_b_run", "noreplay_run"])
def test_value_columns_of_every_row_are_reproduced_from_the_checkpoints(which, request):
    """D6: the four cells of row i are ``evaluate_batched(DQNAgent.load(checkpoint_i), cfg.env,
    eval_episodes, eval_seed, gamma).value_summary()`` (checkpoint <-> row, as for the score), exactly."""
    run = request.getfixturevalue(which)
    cfg, tc = run.cfg, run.cfg.train
    rows = read_metrics(run.dir)
    names = checkpoint_names(run.dir)
    assert names == [f"step_{row['env_steps']:08d}.pt" for row in rows]
    for name, row in zip(names, rows):
        agent = DQNAgent.load(run.dir / "checkpoints" / name)
        result = evaluate_batched(agent, cfg.env, tc.eval_episodes, tc.eval_seed, gamma=cfg.agent.gamma)
        values = result.value_summary()
        assert values is not None and set(values) == set(VALUE_FIELDS)
        for column, key in zip(VALUE_COLUMNS, VALUE_FIELDS):
            assert type(row[column]) is float and np.isfinite(row[column]), (column, row["env_steps"])
            assert row[column] == values[key], (column, row["env_steps"])
        assert result.mean_score == row["eval_mean_score"]
    # Rows that share a network (no gradient step in between) share the values. The estimate and the
    # realised return are different numbers: an untrained network's claim has nothing to do with what the
    # policy collects.
    for a, b in zip(rows, rows[1:]):
        if a["grad_steps"] == b["grad_steps"]:
            assert all(a[column] == b[column] for column in VALUE_COLUMNS)
    assert any(row["eval_value_visited"] != row["eval_return_visited"] for row in rows)
    assert any(row["eval_value_start"] != row["eval_return_start"] for row in rows)
    # A realised return of rewards in [-1, 1] lies within 1 / (1 - gamma).
    bound = 1.0 / (1.0 - cfg.agent.gamma)
    assert all(abs(row[column]) <= bound for row in rows for column in ("eval_return_start", "eval_return_visited"))


@pytest.mark.parametrize("which", ["tiny_run", "variant_run", "tiny_b_run", "noreplay_run"])
def test_final_eval_values_block_equals_the_last_metrics_row(which, request):
    run = request.getfixturevalue(which)
    last = read_metrics(run.dir)[-1]
    stored = json.loads((run.dir / "final_eval.json").read_text(encoding="utf-8"))
    assert stored["values"] == {key: last[column] for column, key in zip(VALUE_COLUMNS, VALUE_FIELDS)}
    assert list(stored["values"]) == list(VALUE_FIELDS)
    for e in stored["episodes"]:
        assert set(e) == {*OLD_RECORD_FIELDS, *VALUE_FIELDS}
        assert all(type(e[name]) is float for name in VALUE_FIELDS)
    final = EvalResult.load(run.dir / "final_eval.json")
    assert final.value_summary() == stored["values"]  # recomputed from the records, not read
    assert all(getattr(e, name) is not None for e in final.episodes for name in VALUE_FIELDS)


def test_the_dqn_trainer_evaluates_with_the_run_discount_and_the_tabular_trainer_without(tmp_path, monkeypatch):
    """D6: ``_evaluate_dqn`` calls ``evaluate_batched(agent, cfg.env, eval_episodes, eval_seed,
    gamma=ac.gamma)``; the tabular trainer passes no ``gamma``."""
    calls = []
    real = train_module.evaluate_batched

    def spy(policy, cfg, n_episodes=100, seed=evaluate_module.DEFAULT_EVAL_SEED, gamma=None, **kwargs):
        result = real(policy, cfg, n_episodes, seed, gamma=gamma, **kwargs)
        calls.append(SimpleNamespace(policy=policy, cfg=cfg, n_episodes=n_episodes, seed=seed, gamma=gamma, kwargs=kwargs,
                                     values=result.value_summary()))
        return result

    monkeypatch.setattr(train_module, "evaluate_batched", spy)
    cfg = load_config(SMOKE, [*TINY, "train.total_env_steps=1600", "agent.gamma=0.8"])
    train(cfg, 0, tmp_path / "dqn", quiet=True)
    assert len(calls) == 3 and all(c.gamma == 0.8 and c.kwargs == {} for c in calls)
    assert all(c.cfg == cfg.env and c.n_episodes == cfg.train.eval_episodes and c.seed == cfg.train.eval_seed for c in calls)
    assert all(isinstance(c.policy, DQNAgent) for c in calls)
    # The rows are filled from value_summary() of exactly these calls.
    rows = read_metrics(tmp_path / "dqn" / "tiny_dqn" / "0")
    assert len(rows) == 3
    for row, call in zip(rows, calls):
        assert {key: row[column] for column, key in zip(VALUE_COLUMNS, VALUE_FIELDS)} == call.values
    calls.clear()
    tabular = load_config(SMOKE_TABULAR, ["train.n_envs=8", "train.total_env_steps=800", "train.eval_every=800",
                                          "train.eval_episodes=4", "env.starve_limit=20"])
    train(tabular, 0, tmp_path / "tab", quiet=True)
    assert len(calls) == 2 and all(c.gamma is None and c.kwargs == {} for c in calls)


def test_a_tabular_run_has_no_value_columns_and_no_values_block(tmp_path):
    """D6: ``TABULAR_COLUMNS`` are unchanged and a tabular ``final_eval.json`` is the M5 file."""
    cfg = load_config(SMOKE_TABULAR, ["train.n_envs=8", "train.total_env_steps=800", "train.eval_every=800",
                                      "train.eval_episodes=4", "env.starve_limit=20"])
    run_dir = train(cfg, 0, tmp_path, quiet=True)
    lines = (run_dir / "metrics.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == TABULAR_HEADER and "value" not in lines[0] and len(lines) == 3
    stored = json.loads((run_dir / "final_eval.json").read_text(encoding="utf-8"))
    assert set(stored) == {"policy", "env", "seed", "git", "wall_time_s", "summary", "episodes"}
    assert all(set(e) == set(OLD_RECORD_FIELDS) for e in stored["episodes"])
    final = EvalResult.load(run_dir / "final_eval.json")
    assert final.value_summary() is None
    assert all(getattr(e, name) is None for e in final.episodes for name in VALUE_FIELDS)


# ============================================== M7: scripts/backfill_values.py (D7)
def run_backfill(*args: str, timeout: float = 300) -> subprocess.CompletedProcess:
    """The script in a fresh process, on one BLAS / OpenMP thread.

    The tiny runs are trained with ``agent.torch_threads = 1``; a multi-threaded
    convolution reduces in another order, so with torch's default thread count
    the CNN's max-Q values come out about 1e-8 off the trainer's (measured: the
    realised returns and the scores are unaffected, the ``eval_value_*`` cells
    differ in the 9th digit). One thread reproduces the trainer's numerics
    exactly, which is what the cell-by-cell comparison below is about.
    """
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    return subprocess.run([sys.executable, str(BACKFILL), *args], cwd=REPO_ROOT, capture_output=True, text=True,
                          timeout=timeout, env=env)


def read_values_csv(run_dir: Path) -> list[dict[str, str]]:
    """``values.csv`` as text cells (what the script wrote); checks the header and the LF line endings."""
    raw = (run_dir / "values.csv").read_bytes()
    assert b"\r" not in raw
    assert raw.decode("utf-8").splitlines()[0] == VALUES_HEADER
    with open(run_dir / "values.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def metrics_cells(run_dir: Path) -> list[dict[str, str]]:
    with open(run_dir / "metrics.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def names_path(line: str, path: Path) -> bool:
    return str(path) in line or path.as_posix() in line


@pytest.fixture(scope="module")
def backfilled(tiny_run, tiny_b_run, tmp_path_factory):
    """Copies of the tiny A and B runs (the fixtures' own directories are never touched), a run directory
    without ``checkpoints/`` next to the A copy, and one ``--force`` pass of the script over both experiments
    (``--force`` because a run trained after M7 already has the columns in its ``metrics.csv``, and the
    script's own job is a run trained before M7)."""
    root = tmp_path_factory.mktemp("backfill")
    a = root / "tiny_dqn" / str(TINY_SEED)
    b = root / "tiny_dqn_B" / str(TINY_SEED)
    shutil.copytree(tiny_run.dir, a)
    shutil.copytree(tiny_b_run.dir, b)
    no_checkpoints = root / "tiny_dqn" / "7"  # like a run with keep_checkpoints: false, or a tabular run
    no_checkpoints.mkdir()
    for name in ("config.yaml", "metrics.csv", "final_eval.json", "model.pt"):
        shutil.copy(tiny_run.dir / name, no_checkpoints / name)
    before = {path: tree(path) for path in (a, b, no_checkpoints)}
    done = run_backfill("--exp", str(a.parent), str(b.parent), "--device", "cpu", "--force")
    return SimpleNamespace(root=root, a=a, b=b, no_checkpoints=no_checkpoints, before=before, done=done)


def test_backfill_writes_values_csv_equal_to_the_metrics_rows(backfilled, tiny_run, tiny_b_run):
    done = backfilled.done
    assert done.returncode == 0, (done.stdout, done.stderr)
    columns = VALUES_HEADER.split(",")
    for run_dir, source in ((backfilled.a, tiny_run.dir), (backfilled.b, tiny_b_run.dir)):
        rows = read_values_csv(run_dir)
        metrics = metrics_cells(source)
        assert len(rows) == len(metrics) == 4
        for got, want in zip(rows, metrics):  # the same text: repr floats, plain ints
            assert got == {name: want[name] for name in columns}, run_dir
        assert [int(r["env_steps"]) for r in rows] == [0, 800, 1_600, 2_400]
        assert all(re.fullmatch(r"\d+", r["env_steps"]) for r in rows)
        # Nothing else of the run was touched.
        after = tree(run_dir)
        assert after.pop("values.csv") is not None
        assert after == backfilled.before[run_dir]
    # One line per run, saying how many checkpoints went into values.csv.
    lines = done.stdout.splitlines()
    for run_dir in (backfilled.a, backfilled.b):
        assert sum(names_path(line, run_dir) and "4 checkpoints -> values.csv" in line for line in lines) == 1, lines
    # The directory without checkpoints/ is skipped with a line and gets no values.csv.
    assert not (backfilled.no_checkpoints / "values.csv").exists()
    assert tree(backfilled.no_checkpoints) == backfilled.before[backfilled.no_checkpoints]
    assert any(names_path(line, backfilled.no_checkpoints) for line in lines), lines


def test_backfill_skips_a_run_whose_values_csv_exists_unless_forced(backfilled):
    a = backfilled.a
    kept = (a / "values.csv").read_bytes()
    stamp = (a / "values.csv").stat().st_mtime_ns
    done = run_backfill("--exp", str(a.parent), "--device", "cpu")
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert (a / "values.csv").read_bytes() == kept and (a / "values.csv").stat().st_mtime_ns == stamp
    lines = [line for line in done.stdout.splitlines() if names_path(line, a)]
    assert len(lines) == 1 and "values.csv" in lines[0] and "4 checkpoints -> values.csv" not in lines[0], done.stdout
    # --force on a fresh copy with a stale file, --quiet: the same file is written and nothing is printed.
    quiet = backfilled.root / "quiet" / "tiny_dqn" / str(TINY_SEED)
    shutil.copytree(a, quiet)
    (quiet / "values.csv").write_text("stale\n", encoding="utf-8")
    done = run_backfill("--exp", str(quiet.parent), "--device", "cpu", "--force", "--quiet")
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert done.stdout == ""
    assert (quiet / "values.csv").read_bytes() == kept


def test_backfill_reports_a_mismatch_with_exit_code_1_and_still_writes_values_csv(backfilled, tmp_path):
    """D7: a row whose ``eval_mean_score`` in metrics.csv is not what its checkpoint scores is printed and
    the exit code is 1 at the end; the run is still processed (values.csv holds the recomputed numbers)."""
    run_dir = tmp_path / "tiny_dqn" / str(TINY_SEED)
    shutil.copytree(backfilled.a, run_dir)
    (run_dir / "values.csv").unlink()
    with open(run_dir / "metrics.csv", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header, rows = next(reader), list(reader)
    i = header.index("eval_mean_score")
    rows[2][i] = repr(float(rows[2][i]) + 1.0)  # the row at 1600 env steps claims one food more
    with open(run_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    done = run_backfill("--exp", str(run_dir.parent), "--device", "cpu")
    assert done.returncode == 1, (done.stdout, done.stderr)
    assert "1600" in done.stdout + done.stderr
    assert read_values_csv(run_dir) == read_values_csv(backfilled.a)  # the checkpoint's own evaluation, not the edited cell


def test_backfill_help_and_missing_exp():
    done = run_backfill("--help")
    assert done.returncode == 0, done.stderr
    for flag in ("--exp", "--device", "--force", "--quiet"):
        assert flag in done.stdout, flag
    assert run_backfill().returncode != 0  # --exp is required


# ========================================================= M6: a run on observation B
def test_a_b_run_writes_the_run_files_and_trains_the_cnn(tiny_b_run):
    cfg, run_dir = tiny_b_run.cfg, tiny_b_run.dir
    assert cfg.env.obs == OBS_B8 and cfg.agent.conv_channels == (4, 4) and cfg.agent.hidden == (16,)
    assert run_dir == tiny_b_run.runs / "tiny_dqn_B" / str(TINY_SEED)
    assert sorted(p.name for p in run_dir.iterdir()) == RUN_ENTRIES
    assert checkpoint_names(run_dir) == ["step_00000000.pt", "step_00000800.pt", "step_00001600.pt",
                                         "step_00002400.pt"]
    recorded = read_run_config(run_dir)
    run = recorded.pop("run")
    assert recorded == config_to_dict(cfg)
    assert recorded["agent"]["conv_channels"] == [4, 4] and recorded["env"]["obs"]["kind"] == "B"
    assert run["device"] == "cpu" and run["torch_threads"] == 1 and run["env_seed"] == 100_000 * (TINY_SEED + 1)
    assert load_config(run_dir / "config.yaml") == cfg

    rows = read_metrics(run_dir)
    assert (run_dir / "metrics.csv").read_text(encoding="utf-8").splitlines()[0] == HEADER
    assert [r["env_steps"] for r in rows] == [0, 800, 1_600, 2_400]
    assert [r["grad_steps"] for r in rows] == [0, 0, 152, 352]  # 8 boards: 2 gradient steps per vec-step from 1000 on
    assert [r["buffer_size"] for r in rows] == [0, 800, 1_000, 1_000]
    for row in rows[:2]:
        assert all(row[name] is None for name in UPDATE_COLUMNS)
    for row in rows[2:]:
        assert all(np.isfinite(row[name]) for name in UPDATE_COLUMNS) and row["loss"] > 0 and row["grad_norm"] > 0
    assert all(rows[0][name] == rows[1][name] for name in train_module.EVAL_COLUMNS)

    agent = DQNAgent.load(run_dir / "model.pt")
    assert isinstance(agent.q_net, CNN) and agent.obs_shape == (8, 8, 8) and agent.grad_steps == 352
    assert agent.cfg == cfg.agent and agent.cfg.conv_channels == (4, 4)
    convs = [m for m in agent.q_net.modules() if isinstance(m, torch.nn.Conv2d)]
    assert [tuple(c.weight.shape) for c in convs] == [(4, 8, 3, 3), (4, 4, 3, 3)]
    linears = [m for m in agent.q_net.modules() if isinstance(m, torch.nn.Linear)]
    assert [tuple(m.weight.shape) for m in linears] == [(16, 4 * 64), (3, 16)]
    # The convolutions trained: the saved model differs from the initial network in every layer.
    first = flat(run_dir / "checkpoints" / "step_00000000.pt")
    last = flat(run_dir / "checkpoints" / "step_00002400.pt")
    assert first["/grad_steps"] == 0 and last["/grad_steps"] == 352
    for key in first:
        if key.startswith("/q_net/") and key.endswith(".weight"):
            assert first[key] != last[key], key
    # Row 0 is the CNN as initialised from the run seed.
    fresh = DQNAgent((8, 8, 8), 3, cfg.agent, seed=TINY_SEED)
    fresh.save(tiny_b_run.runs / "fresh_init.pt", include_optimizer=False)
    assert flat(tiny_b_run.runs / "fresh_init.pt") == first


def test_a_b_run_is_deterministic(tiny_b_run, tmp_path):
    again = train(tiny_b_run.cfg, tiny_b_run.seed, tmp_path / "again", quiet=True)
    assert_same_run(tiny_b_run.dir, again)
    other = train(tiny_b_run.cfg, tiny_b_run.seed + 1, tmp_path / "other", quiet=True)
    assert flat(other / "model.pt")["/grad_steps"] == 352
    assert final_episodes(other) != final_episodes(tiny_b_run.dir) or metrics_without_times(other) != metrics_without_times(tiny_b_run.dir)


def test_keep_checkpoints_false_writes_no_checkpoints_and_cli_prints_progress(tmp_path, capsys):
    argv = ["--config", str(SMOKE), "--seed", "5", "--runs-dir", str(tmp_path), "--set", *TINY,
            "agent.keep_checkpoints=false"]
    assert main(argv) == 0
    run_dir = tmp_path / "tiny_dqn" / "5"
    assert sorted(p.name for p in run_dir.iterdir()) == ["config.yaml", "final_eval.json", "metrics.csv", "model.pt"]
    assert read_run_config(run_dir)["run"]["argv"] == argv
    out = capsys.readouterr()
    assert out.err == ""
    progress = [line for line in out.out.splitlines() if " steps  eps " in line]
    assert len(progress) == 4
    rows = read_metrics(run_dir)
    for line, row, env_steps in zip(progress, rows, ["0/2,400", "800/2,400", "1,600/2,400", "2,400/2,400"]):
        assert line.startswith("[tiny_dqn/5] ") and f" {env_steps} steps  eps " in line
        assert line.isascii()  # safe on a Windows console
        assert f" eps {row['epsilon']:.3f}  eval {row['eval_mean_score']:.2f} +- {row['eval_ci95']:.2f}  " in line
        assert f"({row['train_episodes']} ep)" in line and f"{row['grad_steps']:,} updates" in line
    assert " loss -  q -  0 updates" in progress[1] and " loss 0." in progress[-1]
    assert out.out.splitlines()[0].startswith("[tiny_dqn/5] dqn -> ")


# ============================================================== determinism
def test_same_config_and_seed_give_the_same_run(tiny_run, tmp_path):
    again = train(tiny_run.cfg, tiny_run.seed, tmp_path / "again", quiet=True)
    assert again != tiny_run.dir
    assert_same_run(tiny_run.dir, again)


def test_a_different_seed_gives_a_different_run(tiny_run, tmp_path):
    other = train(tiny_run.cfg, tiny_run.seed + 1, tmp_path / "other", quiet=True)
    assert other.name == str(TINY_SEED + 1)
    a, b = flat(tiny_run.dir / "model.pt"), flat(other / "model.pt")
    assert a.keys() == b.keys()
    assert a["/q_net/net.0.weight"] != b["/q_net/net.0.weight"]
    # Already the initial networks differ: the weights are initialised from the run seed.
    assert (flat(tiny_run.dir / "checkpoints" / "step_00000000.pt")["/q_net/net.0.weight"]
            != flat(other / "checkpoints" / "step_00000000.pt")["/q_net/net.0.weight"])
    assert metrics_without_times(tiny_run.dir) != metrics_without_times(other)


def test_training_leaves_the_global_torch_generator_alone(tiny_run, tmp_path):
    torch.manual_seed(1234)
    before = torch.get_rng_state()
    train(tiny_run.cfg, 0, tmp_path, quiet=True)
    assert torch.equal(torch.get_rng_state(), before)


def test_fresh_process_reproduces_the_in_process_run(tiny_run, background):
    done = background("fresh")
    assert done.returncode == 0, done.stderr
    assert done.stdout == ""  # --quiet
    run_dir = done.runs / "tiny_dqn" / str(TINY_SEED)
    assert sorted(p.name for p in run_dir.iterdir()) == RUN_ENTRIES
    assert_same_run(tiny_run.dir, run_dir)
    run = read_run_config(run_dir)["run"]
    assert run["argv"] == done.args[2:]
    assert run["git"] == git_hash() != FAKE_GIT  # the real hash, taken by the child process
    assert run["device"] == "cpu" and run["torch_threads"] == 1
    assert load_config(run_dir / "config.yaml") == tiny_run.cfg


# ====================================================== the loop, pinned down
def reference_run(cfg: ExperimentConfig, seed: int, tmp_path: Path):
    """The DQN training protocol written a second time from the specification:
    ``VecSnakeEnv`` + ``DQNAgent`` + ``ReplayBuffer`` with the RNG protocol of
    the spec, without ``snake_rl.train``, ``env_seed_for``, ``linear_schedule``,
    ``act_epsilon`` or ``evaluate_batched``. Returns (model file, metrics rows,
    final evaluation)."""
    ac, tc = cfg.agent, cfg.train
    n = tc.n_envs
    torch.set_num_threads(ac.torch_threads)
    env = VecSnakeEnv(cfg.env, n_envs=n, seed=100_000 * (seed + 1))
    agent = DQNAgent(env.obs_shape, env.n_actions, ac, seed=seed)
    buffer = ReplayBuffer(ac.buffer_size, env.obs_shape)
    explore_rng = np.random.default_rng([seed, 1])
    sample_rng = np.random.default_rng([seed, 2])
    per_vec_step = ac.grad_steps_per_vec_step or max(1, n // 4)
    decay_steps = ac.eps_decay_fraction * tc.total_env_steps
    rows: list[dict] = []
    results: list[EvalResult] = []

    def evaluation(env_steps, epsilon, scores, n_truncated, stats):
        # Scalar env, one episode at a time, with the run's discount (M7: the value fields of the records).
        result = evaluate(agent, cfg.env, tc.eval_episodes, tc.eval_seed, gamma=ac.gamma)
        results.append(result)
        # M7 (D6): the four value columns come from the trainer's own protocol, so they are reproduced exactly.
        values = evaluate_batched(agent, cfg.env, tc.eval_episodes, tc.eval_seed, gamma=ac.gamma).value_summary()
        rows.append({
            "env_steps": env_steps,
            "vec_steps": env_steps // n,
            "grad_steps": agent.grad_steps,
            "epsilon": epsilon,
            "eval_mean_score": float(result.scores.mean()),
            "eval_max_score": int(result.scores.max()),
            "eval_mean_steps": float(result.steps.mean()),
            "eval_frac_starve": result.cause_fractions["starve"],
            **{f"eval_{name}": values[name] for name in VALUE_FIELDS},
            "train_episodes": len(scores),
            "train_mean_score": float(np.mean(scores)) if scores else None,
            "train_truncated": n_truncated,
            **{name: float(np.mean([s[name] for s in stats])) if stats else None for name in UPDATE_COLUMNS},
            "buffer_size": len(buffer),
        })

    obs = env.reset()
    evaluation(0, ac.eps_start, [], 0, [])
    scores, n_truncated, stats = [], 0, []
    for k in range(tc.total_env_steps // n):
        steps_before = k * n
        epsilon = ac.eps_start + (ac.eps_end - ac.eps_start) * min(1.0, steps_before / decay_steps)
        explore = explore_rng.random(n) < epsilon
        random_actions = explore_rng.integers(0, env.n_actions, size=n)
        actions = np.where(explore, random_actions, agent.act_batch(obs))
        next_obs, reward, terminated, truncated, info = env.step(actions)
        buffer.add(obs, actions, reward, info["final_obs"], terminated)
        obs = next_obs
        scores.extend(info["score"][info["done"]].tolist())
        n_truncated += int(truncated.sum())
        if steps_before + n >= ac.learning_starts:
            for _ in range(per_vec_step):
                stats.append(agent.update(buffer.sample(ac.batch_size, sample_rng)))
        if (steps_before + n) % tc.eval_every == 0:
            evaluation(steps_before + n, epsilon, scores, n_truncated, stats)
            scores, n_truncated, stats = [], 0, []
    return agent.save(tmp_path / "reference_model.pt"), rows, results[-1]


@pytest.mark.parametrize("which", ["variant_run", "tiny_run", "tiny_b_run"])
def test_independent_loop_reproduces_train_exactly(which, request, tmp_path):
    run = request.getfixturevalue(which)
    assert run.cfg.agent.replay
    model, rows, final = reference_run(run.cfg, run.seed, tmp_path)
    assert flat(run.dir / "model.pt") == flat(model)  # both networks, the optimiser state, grad_steps, the config
    assert old_fields(final_episodes(run.dir)) == old_fields(final.episodes)
    for got, want in zip(final_episodes(run.dir), final.episodes):  # M7: the scalar protocol's value fields, to rounding
        for name in VALUE_FIELDS:
            assert getattr(got, name) == pytest.approx(getattr(want, name), abs=1e-5), name

    written = read_metrics(run.dir)
    assert len(written) == len(rows) == 4
    for got, want in zip(written, rows):
        for name, value in want.items():
            if name in (*UPDATE_COLUMNS, "train_mean_score") and value is not None:
                assert got[name] == pytest.approx(value, rel=1e-9), name  # summation order is free
            else:
                assert got[name] == value, (name, want["env_steps"])  # the value columns too (M7)
    assert all(name in rows[0] for name in VALUE_COLUMNS)
    assert rows[-1]["grad_steps"] == expected_grad_steps(run.cfg, rows[-1]["vec_steps"]) > 300
    assert sum(r["train_episodes"] for r in rows) > 50


def test_variant_run_exercises_what_the_loop_tests_are_about(variant_run):
    cfg = variant_run.cfg
    rows = read_metrics(variant_run.dir)
    # 6 boards, 3 gradient steps per vec-step from vec-step 50 (300 env steps) on.
    assert [r["grad_steps"] for r in rows] == [0, 153, 453, 753]
    assert [r["buffer_size"] for r in rows] == [0, 500, 500, 500]  # the ring has wrapped
    assert cfg.agent.double and cfg.agent.target_update_every == 7
    model = flat(variant_run.dir / "model.pt")
    target_differs = any(model[key] != model[key.replace("/q_net/", "/target_net/")]
                         for key in model if key.startswith("/q_net/"))
    assert target_differs  # 753 is not a multiple of 7: the target network lags behind


def test_replay_stores_the_true_successor_and_the_terminated_flag_only(variant_run):
    cfg, steps, adds, observed = variant_run.cfg, variant_run.steps, variant_run.adds, variant_run.observed
    n = cfg.train.n_envs
    assert variant_run.envs == [(cfg.env, n, 100_000 * (VARIANT_SEED + 1))]  # one training env, seeded for this run
    assert len(steps) == len(adds) == cfg.train.total_env_steps // n
    truncated_total = terminated_total = moved_on = 0
    for k, (step, add) in enumerate(zip(steps, adds)):
        assert np.array_equal(add.obs, observed[k])  # the observation the action was chosen from
        assert np.array_equal(add.actions, step.actions)
        assert np.array_equal(add.rewards, step.reward)
        assert np.array_equal(add.next_obs, step.final_obs)  # true successor, not the post-reset obs
        assert add.terminated.dtype == bool
        assert np.array_equal(add.terminated, step.terminated)  # terminated only, never done
        assert not add.terminated[step.truncated].any()  # a truncated transition is stored as non-terminal
        truncated_total += int(step.truncated.sum())
        terminated_total += int(step.terminated.sum())
        ended = step.terminated | step.truncated
        moved_on += int((step.final_obs[ended] != observed[k + 1][ended]).any(axis=1).sum())
    # The run must contain what the assertions above are about.
    assert truncated_total > 20 and terminated_total > 20
    assert moved_on > 20  # final_obs and the returned obs really differ where an episode ended
    assert all((step.reward[step.truncated] > -1.0).all() for step in steps)  # starvation is not punished as a death


def test_train_truncated_counts_the_starved_training_episodes(variant_run):
    tc = variant_run.cfg.train
    rows = read_metrics(variant_run.dir)
    per_row = tc.eval_every // tc.n_envs
    assert rows[0]["train_truncated"] == 0
    for j, row in enumerate(rows[1:]):
        window = variant_run.steps[j * per_row:(j + 1) * per_row]
        assert row["train_truncated"] == sum(int(step.truncated.sum()) for step in window)
        assert row["train_episodes"] == sum(int((step.terminated | step.truncated).sum()) for step in window)
        assert 0 < row["train_truncated"] < row["train_episodes"]


# ================================================ M6: the loop without replay
def test_noreplay_run_feeds_each_vec_step_to_exactly_one_update(noreplay_run):
    """D4: no ``ReplayBuffer``; from ``learning_starts`` on, every vec-step is
    followed by one ``agent.update`` on a ``Batch`` of that vec-step's own
    arrays: the observation the actions were chosen from, the actions, the
    rewards, ``info["final_obs"]`` and ``terminated`` (never ``done``)."""
    cfg, steps, observed, updates = noreplay_run.cfg, noreplay_run.steps, noreplay_run.observed, noreplay_run.updates
    ac, n = cfg.agent, cfg.train.n_envs
    assert not ac.replay and ac.grad_steps_per_vec_step == 1
    assert noreplay_run.buffers == []  # no replay buffer was constructed
    assert noreplay_run.envs == [(cfg.env, n, 100_000 * (NOREPLAY_SEED + 1))]
    assert len(steps) == cfg.train.total_env_steps // n
    learning = [k for k in range(len(steps)) if (k + 1) * n >= ac.learning_starts]
    assert len(updates) == len(learning) == len(steps) - (-(-ac.learning_starts // n)) + 1 == 251
    truncated_total = terminated_total = moved_on = 0
    for j, (k, update) in enumerate(zip(learning, updates)):
        assert update.grad_steps == j  # one gradient step per vec-step, in order
        batch, step = update.batch, steps[k]
        assert batch.obs.shape == (n, 11) and batch.next_obs.shape == (n, 11)
        assert batch.actions.shape == batch.rewards.shape == batch.terminated.shape == (n,)
        assert np.array_equal(batch.obs, observed[k])  # the observation the action was chosen from
        assert np.array_equal(batch.actions, step.actions) and batch.actions.dtype == np.int64
        assert np.array_equal(batch.rewards, step.reward)
        assert np.array_equal(batch.next_obs, step.final_obs)  # true successor, not the post-reset obs
        assert batch.terminated.dtype == np.bool_ and np.array_equal(batch.terminated, step.terminated)
        assert not batch.terminated[step.truncated].any()  # a truncated transition is not terminal
        truncated_total += int(step.truncated.sum())
        terminated_total += int(step.terminated.sum())
        ended = step.terminated | step.truncated
        moved_on += int((step.final_obs[ended] != observed[k + 1][ended]).any(axis=1).sum())
    # The run must contain what the assertions above are about.
    assert truncated_total > 20 and terminated_total > 20 and moved_on > 20


def test_noreplay_metrics_columns(noreplay_run):
    """D4: ``grad_steps`` counts the vec-steps with ``env_steps >= learning_starts``,
    ``buffer_size`` is 0 in every row, the rest as with replay."""
    cfg = noreplay_run.cfg
    ac, n = cfg.agent, cfg.train.n_envs
    lines = (noreplay_run.dir / "metrics.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == HEADER
    rows = read_metrics(noreplay_run.dir)
    assert [r["env_steps"] for r in rows] == [0, 600, 1_200, 1_800]
    # 6 boards, learning from vec-step 50 (300 env steps) on: one gradient step per vec-step.
    assert [r["grad_steps"] for r in rows] == [0, 51, 151, 251]
    assert [r["grad_steps"] for r in rows] == [
        sum(1 for v in range(1, r["vec_steps"] + 1) if v * n >= ac.learning_starts) for r in rows]
    assert [r["grad_steps"] for r in rows] == [expected_grad_steps(cfg, r["vec_steps"]) for r in rows]
    assert [r["buffer_size"] for r in rows] == [0, 0, 0, 0]
    assert all(rows[0][name] is None for name in UPDATE_COLUMNS) and rows[0]["time_update_s"] == 0.0
    for row in rows[1:]:
        assert all(np.isfinite(row[name]) for name in UPDATE_COLUMNS) and row["loss"] > 0 and row["grad_norm"] > 0
        assert row["time_update_s"] > 0
    assert rows[0]["epsilon"] == ac.eps_start and ac.eps_end < rows[1]["epsilon"] < ac.eps_start
    assert sum(r["train_episodes"] for r in rows) > 50
    agent = DQNAgent.load(noreplay_run.dir / "model.pt")
    assert agent.grad_steps == 251 and agent.cfg == ac and agent.cfg.replay is False
    assert sorted(p.name for p in noreplay_run.dir.iterdir()) == RUN_ENTRIES
    assert checkpoint_names(noreplay_run.dir) == [f"step_{s:08d}.pt" for s in (0, 600, 1_200, 1_800)]
    assert load_config(noreplay_run.dir / "config.yaml") == cfg


def reference_noreplay_run(cfg: ExperimentConfig, seed: int, tmp_path: Path):
    """The no-replay loop of D4, written out from its pseudo-code: a ``Batch``
    of the vec-step's own arrays and one ``agent.update`` per vec-step from
    ``learning_starts`` on; ``sample_rng`` is created and never drawn. Returns
    (model file, metrics rows, final evaluation)."""
    ac, tc = cfg.agent, cfg.train
    n = tc.n_envs
    torch.set_num_threads(ac.torch_threads)
    env = VecSnakeEnv(cfg.env, n, seed=100_000 * (seed + 1))  # env_seed_for(seed, n_envs)
    obs = env.reset()
    agent = DQNAgent(env.obs_shape, env.n_actions, ac, seed=seed)
    explore_rng = np.random.default_rng([seed, 1])
    sample_rng = np.random.default_rng([seed, 2])  # created, never drawn
    untouched = np.random.default_rng([seed, 2]).bit_generator.state
    decay_steps = ac.eps_decay_fraction * tc.total_env_steps
    rows: list[dict] = []
    results: list[EvalResult] = []

    def evaluation(env_steps, epsilon, scores, n_truncated, stats):
        # Scalar env, one episode at a time, with the run's discount (M7); the value columns from the
        # trainer's own protocol (exact), as in ``reference_run``.
        result = evaluate(agent, cfg.env, tc.eval_episodes, tc.eval_seed, gamma=ac.gamma)
        results.append(result)
        values = evaluate_batched(agent, cfg.env, tc.eval_episodes, tc.eval_seed, gamma=ac.gamma).value_summary()
        rows.append({
            "env_steps": env_steps,
            "vec_steps": env_steps // n,
            "grad_steps": agent.grad_steps,
            "epsilon": epsilon,
            "eval_mean_score": float(result.scores.mean()),
            "eval_max_score": int(result.scores.max()),
            "eval_mean_steps": float(result.steps.mean()),
            "eval_frac_starve": result.cause_fractions["starve"],
            **{f"eval_{name}": values[name] for name in VALUE_FIELDS},
            "train_episodes": len(scores),
            "train_mean_score": float(np.mean(scores)) if scores else None,
            "train_truncated": n_truncated,
            **{name: float(np.mean([s[name] for s in stats])) if stats else None for name in UPDATE_COLUMNS},
            "buffer_size": 0,
        })

    evaluation(0, ac.eps_start, [], 0, [])  # row 0
    scores, n_truncated, stats = [], 0, []
    for k in range(tc.total_env_steps // n):
        epsilon = linear_schedule(k * n, ac.eps_start, ac.eps_end, decay_steps)
        actions = agent.act_epsilon(obs, epsilon, explore_rng)
        next_obs, reward, terminated, truncated, info = env.step(actions)
        env_steps = (k + 1) * n
        if env_steps >= ac.learning_starts:
            stats.append(agent.update(Batch(obs=obs, actions=actions, rewards=reward, next_obs=info["final_obs"],
                                            terminated=terminated)))  # ONE gradient step, this vec-step's transitions
        obs = next_obs
        scores.extend(info["score"][info["done"]].tolist())
        n_truncated += int(truncated.sum())
        if env_steps % tc.eval_every == 0:
            evaluation(env_steps, epsilon, scores, n_truncated, stats)
            scores, n_truncated, stats = [], 0, []
    assert sample_rng.bit_generator.state == untouched
    return agent.save(tmp_path / "reference_noreplay_model.pt"), rows, results[-1]


def test_independent_noreplay_loop_reproduces_train_exactly(noreplay_run, tmp_path):
    run = noreplay_run
    model, rows, final = reference_noreplay_run(run.cfg, run.seed, tmp_path)
    assert flat(run.dir / "model.pt") == flat(model)  # both networks, the optimiser state, grad_steps, the config
    assert old_fields(final_episodes(run.dir)) == old_fields(final.episodes)
    for got, want in zip(final_episodes(run.dir), final.episodes):  # M7: the scalar protocol's value fields, to rounding
        for name in VALUE_FIELDS:
            assert getattr(got, name) == pytest.approx(getattr(want, name), abs=1e-5), name

    written = read_metrics(run.dir)
    assert len(written) == len(rows) == 4
    for got, want in zip(written, rows):
        for name, value in want.items():
            if name in (*UPDATE_COLUMNS, "train_mean_score") and value is not None:
                assert got[name] == pytest.approx(value, rel=1e-9), name  # summation order is free
            else:
                assert got[name] == value, (name, want["env_steps"])
    assert rows[-1]["grad_steps"] == 251


def test_noreplay_run_is_deterministic_and_differs_from_the_replay_run(noreplay_run, tmp_path):
    again = train(noreplay_run.cfg, noreplay_run.seed, tmp_path / "again", quiet=True)
    assert_same_run(noreplay_run.dir, again)
    # The same configuration with a replay buffer (one gradient step per vec-step as well) is another run.
    with_replay = load_config(SMOKE, [*NOREPLAY, "agent.replay=true", "exp=replay_dqn"])
    assert with_replay.agent == dataclasses.replace(noreplay_run.cfg.agent, replay=True)
    other = train(with_replay, noreplay_run.seed, tmp_path / "replay", quiet=True)
    rows = read_metrics(other)
    assert [r["grad_steps"] for r in rows] == [0, 51, 151, 251]
    assert [r["buffer_size"] for r in rows] == [0, 500, 500, 500]
    a, b = flat(noreplay_run.dir / "model.pt"), flat(other / "model.pt")
    assert a["/q_net/net.0.weight"] != b["/q_net/net.0.weight"]
    assert a["/config/replay"] is False and b["/config/replay"] is True

    def network(path: Path) -> dict:
        return {key: value for key, value in flat(path).items() if key.startswith("/q_net/")}

    # The same initial network (same seed): the two runs part ways through the updates only.
    assert network(noreplay_run.dir / "checkpoints" / "step_00000000.pt") == network(other / "checkpoints" / "step_00000000.pt")
    assert network(noreplay_run.dir / "checkpoints" / "step_00000600.pt") != network(other / "checkpoints" / "step_00000600.pt")


# =========================================================== abort on NaN
def test_a_non_finite_update_aborts_the_run(tmp_path, monkeypatch, capsys):
    real_update = DQNAgent.update
    calls = []

    def failing_update(self, batch):
        calls.append(self.grad_steps)
        if len(calls) == 120:  # between the rows at 800 and 1600 env steps
            raise FloatingPointError("non-finite DQN update at gradient step 120: loss nan, gradient norm nan")
        return real_update(self, batch)

    monkeypatch.setattr(DQNAgent, "update", failing_update)
    argv = ["--config", str(SMOKE), "--seed", "0", "--runs-dir", str(tmp_path), "--quiet", "--set", *TINY]
    assert main(argv) == 3
    out = capsys.readouterr()
    assert out.out == "" and out.err.startswith("error: non-finite DQN update at gradient step 120")
    run_dir = tmp_path / "tiny_dqn" / "0"
    # Unfinished and recognisable as such: no final_eval.json, no model.pt; M6: the aborted.json marker.
    assert sorted(p.name for p in run_dir.iterdir()) == ["aborted.json", "checkpoints", "config.yaml", "metrics.csv"]
    rows = read_metrics(run_dir)  # closed and readable, with the rows written before the abort
    assert [r["env_steps"] for r in rows] == [0, 800]
    assert checkpoint_names(run_dir) == ["step_00000000.pt", "step_00000800.pt"]
    assert len(calls) == 120 and calls[-1] == 119
    # D4: the marker names the vec-step whose update failed and the completed gradient steps. TINY: 8
    # boards, 2 gradient steps per vec-step from vec-step 125 (1000 env steps) on, so call 120 is the
    # second update of vec-step 125 + 59 = 184 (1472 env steps), after 119 completed gradient steps.
    cfg = load_config(SMOKE, TINY)
    n = cfg.train.n_envs
    per_vec_step = max(1, n // 4)
    failing_vec_step = -(-cfg.agent.learning_starts // n) + (120 - 1) // per_vec_step
    assert failing_vec_step == 184 and 800 < failing_vec_step * n < 1600
    marker = json.loads((run_dir / "aborted.json").read_text(encoding="utf-8"))
    assert isinstance(marker, dict) and {"error", "env_steps", "vec_steps", "grad_steps"} <= set(marker)
    assert marker["error"] == "non-finite DQN update at gradient step 120: loss nan, gradient norm nan"
    assert marker["env_steps"] == failing_vec_step * n == 1472 and type(marker["env_steps"]) is int
    assert marker["vec_steps"] == failing_vec_step and type(marker["vec_steps"]) is int
    assert marker["grad_steps"] == 119 and type(marker["grad_steps"]) is int
    # The aborted run is not resumed or overwritten.
    before = tree(tmp_path)
    assert main(argv) == 1
    assert "never overwritten" in capsys.readouterr().err
    assert tree(tmp_path) == before

    # Through the Python API the error itself propagates.
    calls.clear()
    with pytest.raises(FloatingPointError, match="gradient step 120"):
        train(load_config(SMOKE, TINY), 1, tmp_path, quiet=True)
    assert not (tmp_path / "tiny_dqn" / "1" / "final_eval.json").exists()
    assert not (tmp_path / "tiny_dqn" / "1" / "model.pt").exists()
    with open(tmp_path / "tiny_dqn" / "1" / "metrics.csv", "a", encoding="utf-8"):
        pass  # the file handle was released
    marker = json.loads((tmp_path / "tiny_dqn" / "1" / "aborted.json").read_text(encoding="utf-8"))
    assert (marker["env_steps"], marker["vec_steps"], marker["grad_steps"]) == (1472, 184, 119)
    assert marker["error"].startswith("non-finite DQN update at gradient step 120")


def test_a_noreplay_abort_counts_vec_steps_as_gradient_steps(tmp_path, monkeypatch):
    """Without replay one update per vec-step: the marker of a failure at
    update 30 names vec-step 50 + 29 = 79 (474 env steps) and 29 completed steps."""
    real_update = DQNAgent.update
    calls = []

    def failing_update(self, batch):
        calls.append(self.grad_steps)
        if len(calls) == 30:
            raise FloatingPointError("non-finite DQN update at gradient step 30: loss inf, gradient norm inf")
        return real_update(self, batch)

    monkeypatch.setattr(DQNAgent, "update", failing_update)
    cfg = load_config(SMOKE, NOREPLAY)
    with pytest.raises(FloatingPointError, match="gradient step 30"):
        train(cfg, 0, tmp_path, quiet=True)
    run_dir = tmp_path / "noreplay_dqn" / "0"
    assert sorted(p.name for p in run_dir.iterdir()) == ["aborted.json", "checkpoints", "config.yaml", "metrics.csv"]
    marker = json.loads((run_dir / "aborted.json").read_text(encoding="utf-8"))
    assert marker["error"] == "non-finite DQN update at gradient step 30: loss inf, gradient norm inf"
    assert (marker["env_steps"], marker["vec_steps"], marker["grad_steps"]) == (79 * 6, 79, 29)
    assert [r["env_steps"] for r in read_metrics(run_dir)] == [0]


def test_a_directory_holding_only_aborted_json_is_refused_with_the_interrupted_run_hint(tmp_path):
    """D4: ``aborted.json`` is a run file; the refusal says how to repeat an
    interrupted run under the same name (move its directory away)."""
    cfg = load_config(SMOKE, TINY)
    run_dir = tmp_path / cfg.exp / "0"
    leave_behind(run_dir, ABORTED_FILE)
    before = tree(tmp_path)
    with pytest.raises(FileExistsError) as excinfo:
        train(cfg, 0, tmp_path, quiet=True)
    message = str(excinfo.value)
    assert ABORTED_FILE in message and str(run_dir) in message
    assert "never overwritten" in message and "--exp or --seed" in message
    assert "move" in message.lower()  # ... the directory out of its experiment directory
    assert tree(tmp_path) == before
    assert train_module.ABORTED_FILE == ABORTED_FILE and ABORTED_FILE in train_module.RUN_FILES


# ===================================================== runs are never touched
LEFTOVERS = ["config.yaml", "metrics.csv", "q_table.npz", "model.pt", "final_eval.json", "checkpoints/", "aborted.json"]


def leave_behind(run_dir: Path, leftover: str) -> None:
    run_dir.mkdir(parents=True)
    if leftover.endswith("/"):
        (run_dir / leftover).mkdir()
    else:
        (run_dir / leftover).write_text("left over from another run", encoding="utf-8")


@pytest.mark.parametrize("leftover", LEFTOVERS)
def test_a_directory_holding_a_run_file_of_either_kind_is_refused(leftover, tmp_path):
    cfg = load_config(SMOKE, TINY)
    leave_behind(tmp_path / cfg.exp / "0", leftover)
    before = tree(tmp_path)
    with pytest.raises(FileExistsError, match=leftover.rstrip("/")):
        train(cfg, 0, tmp_path, quiet=True)
    assert tree(tmp_path) == before


@pytest.mark.parametrize("leftover", ["model.pt", "checkpoints/"])
def test_a_tabular_run_refuses_a_directory_with_dqn_files(leftover, tmp_path):
    cfg = load_config(SMOKE_TABULAR, ["train.n_envs=8", "train.total_env_steps=800", "train.eval_every=800",
                                      "train.eval_episodes=4"])
    leave_behind(tmp_path / cfg.exp / "0", leftover)
    before = tree(tmp_path)
    with pytest.raises(FileExistsError, match=leftover.rstrip("/")):
        train(cfg, 0, tmp_path, quiet=True)
    assert tree(tmp_path) == before


def test_a_finished_dqn_run_is_refused_and_left_untouched(tiny_run):
    before = tree(tiny_run.runs)
    with pytest.raises(FileExistsError, match="tiny_dqn"):
        train(tiny_run.cfg, tiny_run.seed, tiny_run.runs, quiet=True)
    assert tree(tiny_run.runs) == before


def test_cuda_without_cuda_fails_before_anything_is_written(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="cuda"):
        train(load_config(SMOKE, [*TINY, "agent.device=cuda"]), 0, tmp_path, quiet=True)
    assert list(tmp_path.iterdir()) == []


# ====================================================================== cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_a_cuda_run_records_the_device_and_loads_on_the_cpu(tmp_path):
    cfg = load_config(SMOKE, [*TINY, "exp=tiny_cuda", "agent.device=cuda", "agent.grad_steps_per_vec_step=1"])
    run_dir = train(cfg, 0, tmp_path, quiet=True)
    assert sorted(p.name for p in run_dir.iterdir()) == RUN_ENTRIES
    run = read_run_config(run_dir)["run"]
    assert run["device"] == "cuda" and run["cuda_device"] == torch.cuda.get_device_name(0)
    rows = read_metrics(run_dir)
    assert [r["grad_steps"] for r in rows] == [0, 0, 76, 176]
    assert all(np.isfinite(rows[-1][name]) for name in UPDATE_COLUMNS) and rows[-1]["time_update_s"] > 0
    agent = DQNAgent.load(run_dir / "model.pt", device="cpu")  # a GPU run opens without a GPU
    assert agent.device.type == "cpu" and agent.grad_steps == 176
    assert all(not tensor.is_cuda for tensor in torch.load(run_dir / "model.pt", weights_only=True)["q_net"].values())


# ============================================== batch runner and acceptance
def test_run_matrix_runs_two_seeds_of_the_smoke_config(background):
    done = background("matrix")
    assert done.returncode == 0, done.stderr
    assert done.stderr == ""
    lines = done.stdout.splitlines()
    assert lines[0] == "2 runs, 2 to start, at most 2 at a time"
    assert lines[-1] == "2 runs: 2 ok"
    for seed in (0, 1):
        run_dir = done.runs / "smoke_dqn_W8" / str(seed)
        assert sorted(p.name for p in run_dir.iterdir()) == RUN_ENTRIES
        assert checkpoint_names(run_dir) == [f"step_{steps:08d}.pt" for steps in (0, 4_000, 8_000, 12_000, 16_000)]
        assert load_config(run_dir / "config.yaml") == SHIPPED["smoke_8x8.yaml"]
        run = read_run_config(run_dir)["run"]
        assert run["seed"] == seed and run["device"] == "cpu" and run["torch_threads"] == 1
        rows = read_metrics(run_dir)
        assert [r["env_steps"] for r in rows] == [0, 4_000, 8_000, 12_000, 16_000]
        # 16 boards -> 4 gradient steps per vec-step from vec-step 64 (1024 env steps) on.
        assert rows[-1]["grad_steps"] == 4 * (1_000 - 63) == 3_748
        assert rows[-1]["buffer_size"] == 10_000
    assert flat(done.runs / "smoke_dqn_W8" / "0" / "model.pt") != flat(done.runs / "smoke_dqn_W8" / "1" / "model.pt")


def test_shipped_smoke_config_ends_far_above_the_random_player(background):
    """M5 acceptance at the level of the entry point: the smoke budget (16,000
    env steps, 3,748 gradient steps) already beats the random player by a wide
    margin under the same evaluation protocol. Measured on the development
    machine: final mean score 17.20 (seed 0) and 15.05 (seed 1) over 20
    episodes, random player 0.15 +- 0.16; the threshold of 8 leaves more than
    30 % slack below the lower of the two."""
    done = background("matrix")
    assert done.returncode == 0, done.stderr
    cfg = load_config(SMOKE)
    random_player = evaluate(make_policy("random", seed=cfg.train.eval_seed), cfg.env, cfg.train.eval_episodes,
                             cfg.train.eval_seed)
    assert random_player.mean_score < 1.0
    for seed in (0, 1):
        run_dir = done.runs / "smoke_dqn_W8" / str(seed)
        final = EvalResult.load(run_dir / "final_eval.json")
        assert [e.seed for e in final.episodes] == [e.seed for e in random_player.episodes]  # the same boards
        assert final.mean_score > 8.0
        assert final.mean_score - final.ci95 > 5 * (random_player.mean_score + random_player.ci95)
        curve = [row["eval_mean_score"] for row in read_metrics(run_dir)]
        assert len(curve) == 5 and curve[0] < 1.0 and curve[-1] == final.mean_score


def test_shipped_smoke_b_config_runs_in_a_fresh_process(background):
    """M6 (D6): the observation-B smoke config, as shipped, in a fresh process:
    exit code 0, every file of a DQN run directory, the CNN of the file, a
    learning signal in the update statistics, and a final evaluation on the
    boards of the random player.

    The bar is lower than the A smoke test's (final mean score > 8): the CNN
    needs far more gradient steps than the MLP before the greedy policy eats.
    The first draft of the file (two 8-channel layers, 32 hidden units,
    16,000 steps) scored 0.15 / 0.30 / 0.15 on seeds 0 / 1 / 2 against a
    random player at 0.15 +- 0.16, i.e. nothing, and 1.35 / 0.90 at 64,000
    steps; the shipped file (16 / 32 channels, 128 hidden units, 48,000 steps)
    scores 9.65 on seed 0 and 13.70 on seed 1 (M6 measurement). The bar of 5
    is far above random and half of the lower measurement. The action values
    must also show learning: mean_q rises from the first learning row to the
    last. The loss is no bar: it falls while the targets are near zero and
    rises again once the policy starts to eat."""
    done = background("smoke_b")
    assert done.returncode == 0, done.stderr
    assert done.stdout == ""  # --quiet
    cfg = load_config(SMOKE_B)  # pinned to D6 by test_shipped_dqn_config_files_load; the rows follow the file
    ac, tc = cfg.agent, cfg.train
    assert cfg.env.obs == OBS_B8 and cfg.env.W == 8 and ac.device == "cpu" and ac.torch_threads == 1
    steps = [i * tc.eval_every for i in range(tc.total_env_steps // tc.eval_every + 1)]
    run_dir = done.runs / "smoke_dqn_B8_W8" / "0"
    assert sorted(p.name for p in run_dir.iterdir()) == RUN_ENTRIES
    assert checkpoint_names(run_dir) == [f"step_{s:08d}.pt" for s in steps]
    assert load_config(run_dir / "config.yaml") == cfg
    run = read_run_config(run_dir)["run"]
    assert run["seed"] == 0 and run["device"] == "cpu" and run["torch_threads"] == 1 and run["argv"] == done.args[2:]
    rows = read_metrics(run_dir)
    assert [r["env_steps"] for r in rows] == steps
    assert [r["grad_steps"] for r in rows] == [expected_grad_steps(cfg, s // tc.n_envs) for s in steps]
    assert rows[-1]["grad_steps"] > 11_000  # 11,748 at the shipped budget: 4 per vec-step from vec-step 64 on
    assert [r["buffer_size"] for r in rows] == [min(s, ac.buffer_size) for s in steps]
    learning = [row for row in rows if row["grad_steps"] > 0]
    assert len(learning) >= 2
    for row in learning:
        assert all(np.isfinite(row[name]) for name in UPDATE_COLUMNS) and row["loss"] > 0 and row["grad_norm"] > 0
    assert rows[-1]["mean_q"] - learning[0]["mean_q"] > 0.08  # the action values grow towards the food return
    assert rows[-1]["time_update_s"] > 0 and rows[-1]["time_eval_s"] > 0

    agent = DQNAgent.load(run_dir / "model.pt")
    assert isinstance(agent.q_net, CNN) and agent.obs_shape == (8, 8, 8) and agent.grad_steps == rows[-1]["grad_steps"]
    assert agent.cfg == ac
    convs = [m for m in agent.q_net.modules() if isinstance(m, torch.nn.Conv2d)]
    assert [tuple(c.weight.shape) for c in convs] == [(c, previous, 3, 3) for c, previous in
                                                      zip(ac.conv_channels, (8, *ac.conv_channels))]
    final = EvalResult.load(run_dir / "final_eval.json")
    assert final.policy == "dqn" and len(final.episodes) == cfg.train.eval_episodes
    # The trainer's own protocol with the run's discount (M7): the records, value fields included, are reproduced.
    assert evaluate_batched(agent, cfg.env, cfg.train.eval_episodes, cfg.train.eval_seed, gamma=ac.gamma).episodes == final.episodes
    random_player = evaluate(make_policy("random", seed=cfg.train.eval_seed), cfg.env, cfg.train.eval_episodes,
                             cfg.train.eval_seed)
    assert random_player.mean_score < 1.0
    assert [e.seed for e in final.episodes] == [e.seed for e in random_player.episodes]  # the same boards
    assert final.mean_score > 5.0  # measured 9.65 (seed 0); random 0.15
    assert final.mean_score - final.ci95 > 5 * (random_player.mean_score + random_player.ci95)
    curve = [row["eval_mean_score"] for row in read_metrics(run_dir)]
    assert len(curve) == 5 and curve[0] < 1.0 and curve[-1] == final.mean_score

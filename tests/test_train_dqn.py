"""The DQN training path (PLAN.md milestone M5): the ``dqn`` agent kind of
``snake_rl.config``, the trainer ``_train_dqn`` of ``snake_rl.train`` and the
three shipped DQN configuration files.

Every run of this file goes to a pytest temporary directory, on the CPU with
one torch thread and tiny networks and budgets. The subprocess jobs (a fresh
training process and ``scripts/run_matrix.py`` on the smoke config) are started
when the module starts, so they run while the in-process tests do.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

import snake_rl.evaluate as evaluate_module
import snake_rl.train as train_module
from snake_rl.agents.baselines import make_policy
from snake_rl.agents.dqn import DQNConfig, ReplayBuffer
from snake_rl.agents.dqn.agent import DQNAgent
from snake_rl.agents.tabular_q import TabularQConfig
from snake_rl.config import ExperimentConfig, TrainConfig, config_from_dict, config_to_dict, load_config
from snake_rl.env import EnvConfig, ObsConfig, RewardConfig, VecSnakeEnv
from snake_rl.evaluate import EvalResult, evaluate, evaluate_batched
from snake_rl.train import main, train
from snake_rl.utils import git_hash

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"
SMOKE = CONFIGS / "smoke_8x8.yaml"
SMOKE_TABULAR = CONFIGS / "smoke_tabular.yaml"

HEADER = (
    "env_steps,vec_steps,grad_steps,epsilon,eval_mean_score,eval_ci95,eval_median_score,eval_max_score,"
    "eval_mean_steps,eval_frac_wall,eval_frac_self,eval_frac_starve,eval_frac_win,train_episodes,train_mean_score,"
    "train_truncated,loss,mean_q,mean_target,grad_norm,buffer_size,time_env_s,time_act_s,time_update_s,time_eval_s,"
    "wall_time_s"
)
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
        "agent": {"kind": "dqn", "gamma": 0.9, "lr": 2.5e-4, "batch_size": 16, "buffer_size": 1000,
                  "learning_starts": 400, "grad_steps_per_vec_step": None, "target_update_every": 50,
                  "grad_clip": 10.0, "eps_start": 1.0, "eps_end": 0.05, "eps_decay_fraction": 0.2,
                  "hidden": [16, 16], "double": False, "keep_checkpoints": True, "device": "cpu",
                  "torch_threads": 1},
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
    """G gradient steps after every vec-step that ends at or after ``learning_starts``."""
    ac, n = cfg.agent, cfg.train.n_envs
    per_vec_step = ac.grad_steps_per_vec_step or max(1, n // 4)
    return per_vec_step * sum(1 for v in range(1, vec_steps + 1) if v * n >= ac.learning_starts)


# =================================================================== config
def expected_config(exp: str, W: int, agent: DQNConfig, train_cfg: TrainConfig) -> ExperimentConfig:
    return ExperimentConfig(
        exp=exp,
        env=EnvConfig(W=W, action_mode="relative", starve_limit=None, obs=ObsConfig(kind="A"),
                      reward=RewardConfig(variant="R0")),
        agent_kind="dqn",
        agent=agent,
        train=train_cfg,
    )


SHIPPED = {
    # gamma 0.9: the value of the tabular reference (M4 sweep). cpu with 2 threads: measured about three
    # times faster per gradient step than cuda on this MLP, and bit-for-bit reproducible. eval_every
    # 12,800: the greedy score plateaus within the first 50,000 steps. The rest is the DQNConfig default.
    "dqn_A_10x10.yaml": expected_config("dqn_A_W10", 10, DQNConfig(gamma=0.9, device="cpu", torch_threads=2),
                                        TrainConfig(128, 1_024_000, 12_800, 100, 10_000)),
    "dqn_A_8x8.yaml": expected_config("dqn_A_W8", 8, DQNConfig(gamma=0.9, device="cpu", torch_threads=2),
                                      TrainConfig(128, 512_000, 12_800, 100, 10_000)),
    "smoke_8x8.yaml": expected_config(
        "smoke_dqn_W8", 8,
        DQNConfig(gamma=0.9, batch_size=64, buffer_size=10_000, learning_starts=1024, target_update_every=250,
                  hidden=(64, 64), device="cpu", torch_threads=1),
        TrainConfig(16, 16_000, 4_000, 20, 10_000),
    ),
}


def test_dqn_config_defaults():
    assert DQNConfig() == DQNConfig(
        gamma=0.99, lr=2.5e-4, batch_size=256, buffer_size=100_000, learning_starts=10_000,
        grad_steps_per_vec_step=None, target_update_every=1_000, grad_clip=10.0, eps_start=1.0, eps_end=0.05,
        eps_decay_fraction=0.2, hidden=(256, 256), double=False, keep_checkpoints=True, device="auto",
        torch_threads=None,
    )
    # The defaults pass the validation (with a budget that reaches learning_starts).
    cfg = ExperimentConfig("d", EnvConfig(), "dqn", DQNConfig(), TrainConfig())
    assert config_from_dict(config_to_dict(cfg)) == cfg


@pytest.mark.parametrize("name", sorted(SHIPPED))
def test_shipped_dqn_config_files_load(name):
    cfg = load_config(CONFIGS / name)
    assert cfg == SHIPPED[name]
    assert type(cfg.agent) is DQNConfig and type(cfg.agent.hidden) is tuple
    # Every field is written out in the file, none is left to a default.
    written = yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))
    assert set(written["agent"]) == {"kind", *(f.name for f in dataclasses.fields(DQNConfig))}
    assert set(written["train"]) == {f.name for f in dataclasses.fields(TrainConfig)}


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
                 "agent.eps_decay_fraction")
FLAGS = ("agent.double", "agent.keep_checkpoints")

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
    ({"agent.hidden": 64}, "agent.hidden"),
    ({"agent.hidden": "64"}, "agent.hidden"),
    ({"agent.hidden": None}, "agent.hidden"),
    ({"agent.hidden": [64, 0]}, "agent.hidden"),
    ({"agent.hidden": [-8]}, "agent.hidden"),
    ({"agent.hidden": [64.0]}, "agent.hidden"),
    ({"agent.hidden": [True]}, "agent.hidden"),
    ({"agent.hidden": [[64]]}, "agent.hidden"),
    ({"agent.hidden": {"a": 64}}, "agent.hidden"),
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
    # ---- combinations
    ({"env.obs.kind": "B"}, "env.obs.kind"),  # the CNN for observation B arrives in M6
    ({"env.reward.variant": "R2"}, "env.reward.gamma"),  # 0.99 against agent.gamma 0.9
    ({"env.reward.variant": "R2"}, "agent.gamma"),
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


def test_observation_b_is_refused_with_the_milestone_that_brings_it(tmp_path):
    message = config_error(write_yaml(tmp_path / "c.yaml", edited({"env.obs.kind": "B"})))
    assert "M6" in message and "dqn" in message


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
]


@pytest.mark.parametrize("case", VALID, ids=lambda case: ",".join(f"{k}={v!r}" for k, v in case[0].items()))
def test_dqn_boundary_values_are_accepted(case, tmp_path):
    changes, check = case
    cfg = load_config(write_yaml(tmp_path / "c.yaml", edited(changes)))
    assert check(cfg)
    assert type(cfg.agent.batch_size) is int and type(cfg.agent.hidden) is tuple


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


def test_dqn_config_to_dict_spells_out_every_field_with_hidden_as_a_list():
    d = config_to_dict(load_config(SMOKE))
    assert d["agent"] == {
        "kind": "dqn", "gamma": 0.9, "lr": 2.5e-4, "batch_size": 64, "buffer_size": 10000, "learning_starts": 1024,
        "grad_steps_per_vec_step": None, "target_update_every": 250, "grad_clip": 10.0, "eps_start": 1.0,
        "eps_end": 0.05, "eps_decay_fraction": 0.2, "hidden": [64, 64], "double": False, "keep_checkpoints": True,
        "device": "cpu", "torch_threads": 1,
    }
    assert type(d["agent"]["hidden"]) is list
    assert list(d) == ["exp", "env", "agent", "train"]
    assert list(d["agent"]) == ["kind", *(f.name for f in dataclasses.fields(DQNConfig))]


ROUND_TRIP = [
    (SMOKE, ()),
    (CONFIGS / "dqn_A_8x8.yaml", ()),
    (CONFIGS / "dqn_A_10x10.yaml", ()),
    (SMOKE, TINY),
    (SMOKE, VARIANT),
    (SMOKE, ("agent.hidden=[]", "env.reward.variant=R2", "env.reward.gamma=0.9", "agent.lr=1.2345678901234567e-4")),
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


@pytest.mark.parametrize("which", ["tiny_run", "variant_run"])
def test_final_eval_is_the_scalar_greedy_evaluation_of_the_saved_model(which, request):
    run = request.getfixturevalue(which)
    cfg, tc = run.cfg, run.cfg.train
    agent = DQNAgent.load(run.dir / "model.pt")
    assert agent.cfg == cfg.agent and agent.obs_shape == (11,) and agent.n_actions == 3
    stored = json.loads((run.dir / "final_eval.json").read_text(encoding="utf-8"))
    final = EvalResult.load(run.dir / "final_eval.json")
    assert [e.seed for e in final.episodes] == list(range(tc.eval_seed, tc.eval_seed + tc.eval_episodes))
    scalar = evaluate(agent, cfg.env, tc.eval_episodes, tc.eval_seed)  # one episode at a time, SnakeEnv
    assert final.episodes == scalar.episodes
    for key in ("policy", "env", "seed", "git", "summary", "episodes"):  # everything but the wall time
        assert stored[key] == scalar.to_dict()[key]
    assert stored["policy"] == "dqn" and stored["git"] == FAKE_GIT
    assert not list(run.dir.glob("*.tmp"))  # written next to it, then renamed


@pytest.mark.parametrize("which", ["tiny_run", "variant_run"])
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
    # The checkpoints are different networks, not one file four times: they play differently.
    assert len({(row["eval_mean_score"], row["eval_mean_steps"], row["eval_frac_starve"]) for row in rows}) >= 3


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
        result = evaluate(agent, cfg.env, tc.eval_episodes, tc.eval_seed)  # scalar env, one episode at a time
        results.append(result)
        rows.append({
            "env_steps": env_steps,
            "vec_steps": env_steps // n,
            "grad_steps": agent.grad_steps,
            "epsilon": epsilon,
            "eval_mean_score": float(result.scores.mean()),
            "eval_max_score": int(result.scores.max()),
            "eval_mean_steps": float(result.steps.mean()),
            "eval_frac_starve": result.cause_fractions["starve"],
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


@pytest.mark.parametrize("which", ["variant_run", "tiny_run"])
def test_independent_loop_reproduces_train_exactly(which, request, tmp_path):
    run = request.getfixturevalue(which)
    model, rows, final = reference_run(run.cfg, run.seed, tmp_path)
    assert flat(run.dir / "model.pt") == flat(model)  # both networks, the optimiser state, grad_steps, the config
    assert final_episodes(run.dir) == final.episodes

    written = read_metrics(run.dir)
    assert len(written) == len(rows) == 4
    for got, want in zip(written, rows):
        for name, value in want.items():
            if name in (*UPDATE_COLUMNS, "train_mean_score") and value is not None:
                assert got[name] == pytest.approx(value, rel=1e-9), name  # summation order is free
            else:
                assert got[name] == value, (name, want["env_steps"])
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
    # Unfinished and recognisable as such: no final_eval.json, no model.pt.
    assert sorted(p.name for p in run_dir.iterdir()) == ["checkpoints", "config.yaml", "metrics.csv"]
    rows = read_metrics(run_dir)  # closed and readable, with the rows written before the abort
    assert [r["env_steps"] for r in rows] == [0, 800]
    assert checkpoint_names(run_dir) == ["step_00000000.pt", "step_00000800.pt"]
    assert len(calls) == 120
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
    with open(tmp_path / "tiny_dqn" / "1" / "metrics.csv", "a", encoding="utf-8"):
        pass  # the file handle was released


# ===================================================== runs are never touched
LEFTOVERS = ["config.yaml", "metrics.csv", "q_table.npz", "model.pt", "final_eval.json", "checkpoints/"]


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

"""Experiment configuration (``snake_rl.config``) and the training entry point
(``snake_rl.train``), PLAN.md milestone M4.

Every run of this file goes to a pytest temporary directory; the working
directory of each test is a temporary directory too, so even a run started
with the default ``runs_dir`` could not reach the repository's ``runs/``.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import platform
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import snake_rl.evaluate as evaluate_module
import snake_rl.train as train_module
from snake_rl.agents.baselines import make_policy
from snake_rl.agents.tabular_q import TabularQAgent, TabularQConfig
from snake_rl.config import (
    ExperimentConfig,
    TrainConfig,
    config_from_dict,
    config_to_dict,
    env_seed_for,
    load_config,
)
from snake_rl.env import EnvConfig, ObsConfig, RewardConfig, VecSnakeEnv
from snake_rl.evaluate import DEFAULT_EVAL_SEED, EvalResult, evaluate, evaluate_batched
from snake_rl.train import MetricsWriter, main, train
from snake_rl.utils import git_hash

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"
SMOKE = CONFIGS / "smoke_tabular.yaml"
BOARD_8 = CONFIGS / "tabular_q_8x8.yaml"
BOARD_10 = CONFIGS / "tabular_q_10x10.yaml"

HEADER = (
    "env_steps,vec_steps,epsilon,eval_mean_score,eval_ci95,eval_median_score,eval_max_score,eval_mean_steps,"
    "eval_frac_wall,eval_frac_self,eval_frac_starve,eval_frac_win,train_episodes,train_mean_score,mean_abs_td,"
    "cells_visited,wall_time_s"
)
INT_COLUMNS = ("env_steps", "vec_steps", "eval_max_score", "train_episodes", "cells_visited")
RUN_FILES = ["config.yaml", "final_eval.json", "metrics.csv", "q_table.npz"]
FAKE_GIT = "testhash"

#: The smoke config shrunk to a tenth of a second: 300 vec-steps, 4 evaluations
#: of 6 episodes, and a starvation limit that keeps a looping policy short.
TINY = ("train.n_envs=8", "train.total_env_steps=2400", "train.eval_every=800", "train.eval_episodes=6",
        "env.starve_limit=20")

#: Small boards with a short starvation limit, so that truncations are frequent,
#: and schedules whose decay phase contains several evaluations.
VARIANTS = {
    "relative_R0": (
        2,
        ("env.W=6", "env.starve_limit=8", "train.n_envs=8", "train.total_env_steps=9600", "train.eval_every=1600",
         "train.eval_episodes=10", "train.eval_seed=777", "agent.alpha=0.5", "agent.gamma=0.9",
         "agent.eps_end=0.1", "agent.eps_decay_fraction=0.5"),
    ),
    "absolute_R3": (
        0,
        ("env.W=5", "env.action_mode=absolute", "env.reward.variant=R3", "env.starve_limit=6", "train.n_envs=5",
         "train.total_env_steps=6000", "train.eval_every=1000", "train.eval_episodes=8", "train.eval_seed=31",
         "agent.alpha=0.3", "agent.gamma=0.8", "agent.eps_start=0.8", "agent.eps_end=0.2",
         "agent.eps_decay_fraction=1.0", "agent.q_init=0.5"),
    ),
}


# ----------------------------------------------------------------- fixtures
@pytest.fixture(scope="module", autouse=True)
def fast_git_hash():
    """``git_hash()`` starts two git processes (about 50 ms) and runs once per
    evaluation; a constant keeps the file fast. The fresh-process test checks
    the real hash."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train_module, "git_hash", lambda: FAKE_GIT)
        patch.setattr(evaluate_module, "git_hash", lambda: FAKE_GIT)
        yield


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path, monkeypatch):
    """A relative ``runs`` directory can only ever land in ``tmp_path``."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    """The shipped smoke config trained once, in process, seed 0."""
    runs = tmp_path_factory.mktemp("smoke_runs")
    cfg = load_config(SMOKE)
    argv = ["--config", "configs/smoke_tabular.yaml", "--seed", "0"]
    run_dir = train(cfg, 0, runs, argv=argv, quiet=True)
    return SimpleNamespace(cfg=cfg, seed=0, runs=runs, dir=run_dir, argv=argv)


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    runs = tmp_path_factory.mktemp("tiny_runs")
    cfg = load_config(SMOKE, TINY)
    return SimpleNamespace(cfg=cfg, seed=3, runs=runs, dir=train(cfg, 3, runs, quiet=True))


@pytest.fixture(scope="module", params=sorted(VARIANTS))
def variant_run(request, tmp_path_factory):
    seed, overrides = VARIANTS[request.param]
    runs = tmp_path_factory.mktemp(f"variant_{request.param}")
    cfg = load_config(SMOKE, [*overrides, f"exp={request.param}"])
    return SimpleNamespace(cfg=cfg, seed=seed, runs=runs, dir=train(cfg, seed, runs, quiet=True))


# ------------------------------------------------------------------ helpers
def valid_dict() -> dict:
    """A complete, valid configuration written out by hand (YAML layout)."""
    return {
        "exp": "unit",
        "env": {
            "W": 8,
            "action_mode": "relative",
            "starve_limit": None,
            "obs": {"kind": "A", "age_plane": True, "heading_planes": True},
            "reward": {"variant": "R0", "food": 1.0, "death": -1.0, "step_penalty": 0.01, "gamma": 0.99,
                       "closer": 0.01, "farther": 0.0},
        },
        "agent": {"kind": "tabular_q", "alpha": 0.1, "gamma": 0.99, "eps_start": 1.0, "eps_end": 0.05,
                  "eps_decay_fraction": 0.2, "q_init": 0.0},
        "train": {"n_envs": 16, "total_env_steps": 32000, "eval_every": 8000, "eval_episodes": 20,
                  "eval_seed": 10000},
    }


DELETE = object()


def edited(changes: dict) -> dict:
    """``valid_dict()`` with ``{dotted.key: value}`` applied (``DELETE`` removes the key)."""
    d = valid_dict()
    for dotted, value in changes.items():
        *parents, leaf = dotted.split(".")
        node = d
        for name in parents:
            node = node[name]
        if value is DELETE:
            del node[leaf]
        else:
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
    assert message.startswith(f"{path}: "), message  # the error says which file
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


def metrics_without_wall_time(run_dir: Path) -> list[str]:
    """The lines of ``metrics.csv`` with the last column (``wall_time_s``) cut off."""
    lines = (run_dir / "metrics.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0].endswith(",wall_time_s")
    return [line.rsplit(",", 1)[0] for line in lines]


def load_table(run_dir: Path) -> TabularQAgent:
    return TabularQAgent.load(run_dir / "q_table.npz")


def final_episodes(run_dir: Path) -> list:
    return EvalResult.load(run_dir / "final_eval.json").episodes


def assert_same_run(a: Path, b: Path) -> None:
    """Identical tables, final-evaluation episodes and metrics (except wall time)."""
    ta, tb = load_table(a), load_table(b)
    assert ta.q.tobytes() == tb.q.tobytes()
    assert ta.visits.tobytes() == tb.visits.tobytes()
    assert final_episodes(a) == final_episodes(b)
    assert metrics_without_wall_time(a) == metrics_without_wall_time(b)


def snapshot(directory: Path) -> dict:
    """Relative path -> (bytes, modification time) of every file below ``directory``."""
    return {
        p.relative_to(directory).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


# =================================================================== config
SHIPPED = {
    "tabular_q_8x8.yaml": dict(exp="tabular_q_W8", W=8, n_envs=50, total=500_000, every=25_000, episodes=100),
    "tabular_q_10x10.yaml": dict(exp="tabular_q_W10", W=10, n_envs=50, total=1_000_000, every=50_000, episodes=100),
    "smoke_tabular.yaml": dict(exp="smoke_tabular", W=8, n_envs=16, total=32_000, every=8_000, episodes=20),
}


@pytest.mark.parametrize("name", sorted(SHIPPED))
def test_shipped_config_files_load(name):
    want = SHIPPED[name]
    cfg = load_config(CONFIGS / name)
    assert cfg == ExperimentConfig(
        exp=want["exp"],
        env=EnvConfig(W=want["W"], action_mode="relative", starve_limit=None, obs=ObsConfig(kind="A"),
                      reward=RewardConfig(variant="R0")),
        agent_kind="tabular_q",
        agent=TabularQConfig(alpha=0.1, gamma=0.99, eps_start=1.0, eps_end=0.05, eps_decay_fraction=0.2, q_init=0.0),
        train=TrainConfig(n_envs=want["n_envs"], total_env_steps=want["total"], eval_every=want["every"],
                          eval_episodes=want["episodes"], eval_seed=10_000),
    )
    assert cfg.env.H == 100


@pytest.mark.parametrize("name", sorted(SHIPPED))
def test_shipped_config_files_start_with_a_comment_saying_how_to_run_them(name):
    text = (CONFIGS / name).read_text(encoding="utf-8")
    comment = "".join(line for line in text.splitlines(keepends=True) if line.startswith("#"))
    assert text.startswith("# ")
    assert f"python -m snake_rl.train --config configs/{name} --seed 0" in comment


def test_only_exp_and_agent_kind_are_required(tmp_path):
    cfg = load_config(write_yaml(tmp_path / "c.yaml", {"exp": "minimal", "agent": {"kind": "tabular_q"}}))
    assert cfg == ExperimentConfig("minimal", EnvConfig(), "tabular_q", TabularQConfig(), TrainConfig())


def test_dataclass_defaults():
    assert TrainConfig() == TrainConfig(n_envs=50, total_env_steps=1_000_000, eval_every=50_000,
                                        eval_episodes=100, eval_seed=DEFAULT_EVAL_SEED)
    assert DEFAULT_EVAL_SEED == 10_000
    assert TabularQConfig() == TabularQConfig(alpha=0.1, gamma=0.99, eps_start=1.0, eps_end=0.05,
                                              eps_decay_fraction=0.2, q_init=0.0)
    # The defaults must themselves pass the validation.
    cfg = ExperimentConfig("d", EnvConfig(), "tabular_q", TabularQConfig(), TrainConfig())
    assert config_from_dict(config_to_dict(cfg)) == cfg


def test_missing_keys_take_the_defaults_key_by_key(tmp_path):
    data = {
        "exp": "partial",
        "env": {"W": 6, "reward": {"variant": "R3"}},
        "agent": {"kind": "tabular_q", "gamma": 0.9},
        "train": {"eval_episodes": 30},
    }
    cfg = load_config(write_yaml(tmp_path / "c.yaml", data))
    assert cfg.env == EnvConfig(W=6, reward=RewardConfig(variant="R3"))
    assert cfg.agent == TabularQConfig(gamma=0.9)
    assert cfg.train == TrainConfig(eval_episodes=30)


def test_an_empty_section_means_all_defaults(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("exp: e\nenv:\n  obs:\n  reward:\nagent: {kind: tabular_q}\ntrain:\n", encoding="utf-8")
    assert load_config(path) == ExperimentConfig("e", EnvConfig(), "tabular_q", TabularQConfig(), TrainConfig())


def test_the_hand_written_valid_dict_loads(tmp_path):
    cfg = load_config(write_yaml(tmp_path / "c.yaml", valid_dict()))
    assert cfg.train == TrainConfig(16, 32_000, 8_000, 20, 10_000)
    assert config_to_dict(cfg) == valid_dict()


INT_FIELDS = ("train.n_envs", "train.total_env_steps", "train.eval_every", "train.eval_episodes",
              "train.eval_seed", "env.W", "env.starve_limit")
NUMBER_FIELDS = ("agent.alpha", "agent.gamma", "agent.eps_start", "agent.eps_end", "agent.eps_decay_fraction",
                 "agent.q_init", "env.reward.food", "env.reward.death", "env.reward.step_penalty",
                 "env.reward.gamma", "env.reward.closer", "env.reward.farther")

#: (changes to the valid dict, key the error must name)
INVALID = [
    # ---- unknown keys, at every nesting level
    ({"bogus": 1}, "bogus"),
    ({"env.w": 8}, "env.w"),
    ({"env.obs.kinds": "A"}, "env.obs.kinds"),
    ({"env.reward.gama": 0.9}, "env.reward.gama"),
    ({"agent.lr": 0.1}, "agent.lr"),
    ({"agent.epsilon": 0.1}, "agent.epsilon"),
    ({"train.envs": 8}, "train.envs"),
    ({"train.seed": 0}, "train.seed"),
    # ---- types
    *[({key: 128.0}, key) for key in INT_FIELDS],
    *[({key: True}, key) for key in INT_FIELDS],
    *[({key: "16"}, key) for key in INT_FIELDS],
    *[({key: None}, key) for key in INT_FIELDS if key != "env.starve_limit"],
    *[({key: "0.5"}, key) for key in NUMBER_FIELDS],
    *[({key: True}, key) for key in NUMBER_FIELDS],
    *[({key: None}, key) for key in NUMBER_FIELDS],
    ({"agent.q_init": float("nan")}, "agent.q_init"),
    ({"env.reward.death": float("-inf")}, "env.reward.death"),
    ({"env.obs.age_plane": 1}, "env.obs.age_plane"),
    ({"env.obs.heading_planes": "yes"}, "env.obs.heading_planes"),
    ({"env": 5}, "env"),
    ({"env.obs": "A"}, "env.obs"),
    ({"env.reward": "R0"}, "env.reward"),
    ({"agent": "tabular_q"}, "agent"),
    ({"train": [1, 2]}, "train"),
    # ---- environment
    ({"env.W": 3}, "env.W"),
    ({"env.W": -8}, "env.W"),
    ({"env.starve_limit": 0}, "env.starve_limit"),
    ({"env.starve_limit": -5}, "env.starve_limit"),
    ({"env.action_mode": "diagonal"}, "env.action_mode"),
    ({"env.action_mode": 3}, "env.action_mode"),
    ({"env.obs.kind": "C"}, "env.obs.kind"),
    ({"env.obs.kind": ["A"]}, "env.obs.kind"),
    ({"env.reward.variant": "R4"}, "env.reward.variant"),
    ({"env.reward.variant": "r0"}, "env.reward.variant"),
    ({"env.reward.variant": 0}, "env.reward.variant"),
    # ---- agent
    ({"agent.kind": "dqn"}, "agent.kind"),
    ({"agent.kind": "sarsa"}, "agent.kind"),
    ({"agent.kind": 1}, "agent.kind"),
    ({"agent.kind": DELETE}, "agent.kind"),
    ({"agent": DELETE}, "agent.kind"),
    ({"env.obs.kind": "B"}, "env.obs.kind"),  # tabular_q needs observation A
    ({"agent.alpha": 0}, "agent.alpha"),
    ({"agent.alpha": 0.0}, "agent.alpha"),
    ({"agent.alpha": 1.5}, "agent.alpha"),
    ({"agent.alpha": -0.1}, "agent.alpha"),
    ({"agent.gamma": -0.01}, "agent.gamma"),
    ({"agent.gamma": 1.01}, "agent.gamma"),
    ({"agent.eps_start": 0.2, "agent.eps_end": 0.5}, "agent.eps_end"),
    ({"agent.eps_start": 1.5}, "agent.eps_start"),
    ({"agent.eps_end": -0.1}, "agent.eps_end"),
    ({"agent.eps_decay_fraction": 1.5}, "agent.eps_decay_fraction"),
    ({"agent.eps_decay_fraction": -0.1}, "agent.eps_decay_fraction"),
    # ---- training budget
    ({"train.n_envs": 0}, "train.n_envs"),
    ({"train.n_envs": -16}, "train.n_envs"),
    ({"train.n_envs": 100_001, "train.eval_every": 100_001, "train.total_env_steps": 100_001}, "train.n_envs"),
    ({"train.eval_every": 8}, "train.eval_every"),  # below n_envs = 16
    ({"train.eval_every": 0}, "train.eval_every"),
    ({"train.eval_every": 8_008, "train.total_env_steps": 16_016}, "train.eval_every"),  # not a multiple of 16
    ({"train.n_envs": 64, "train.eval_every": 50_000, "train.total_env_steps": 1_000_000}, "train.eval_every"),
    ({"train.total_env_steps": 36_000}, "train.total_env_steps"),  # not a multiple of 8000
    ({"train.total_env_steps": 0}, "train.total_env_steps"),
    ({"train.total_env_steps": -32_000}, "train.total_env_steps"),
    ({"train.eval_episodes": 1}, "train.eval_episodes"),
    ({"train.eval_episodes": 0}, "train.eval_episodes"),
    ({"train.eval_seed": -1}, "train.eval_seed"),
    ({"train.eval_seed": 99_990}, "train.eval_seed"),  # 20 boards would reach the first training seed block
    # ---- experiment name
    ({"exp": DELETE}, "exp"),
    ({"exp": None}, "exp"),
    ({"exp": ""}, "exp"),
    ({"exp": "a/b"}, "exp"),
    ({"exp": "a\\b"}, "exp"),
    ({"exp": "../up"}, "exp"),
    ({"exp": ".."}, "exp"),
    ({"exp": "."}, "exp"),
    ({"exp": "name."}, "exp"),
    ({"exp": "has space"}, "exp"),
    ({"exp": "a:b"}, "exp"),
    ({"exp": "NUL"}, "exp"),
    ({"exp": 2026}, "exp"),
    ({"exp": ["a"]}, "exp"),
    # ---- reward R2 is tied to the agent's discount
    ({"env.reward.variant": "R2", "env.reward.gamma": 0.9}, "env.reward.gamma"),
    ({"env.reward.variant": "R2", "agent.gamma": 0.9}, "agent.gamma"),
]


def _case_id(case) -> str:
    changes, _ = case
    return ",".join(f"{k}={'<del>' if v is DELETE else v!r}" for k, v in changes.items())


@pytest.mark.parametrize("case", INVALID, ids=_case_id)
def test_invalid_config_is_a_value_error_naming_the_key(case, tmp_path):
    changes, key = case
    path = write_yaml(tmp_path / "c.yaml", edited(changes))
    message = config_error(path)
    assert names_key(message, key), message


def test_an_empty_or_non_mapping_file_is_a_value_error(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    assert names_key(config_error(empty), "exp")
    assert "mapping" in config_error(write_yaml(tmp_path / "list.yaml", [1, 2]))
    broken = tmp_path / "broken.yaml"
    broken.write_text("exp: [unclosed\n", encoding="utf-8")
    assert "YAML" in config_error(broken)


def test_a_missing_config_file_is_reported_as_such(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


VALID = [
    ({"agent.alpha": 1}, lambda c: c.agent.alpha == 1.0 and type(c.agent.alpha) is float),
    ({"agent.alpha": 1.0e-3}, lambda c: c.agent.alpha == 0.001),
    ({"agent.gamma": 0}, lambda c: c.agent.gamma == 0.0),
    ({"agent.gamma": 1}, lambda c: c.agent.gamma == 1.0),
    ({"agent.eps_start": 0.3, "agent.eps_end": 0.3}, lambda c: c.agent.eps_start == c.agent.eps_end == 0.3),
    ({"agent.eps_start": 0, "agent.eps_end": 0}, lambda c: c.agent.eps_start == 0.0),
    ({"agent.eps_decay_fraction": 0}, lambda c: c.agent.eps_decay_fraction == 0.0),
    ({"agent.eps_decay_fraction": 1}, lambda c: c.agent.eps_decay_fraction == 1.0),
    ({"agent.q_init": -2.5}, lambda c: c.agent.q_init == -2.5),
    ({"env.W": 4}, lambda c: c.env.W == 4),
    ({"env.starve_limit": 1}, lambda c: c.env.starve_limit == 1 and c.env.H == 1),
    ({"env.starve_limit": 250}, lambda c: c.env.H == 250),
    ({"env.action_mode": "absolute"}, lambda c: c.env.action_mode == "absolute"),
    ({"env.obs.age_plane": False}, lambda c: c.env.obs == ObsConfig("A", age_plane=False)),
    ({"env.reward.variant": "R1", "env.reward.gamma": 0.5}, lambda c: c.env.reward.gamma == 0.5),  # only R2 ties it
    ({"env.reward.variant": "R2"}, lambda c: c.env.reward.gamma == c.agent.gamma == 0.99),
    ({"env.reward.variant": "R2", "env.reward.gamma": 0.9, "agent.gamma": 0.9}, lambda c: c.agent.gamma == 0.9),
    ({"env.reward.variant": "R3", "env.reward.farther": -0.01}, lambda c: c.env.reward.farther == -0.01),
    ({"train.n_envs": 100_000, "train.eval_every": 100_000, "train.total_env_steps": 200_000},
     lambda c: c.train.n_envs == 100_000),
    ({"train.eval_every": 16, "train.total_env_steps": 16}, lambda c: c.train.eval_every == c.train.n_envs),
    ({"train.eval_episodes": 2}, lambda c: c.train.eval_episodes == 2),
    ({"train.eval_seed": 0}, lambda c: c.train.eval_seed == 0),
    ({"train.eval_seed": 99_980}, lambda c: c.train.eval_seed + c.train.eval_episodes == 100_000),
    ({"exp": "a"}, lambda c: c.exp == "a"),
    ({"exp": "2026"}, lambda c: c.exp == "2026"),
    ({"exp": "tabular_q_W10__alpha=0.1__gamma=0.99"}, lambda c: c.exp == "tabular_q_W10__alpha=0.1__gamma=0.99"),
    ({"exp": "A-b_c.d+e"}, lambda c: c.exp == "A-b_c.d+e"),
    ({"exp": ".hidden"}, lambda c: c.exp == ".hidden"),
]


@pytest.mark.parametrize("case", VALID, ids=_case_id)
def test_boundary_values_are_accepted(case, tmp_path):
    changes, check = case
    cfg = load_config(write_yaml(tmp_path / "c.yaml", edited(changes)))
    assert check(cfg)
    assert type(cfg.train.n_envs) is int and type(cfg.env.W) is int
    assert all(type(v) is float for v in dataclasses.asdict(cfg.agent).values())


def test_config_types_are_frozen_dataclasses():
    cfg = load_config(SMOKE)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.exp = "other"
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.train.n_envs = 1


# ---------------------------------------------------------------- overrides
def test_overrides_reach_every_level():
    cfg = load_config(
        SMOKE,
        ["agent.alpha=0.2", "env.reward.variant=R1", "exp=my_name", "env.starve_limit=40", "env.W=6",
         "env.action_mode=absolute", "train.eval_episodes=10", "env.reward.step_penalty=0.05"],
    )
    base = load_config(SMOKE)
    assert cfg.exp == "my_name"
    assert cfg.agent == dataclasses.replace(base.agent, alpha=0.2)
    assert cfg.env == EnvConfig(W=6, action_mode="absolute", starve_limit=40,
                                reward=RewardConfig(variant="R1", step_penalty=0.05))
    assert cfg.train == dataclasses.replace(base.train, eval_episodes=10)


def test_overrides_are_applied_in_order_and_values_are_yaml():
    assert load_config(SMOKE, ["agent.alpha=0.2", "agent.alpha=0.3"]).agent.alpha == 0.3
    assert load_config(SMOKE, ["env.starve_limit=7", "env.starve_limit=null"]).env.starve_limit is None
    assert load_config(SMOKE, ["env.starve_limit=7", "env.starve_limit="]).env.starve_limit is None
    assert load_config(SMOKE, ["env.obs={kind: A, age_plane: false}"]).env.obs == ObsConfig("A", age_plane=False)
    assert load_config(SMOKE, ["agent.alpha=1.0e-3"]).agent.alpha == 0.001
    assert load_config(SMOKE, ['exp="2026"']).exp == "2026"
    assert load_config(SMOKE, "agent.gamma=0.5").agent.gamma == 0.5  # one override given as a bare string
    assert load_config(SMOKE, ["exp=a=b"]).exp == "a=b"  # only the first "=" separates


def test_overrides_create_missing_sections(tmp_path):
    path = write_yaml(tmp_path / "c.yaml", {"exp": "m", "agent": {"kind": "tabular_q"}})
    cfg = load_config(path, ["env.reward.variant=R3", "env.obs.kind=A", "train.eval_episodes=7"])
    assert cfg.env.reward.variant == "R3" and cfg.train.eval_episodes == 7
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == {"exp": "m", "agent": {"kind": "tabular_q"}}


@pytest.mark.parametrize(
    "override, key",
    [
        ("env.W=3", "env.W"),
        ("train.n_envs=128.0", "train.n_envs"),
        ("train.n_envs=true", "train.n_envs"),
        ("train.n_envs=48", "train.eval_every"),  # 8000 is not a multiple of 48
        ("agent.alhpa=0.2", "agent.alhpa"),
        ("env.reward.variant=R9", "env.reward.variant"),
        ("env.rewards.variant=R1", "env.rewards"),
        ("agent.kind=dqn", "agent.kind"),
        ("env.obs.kind=B", "env.obs.kind"),
        ("exp=a/b", "exp"),
        ("exp=2026", "exp"),  # YAML reads an integer; a name must be a string
        ("agent.alpha=1e-3", "agent.alpha"),  # YAML reads a string; the message says how to write it
        ("seed=3", "seed"),
    ],
)
def test_an_override_that_fails_validation_names_the_key(override, key):
    message = config_error(SMOKE, [override])
    assert names_key(message, key), message


def test_the_scientific_notation_trap_is_explained():
    assert "1.0e-3" in config_error(SMOKE, ["agent.alpha=1e-3"])


@pytest.mark.parametrize("override", ["agent.alpha", "=3", "agent..alpha=1", ".alpha=1", "exp.name=x", ""])
def test_a_malformed_override_is_a_value_error(override):
    message = config_error(SMOKE, [override])
    assert repr(override) in message


# --------------------------------------------------------------- round trip
def test_config_to_dict_spells_out_every_field():
    assert config_to_dict(load_config(SMOKE)) == {
        "exp": "smoke_tabular",
        "env": {
            "W": 8,
            "action_mode": "relative",
            "starve_limit": None,
            "obs": {"kind": "A", "age_plane": True, "heading_planes": True},
            "reward": {"variant": "R0", "food": 1.0, "death": -1.0, "step_penalty": 0.01, "gamma": 0.99,
                       "closer": 0.01, "farther": 0.0},
        },
        "agent": {"kind": "tabular_q", "alpha": 0.1, "gamma": 0.99, "eps_start": 1.0, "eps_end": 0.05,
                  "eps_decay_fraction": 0.2, "q_init": 0.0},
        "train": {"n_envs": 16, "total_env_steps": 32000, "eval_every": 8000, "eval_episodes": 20,
                  "eval_seed": 10000},
    }
    assert list(config_to_dict(load_config(SMOKE))) == ["exp", "env", "agent", "train"]


ROUND_TRIP = [
    (SMOKE, ()),
    (BOARD_8, ()),
    (BOARD_10, ()),
    (SMOKE, ("env.reward.variant=R2", "env.reward.gamma=0.9", "agent.gamma=0.9", "env.starve_limit=33")),
    (SMOKE, VARIANTS["absolute_R3"][1]),
    (SMOKE, ("agent.alpha=0.123456789012345678", "agent.q_init=-1.0e-9", "exp=x=1.5+y")),
]


@pytest.mark.parametrize("path, overrides", ROUND_TRIP)
def test_config_to_dict_round_trips_through_a_yaml_file(path, overrides, tmp_path):
    cfg = load_config(path, overrides)
    d = config_to_dict(cfg)
    assert json.loads(json.dumps(d)) == d  # plain Python types only
    assert config_from_dict(d) == cfg
    assert load_config(write_yaml(tmp_path / "copy.yaml", d)) == cfg
    assert config_to_dict(load_config(tmp_path / "copy.yaml")) == d


def test_a_run_block_is_accepted_and_ignored(tmp_path):
    d = valid_dict()
    d["run"] = {"seed": 3, "git": "abc", "anything": [1, {"at": "all"}]}
    assert load_config(write_yaml(tmp_path / "c.yaml", d)) == load_config(write_yaml(tmp_path / "d.yaml", valid_dict()))


# -------------------------------------------------------------------- seeds
def test_env_seed_for_values():
    assert [env_seed_for(s, 64) for s in range(5)] == [100_000, 200_000, 300_000, 400_000, 500_000]
    assert env_seed_for(0, 1) == 100_000
    assert env_seed_for(7, 100_000) == 800_000
    assert type(env_seed_for(0, 64)) is int


@pytest.mark.parametrize(
    "run_seed, n_envs",
    [(-1, 64), (True, 64), (0.0, 64), ("0", 64), (None, 64), (np.int64(0), 64), (0, 100_001), (0, 0), (0, -1), (0, 64.0)],
)
def test_env_seed_for_errors(run_seed, n_envs):
    with pytest.raises(ValueError):
        env_seed_for(run_seed, n_envs)


def _overlap(a: range, b: range) -> int:
    return len(range(max(a.start, b.start), min(a.stop, b.stop)))


@pytest.mark.parametrize("n_envs", [1, 16, 50, 64, 256, 99_999, 100_000])
def test_seed_blocks_of_runs_and_of_the_evaluation_are_disjoint(n_envs):
    # VecSnakeEnv gives sub-env i the generator seed + i: a run owns range(env_seed, env_seed + n_envs).
    blocks = [range(env_seed_for(s, n_envs), env_seed_for(s, n_envs) + n_envs) for s in range(5)]
    evaluation = range(DEFAULT_EVAL_SEED, DEFAULT_EVAL_SEED + 100)
    widest_evaluation = range(0, 100_000)  # anything the config validation lets through
    for i, a in enumerate(blocks):
        assert len(a) == n_envs
        assert _overlap(a, evaluation) == 0 and _overlap(a, widest_evaluation) == 0
        for b in blocks[i + 1:]:
            assert _overlap(a, b) == 0


# ============================================================ run directory
def test_run_directory_is_runs_exp_seed(smoke_run):
    assert smoke_run.dir == smoke_run.runs / "smoke_tabular" / "0"
    assert smoke_run.dir.is_dir()


def test_smoke_run_writes_exactly_four_files(smoke_run):
    assert sorted(p.name for p in smoke_run.dir.iterdir()) == RUN_FILES
    assert sorted(snapshot(smoke_run.runs)) == [f"smoke_tabular/0/{name}" for name in RUN_FILES]


def test_config_yaml_holds_the_full_config_and_the_run_block(smoke_run):
    recorded = read_run_config(smoke_run.dir)
    run = recorded.pop("run")
    assert recorded == config_to_dict(smoke_run.cfg)
    assert sorted(run) == sorted(["seed", "env_seed", "git", "started", "python", "numpy", "platform", "argv"])
    assert run["seed"] == 0
    assert run["env_seed"] == 100_000
    assert run["git"] == FAKE_GIT
    assert run["argv"] == smoke_run.argv
    assert run["python"] == platform.python_version()
    assert run["numpy"] == np.__version__
    assert run["platform"] == platform.platform()
    started = datetime.fromisoformat(run["started"])
    assert started.tzinfo is not None
    age = (datetime.now().astimezone() - started).total_seconds()
    assert -5 <= age < 3600


def test_config_yaml_of_a_finished_run_loads_again(smoke_run):
    assert load_config(smoke_run.dir / "config.yaml") == smoke_run.cfg
    renamed = load_config(smoke_run.dir / "config.yaml", ["exp=again"])
    assert renamed == dataclasses.replace(smoke_run.cfg, exp="again")


def test_argv_is_null_when_train_is_called_without_one(tiny_run):
    run = read_run_config(tiny_run.dir)["run"]
    assert run["argv"] is None
    assert run["seed"] == 3 and run["env_seed"] == 400_000


def test_metrics_header_and_row_positions(smoke_run):
    raw = (smoke_run.dir / "metrics.csv").read_bytes()
    assert b"\r" not in raw
    lines = raw.decode("utf-8").splitlines()
    assert lines[0] == HEADER
    tc = smoke_run.cfg.train
    rows = read_metrics(smoke_run.dir)
    assert len(lines) - 1 == len(rows) == tc.total_env_steps // tc.eval_every + 1 == 5
    assert [r["env_steps"] for r in rows] == [0, 8_000, 16_000, 24_000, 32_000]
    assert [r["vec_steps"] for r in rows] == [0, 500, 1_000, 1_500, 2_000]


def test_first_metrics_row_is_the_untrained_agent(smoke_run):
    first = read_metrics(smoke_run.dir)[0]
    assert first["epsilon"] == smoke_run.cfg.agent.eps_start == 1.0
    assert first["train_episodes"] == 0
    assert first["train_mean_score"] is None
    assert first["mean_abs_td"] is None
    assert first["cells_visited"] == 0
    # An all-zero table always goes straight: every board ends at the wall after 4 steps.
    assert first["eval_mean_score"] == 0.0 and first["eval_frac_wall"] == 1.0 and first["eval_mean_steps"] == 4.0


def test_later_metrics_rows_are_filled_and_consistent(smoke_run):
    rows = read_metrics(smoke_run.dir)
    table = load_table(smoke_run.dir)
    for row in rows[1:]:
        assert all(value is not None for value in row.values())
        assert row["train_episodes"] > 0 and row["train_mean_score"] >= 0 and row["mean_abs_td"] > 0
        fractions = [row[f"eval_frac_{c}"] for c in ("wall", "self", "starve", "win")]
        assert sum(fractions) == pytest.approx(1.0)
    visited = [r["cells_visited"] for r in rows]
    assert visited == sorted(visited)
    assert visited[-1] == int((table.visits > 0).sum()) > 0
    wall = [r["wall_time_s"] for r in rows]
    assert wall == sorted(wall) and wall[0] >= 0


def test_epsilon_column_is_the_schedule_value_of_the_last_vec_step(variant_run):
    ac, tc = variant_run.cfg.agent, variant_run.cfg.train
    decay_steps = ac.eps_decay_fraction * tc.total_env_steps
    rows = read_metrics(variant_run.dir)
    assert rows[0]["epsilon"] == ac.eps_start
    for row in rows[1:]:
        steps_before = row["env_steps"] - tc.n_envs  # the last vec-step started here
        expected = ac.eps_start + (ac.eps_end - ac.eps_start) * min(1.0, steps_before / decay_steps)
        assert row["epsilon"] == expected
    decaying = [r["epsilon"] for r in rows if ac.eps_end + 1e-9 < r["epsilon"] < ac.eps_start - 1e-9]
    assert len(decaying) >= 3  # the check above is not only about the plateau


def test_without_a_decay_phase_epsilon_is_eps_end_from_the_first_step(tmp_path):
    cfg = load_config(SMOKE, [*TINY, "agent.eps_decay_fraction=0", "agent.eps_start=0.9", "agent.eps_end=0.3"])
    rows = read_metrics(train(cfg, 0, tmp_path, quiet=True))
    assert [r["epsilon"] for r in rows] == [0.9, 0.3, 0.3, 0.3]  # row 0: no step taken yet


def test_last_metrics_row_is_the_final_evaluation(smoke_run):
    last = read_metrics(smoke_run.dir)[-1]
    summary = EvalResult.load(smoke_run.dir / "final_eval.json").summary()
    assert last["eval_mean_score"] == summary["mean_score"]
    assert last["eval_ci95"] == summary["ci95"]
    assert last["eval_median_score"] == summary["median_score"]
    assert last["eval_max_score"] == summary["max_score"]
    assert last["eval_mean_steps"] == summary["mean_steps"]
    for cause in ("wall", "self", "starve", "win"):
        assert last[f"eval_frac_{cause}"] == summary["cause_fractions"][cause]


def test_final_eval_is_the_greedy_evaluation_of_the_saved_table(smoke_run):
    cfg = smoke_run.cfg
    agent = load_table(smoke_run.dir)
    stored = json.loads((smoke_run.dir / "final_eval.json").read_text(encoding="utf-8"))
    final = EvalResult.load(smoke_run.dir / "final_eval.json")
    assert len(final.episodes) == cfg.train.eval_episodes == 20
    assert [e.seed for e in final.episodes] == list(range(10_000, 10_020))

    batched = evaluate_batched(agent, cfg.env, cfg.train.eval_episodes, cfg.train.eval_seed)
    scalar = evaluate(agent, cfg.env, cfg.train.eval_episodes, cfg.train.eval_seed)
    assert final.episodes == batched.episodes == scalar.episodes
    for key in ("policy", "env", "seed", "git", "summary", "episodes"):  # everything but the wall time
        assert stored[key] == batched.to_dict()[key]
    assert stored["policy"] == "tabular_q" and stored["seed"] == 10_000 and stored["git"] == FAKE_GIT
    assert final.mean_score > 1.0  # the smoke run learns something


def test_saved_table_is_the_trained_agent(smoke_run):
    agent = load_table(smoke_run.dir)
    assert agent.cfg == smoke_run.cfg.agent
    assert agent.n_actions == 3
    assert int(agent.visits.sum()) == smoke_run.cfg.train.total_env_steps  # one transition per env step
    assert np.isfinite(agent.q).all() and np.abs(agent.q).max() > 0


def test_evaluation_seed_and_size_come_from_the_config(variant_run):
    tc = variant_run.cfg.train
    final = EvalResult.load(variant_run.dir / "final_eval.json")
    assert final.seed == tc.eval_seed != DEFAULT_EVAL_SEED
    assert [e.seed for e in final.episodes] == list(range(tc.eval_seed, tc.eval_seed + tc.eval_episodes))
    agent = load_table(variant_run.dir)
    assert final.episodes == evaluate(agent, variant_run.cfg.env, tc.eval_episodes, tc.eval_seed).episodes
    assert final.env == config_to_dict(variant_run.cfg)["env"]


# ============================================================== determinism
def test_same_config_and_seed_give_the_same_run(tiny_run, tmp_path):
    again = train(tiny_run.cfg, tiny_run.seed, tmp_path / "again", quiet=True)
    assert again != tiny_run.dir
    assert_same_run(tiny_run.dir, again)


def test_a_different_seed_gives_a_different_run(tiny_run, tmp_path):
    other = train(tiny_run.cfg, tiny_run.seed + 1, tmp_path / "other", quiet=True)
    assert other.name == "4"
    a, b = load_table(tiny_run.dir), load_table(other)
    assert not np.array_equal(a.q, b.q)
    assert not np.array_equal(a.visits, b.visits)
    assert metrics_without_wall_time(tiny_run.dir) != metrics_without_wall_time(other)


def test_rerun_from_the_saved_config_reproduces_the_table(tiny_run, tmp_path):
    cfg = load_config(tiny_run.dir / "config.yaml", ["exp=rerun"])
    seed = read_run_config(tiny_run.dir)["run"]["seed"]
    rerun = train(cfg, seed, tmp_path, quiet=True)
    assert rerun == tmp_path / "rerun" / "3"
    assert_same_run(tiny_run.dir, rerun)


def test_fresh_process_reproduces_the_in_process_run(tiny_run, tmp_path):
    args = ["--config", str(SMOKE), "--seed", str(tiny_run.seed), "--runs-dir", str(tmp_path / "sub"), "--quiet",
            "--set", *TINY]
    done = subprocess.run(
        [sys.executable, "-m", "snake_rl.train", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout == ""  # --quiet
    run_dir = tmp_path / "sub" / "smoke_tabular" / "3"
    assert sorted(p.name for p in run_dir.iterdir()) == RUN_FILES
    assert_same_run(tiny_run.dir, run_dir)
    run = read_run_config(run_dir)["run"]
    assert run["argv"] == args
    assert run["git"] == git_hash() != FAKE_GIT  # the real hash, taken by the child process
    assert load_config(run_dir / "config.yaml") == tiny_run.cfg


# ====================================================== the loop, pinned down
def reference_run(cfg: ExperimentConfig, seed: int):
    """The training protocol written a second time from the specification,
    without ``snake_rl.train``, ``env_seed_for``, ``linear_schedule``,
    ``act_epsilon`` or ``evaluate_batched``. Returns (agent, metrics rows,
    final evaluation)."""
    ac, tc = cfg.agent, cfg.train
    n = tc.n_envs
    env = VecSnakeEnv(cfg.env, n_envs=n, seed=100_000 * (seed + 1))
    agent = TabularQAgent(env.n_actions, ac)
    rng = np.random.default_rng([seed, 1])
    decay_steps = ac.eps_decay_fraction * tc.total_env_steps
    rows: list[dict] = []
    results: list[EvalResult] = []

    def evaluation(env_steps, epsilon, scores, abs_tds):
        result = evaluate(agent, cfg.env, tc.eval_episodes, tc.eval_seed)  # scalar env, one episode at a time
        fractions = result.cause_fractions
        results.append(result)
        rows.append({
            "env_steps": env_steps,
            "vec_steps": env_steps // n,
            "epsilon": epsilon,
            "eval_mean_score": float(result.scores.mean()),
            "eval_ci95": float(1.96 * result.scores.std(ddof=1) / np.sqrt(len(result.episodes))),
            "eval_median_score": float(np.median(result.scores)),
            "eval_max_score": int(result.scores.max()),
            "eval_mean_steps": float(result.steps.mean()),
            "eval_frac_wall": fractions["wall"],
            "eval_frac_self": fractions["self"],
            "eval_frac_starve": fractions["starve"],
            "eval_frac_win": fractions["win"],
            "train_episodes": len(scores),
            "train_mean_score": float(np.mean(scores)) if scores else None,
            "mean_abs_td": float(np.mean(abs_tds)) if abs_tds else None,
            "cells_visited": int((agent.visits > 0).sum()),
        })

    obs = env.reset()
    evaluation(0, ac.eps_start, [], [])
    scores, abs_tds = [], []
    for k in range(tc.total_env_steps // n):
        steps_before = k * n
        epsilon = ac.eps_start + (ac.eps_end - ac.eps_start) * min(1.0, steps_before / decay_steps)
        explore = rng.random(n) < epsilon
        random_actions = rng.integers(0, env.n_actions, size=n)
        actions = np.where(explore, random_actions, agent.act_batch(obs))
        next_obs, reward, terminated, truncated, info = env.step(actions)
        abs_tds.append(agent.update(obs, actions, reward, info["final_obs"], terminated))
        scores.extend(info["score"][info["done"]].tolist())
        obs = next_obs
        if (steps_before + n) % tc.eval_every == 0:
            evaluation(steps_before + n, epsilon, scores, abs_tds)
            scores, abs_tds = [], []
    return agent, rows, results[-1]


def test_independent_loop_reproduces_train_exactly(variant_run):
    agent, rows, final = reference_run(variant_run.cfg, variant_run.seed)
    trained = load_table(variant_run.dir)
    assert trained.q.tobytes() == agent.q.tobytes()
    assert trained.visits.tobytes() == agent.visits.tobytes()
    assert final_episodes(variant_run.dir) == final.episodes

    written = read_metrics(variant_run.dir)
    assert len(written) == len(rows) == 7
    for got, want in zip(written, rows):
        assert set(got) == set(want) | {"wall_time_s"}
        for name, value in want.items():
            if name in ("train_mean_score", "mean_abs_td") and value is not None:
                assert got[name] == pytest.approx(value, rel=1e-12), name  # summation order is free
            else:
                assert got[name] == value, (name, want["env_steps"])
    # The runs these configs produce do exercise what the comparison is about.
    assert any(r["eval_frac_starve"] > 0 for r in rows)
    assert sum(r["train_episodes"] for r in rows) > 100


def test_evaluations_see_the_table_after_the_update_of_their_step(tmp_path, monkeypatch):
    cfg = load_config(SMOKE, TINY)
    tc = cfg.train
    seen = []

    def spy(policy, env_cfg, n_episodes, seed):
        seen.append(SimpleNamespace(transitions=int(policy.visits.sum()), q=policy.q.copy(), env=env_cfg,
                                    n_episodes=n_episodes, seed=seed))
        return evaluate_batched(policy, env_cfg, n_episodes, seed)

    monkeypatch.setattr(train_module, "evaluate_batched", spy)
    run_dir = train(cfg, 0, tmp_path, quiet=True)
    # Each update adds n_envs transitions to the visit counts, so the count
    # says how many vec-steps the table had seen when it was evaluated.
    assert [s.transitions for s in seen] == [0, 800, 1_600, 2_400]
    assert [s.transitions for s in seen] == [r["env_steps"] for r in read_metrics(run_dir)]
    assert all(s.env == cfg.env and s.n_episodes == tc.eval_episodes and s.seed == tc.eval_seed for s in seen)
    assert not seen[0].q.any()
    assert np.array_equal(seen[-1].q, load_table(run_dir).q)  # nothing is learned after the last evaluation


def test_training_bootstraps_truncated_transitions_from_the_true_successor(tmp_path, monkeypatch):
    seed, overrides = VARIANTS["relative_R0"]
    cfg = load_config(SMOKE, overrides)
    envs, observed, steps, updates = [], [], [], []

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

    class SpyAgent(TabularQAgent):
        def update(self, obs, actions, rewards, next_obs, terminated):
            updates.append(SimpleNamespace(obs=np.array(obs), actions=np.array(actions), rewards=np.array(rewards),
                                           next_obs=np.array(next_obs), terminated=np.array(terminated)))
            return super().update(obs, actions, rewards, next_obs, terminated)

    monkeypatch.setattr(train_module, "VecSnakeEnv", SpyEnv)
    monkeypatch.setattr(train_module, "TabularQAgent", SpyAgent)
    train(cfg, seed, tmp_path, quiet=True)

    n = cfg.train.n_envs
    assert envs == [(cfg.env, n, env_seed_for(seed, n))]  # one training env, seeded for this run
    assert len(steps) == len(updates) == cfg.train.total_env_steps // n
    truncated_total = terminated_total = moved_on = 0
    for k, (step, update) in enumerate(zip(steps, updates)):
        assert np.array_equal(update.obs, observed[k])  # the observation the action was chosen from
        assert np.array_equal(update.actions, step.actions)
        assert np.array_equal(update.rewards, step.reward)
        assert np.array_equal(update.next_obs, step.final_obs)  # true successor, not the post-reset obs
        assert update.terminated.dtype == bool
        assert np.array_equal(update.terminated, step.terminated)  # terminated only, never done
        assert not (update.terminated & step.truncated).any()
        truncated_total += int(step.truncated.sum())
        terminated_total += int(step.terminated.sum())
        ended = step.terminated | step.truncated
        moved_on += int((step.final_obs[ended] != observed[k + 1][ended]).any(axis=1).sum())
    # The run must contain what the assertions above are about.
    assert truncated_total > 50 and terminated_total > 50
    assert moved_on > 50  # final_obs and the returned obs really differ where an episode ended


def test_seed_map_of_a_run(tmp_path):
    """Every random generator a run creates, by its seed (added at integration:
    the direct statement of the seeding protocol, which is otherwise pinned
    only through the exact tables of the independent loop above).

    Training boards: ``100_000 * (run_seed + 1) + i``. Exploration: one
    generator, ``[run_seed, 1]``. Each evaluation: the boards ``eval_seed + i``,
    the same ones every time. Nothing else - in particular no unseeded
    generator, and no evaluation board among the training streams."""
    run_seed = 3
    cfg = load_config(SMOKE, [*TINY, "train.eval_seed=4242"])
    tc = cfg.train
    created = []
    real_default_rng = np.random.default_rng

    def spy(seed=None):
        created.append(seed)
        return real_default_rng(seed)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(np.random, "default_rng", spy)
        train(cfg, run_seed, tmp_path, quiet=True)

    assert None not in created  # an unseeded generator would make the run irreproducible
    exploration = [seed for seed in created if isinstance(seed, (list, tuple))]
    boards = [seed for seed in created if not isinstance(seed, (list, tuple))]
    assert [list(seed) for seed in exploration] == [[run_seed, 1]]
    assert all(isinstance(seed, (int, np.integer)) and not isinstance(seed, bool) for seed in boards)

    training = [100_000 * (run_seed + 1) + i for i in range(tc.n_envs)]
    evaluation = [4242 + i for i in range(tc.eval_episodes)]
    n_evaluations = tc.total_env_steps // tc.eval_every + 1
    assert n_evaluations == 4 and not set(training) & set(evaluation)
    expected = Counter(training) + Counter({seed: n_evaluations for seed in evaluation})
    assert Counter(int(seed) for seed in boards) == expected


# ===================================================== runs are never touched
def test_an_existing_run_is_refused_and_left_untouched(smoke_run, tmp_path):
    runs = tmp_path / "copy"
    shutil.copytree(smoke_run.runs, runs)
    before = snapshot(runs)
    assert len(before) == 4
    with pytest.raises(FileExistsError, match="smoke_tabular"):
        train(smoke_run.cfg, smoke_run.seed, runs, quiet=True)
    assert snapshot(runs) == before


@pytest.mark.parametrize("leftover", RUN_FILES)
def test_an_unfinished_run_directory_is_refused_too(leftover, tmp_path):
    cfg = load_config(SMOKE, TINY)
    run_dir = tmp_path / cfg.exp / "0"
    run_dir.mkdir(parents=True)
    (run_dir / leftover).write_text("left over from an interrupted run", encoding="utf-8")
    before = snapshot(tmp_path)
    with pytest.raises(FileExistsError):
        train(cfg, 0, tmp_path, quiet=True)
    assert snapshot(tmp_path) == before


def test_an_empty_run_directory_can_be_used(tmp_path):
    cfg = load_config(SMOKE, TINY)
    (tmp_path / cfg.exp / "0").mkdir(parents=True)
    run_dir = train(cfg, 0, tmp_path, quiet=True)
    assert sorted(p.name for p in run_dir.iterdir()) == RUN_FILES


def test_config_is_written_before_training_and_metrics_are_flushed_per_row(tmp_path, monkeypatch):
    cfg = load_config(SMOKE, TINY)
    run_dir = tmp_path / cfg.exp / "5"
    calls = []

    def spy(policy, env_cfg, n_episodes, seed):
        on_disk = sorted(p.name for p in run_dir.iterdir())
        lines = (run_dir / "metrics.csv").read_text(encoding="utf-8").splitlines()
        calls.append((on_disk, lines, read_run_config(run_dir)["run"]["seed"]))
        if len(calls) == 3:
            raise RuntimeError("interrupted")
        return evaluate_batched(policy, env_cfg, n_episodes, seed)

    monkeypatch.setattr(train_module, "evaluate_batched", spy)
    with pytest.raises(RuntimeError, match="interrupted"):
        train(cfg, 5, tmp_path, quiet=True)
    for j, (on_disk, lines, seed) in enumerate(calls):
        assert on_disk == ["config.yaml", "metrics.csv"]  # config first; table and final eval only at the end
        assert lines[0] == HEADER and len(lines) == 1 + j  # every earlier row is already on disk
        assert seed == 5
    # The interrupted run is recognisable (no final_eval.json) and is not resumed or overwritten.
    assert sorted(p.name for p in run_dir.iterdir()) == ["config.yaml", "metrics.csv"]
    before = snapshot(tmp_path)
    monkeypatch.setattr(train_module, "evaluate_batched", evaluate_batched)
    with pytest.raises(FileExistsError):
        train(cfg, 5, tmp_path, quiet=True)
    assert snapshot(tmp_path) == before


def test_final_eval_is_written_last(tmp_path, monkeypatch):
    """``final_eval.json`` is the "finished" marker for the analysis and the
    batch runner, so the table must already be on disk when it appears."""
    listings = []

    class SpyAgent(TabularQAgent):
        def save(self, path):
            listings.append(sorted(p.name for p in Path(path).parent.iterdir()))
            return super().save(path)

    monkeypatch.setattr(train_module, "TabularQAgent", SpyAgent)
    run_dir = train(load_config(SMOKE, TINY), 0, tmp_path, quiet=True)
    assert listings == [["config.yaml", "metrics.csv"]]  # saved once, before final_eval.json exists
    assert sorted(p.name for p in run_dir.iterdir()) == RUN_FILES


# ================================================================== dispatch
def test_only_tabular_q_is_trainable_in_m4(tmp_path):
    cfg = dataclasses.replace(load_config(SMOKE, TINY), agent_kind="dqn")
    with pytest.raises(ValueError, match="agent.kind"):
        train(cfg, 0, tmp_path, quiet=True)
    assert list(tmp_path.iterdir()) == []


def test_train_dispatches_on_the_agent_kind(tmp_path, monkeypatch):
    cfg = load_config(SMOKE, TINY)
    calls = []

    def fake_trainer(cfg_, seed, run_dir, quiet):
        calls.append((cfg_, seed, run_dir, quiet, sorted(p.name for p in run_dir.iterdir())))

    monkeypatch.setitem(train_module._TRAINERS, "tabular_q", fake_trainer)
    run_dir = train(cfg, 9, tmp_path, quiet=True)
    assert calls == [(cfg, 9, tmp_path / cfg.exp / "9", True, ["config.yaml"])]
    assert run_dir == tmp_path / cfg.exp / "9"

    monkeypatch.delitem(train_module._TRAINERS, "tabular_q")
    with pytest.raises(ValueError, match="tabular_q"):
        train(cfg, 10, tmp_path, quiet=True)
    assert not (tmp_path / cfg.exp / "10").exists()


def test_train_validates_a_hand_built_config(tmp_path):
    base = load_config(SMOKE, TINY)
    not_a_multiple = dataclasses.replace(base, train=TrainConfig(n_envs=64, total_env_steps=100_000, eval_every=50_000))
    with pytest.raises(ValueError, match="train.eval_every"):
        train(not_a_multiple, 0, tmp_path, quiet=True)
    with pytest.raises(ValueError, match="env.obs.kind"):
        train(dataclasses.replace(base, env=EnvConfig(W=8, obs=ObsConfig("B"))), 0, tmp_path, quiet=True)
    with pytest.raises(ValueError, match="exp"):
        train(dataclasses.replace(base, exp="../escape"), 0, tmp_path, quiet=True)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("seed", [-1, True, 1.0, "0", None])
def test_train_rejects_a_bad_seed(seed, tmp_path):
    with pytest.raises(ValueError, match="seed"):
        train(load_config(SMOKE, TINY), seed, tmp_path, quiet=True)
    assert list(tmp_path.iterdir()) == []


def test_default_runs_dir_is_runs_under_the_working_directory(tmp_path):
    cfg = load_config(SMOKE, TINY)
    run_dir = train(cfg, 0, quiet=True)  # isolated_cwd: the working directory is tmp_path
    assert run_dir == Path("runs") / cfg.exp / "0"
    assert sorted(p.name for p in (tmp_path / "runs" / cfg.exp / "0").iterdir()) == RUN_FILES


# ======================================================================= CLI
def test_cli_prints_one_progress_line_per_evaluation(tmp_path, capsys):
    argv = ["--config", str(SMOKE), "--seed", "1", "--runs-dir", str(tmp_path), "--set", *TINY]
    assert main(argv) == 0
    out = capsys.readouterr()
    assert out.err == ""
    progress = [line for line in out.out.splitlines() if " steps  eps " in line]
    assert len(progress) == 4
    for line, env_steps in zip(progress, ["0/2,400", "800/2,400", "1,600/2,400", "2,400/2,400"]):
        assert line.startswith("[smoke_tabular/1] ") and f" {env_steps} steps  eps " in line
        assert line.isascii()  # safe on a Windows console
    # A progress line shows the numbers of its metrics row (the mean, not the median).
    rows = read_metrics(tmp_path / "smoke_tabular" / "1")
    assert len(rows) == 4 and any(row["eval_mean_score"] != row["eval_median_score"] for row in rows)
    for line, row in zip(progress, rows):
        assert f" eps {row['epsilon']:.3f}  eval {row['eval_mean_score']:.2f} +- {row['eval_ci95']:.2f}  " in line
        assert (
            f" wall {row['eval_frac_wall']:.2f} self {row['eval_frac_self']:.2f} "
            f"starve {row['eval_frac_starve']:.2f} win {row['eval_frac_win']:.2f}  "
        ) in line
        assert f"({row['train_episodes']} ep)" in line
    assert read_run_config(tmp_path / "smoke_tabular" / "1")["run"]["argv"] == argv


def test_cli_quiet_prints_nothing(tmp_path, capsys):
    assert main(["--config", str(SMOKE), "--seed", "1", "--runs-dir", str(tmp_path), "--quiet", "--set", *TINY]) == 0
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""
    assert (tmp_path / "smoke_tabular" / "1" / "final_eval.json").is_file()


def test_cli_exp_and_set(tmp_path):
    argv = ["--config", str(SMOKE), "--seed", "7", "--exp", "renamed", "--set", "agent.alpha=0.5", TINY[0],
            "--runs-dir", str(tmp_path), "--set", *TINY[1:], "--quiet"]
    assert main(argv) == 0
    run_dir = tmp_path / "renamed" / "7"
    recorded = read_run_config(run_dir)
    assert recorded["exp"] == "renamed"
    assert recorded["agent"]["alpha"] == 0.5
    assert recorded["train"]["n_envs"] == 8 and recorded["train"]["total_env_steps"] == 2_400
    assert recorded["run"]["seed"] == 7 and recorded["run"]["env_seed"] == 800_000
    assert recorded["run"]["argv"] == argv
    assert load_table(run_dir).cfg.alpha == 0.5
    # The same thing through the Python API gives the same run.
    cfg = load_config(SMOKE, ["agent.alpha=0.5", *TINY, "exp=renamed"])
    assert_same_run(run_dir, train(cfg, 7, tmp_path / "api", quiet=True))


def test_cli_exp_wins_over_set_and_may_look_like_a_number(tmp_path):
    argv = ["--config", str(SMOKE), "--seed", "0", "--exp", "2026", "--set", "exp=ignored", *TINY,
            "--runs-dir", str(tmp_path), "--quiet"]
    assert main(argv) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["2026"]
    assert read_run_config(tmp_path / "2026" / "0")["exp"] == "2026"


@pytest.mark.parametrize(
    "extra, needle",
    [
        (["--set", "agent.alpha=0"], "agent.alpha"),
        (["--set", "agent.alhpa=0.2"], "agent.alhpa"),
        (["--set", "train.n_envs=128.0"], "train.n_envs"),
        (["--exp", "a/b"], "exp"),
        (["--set", "nonsense"], "nonsense"),
    ],
)
def test_cli_reports_an_invalid_config_and_writes_nothing(extra, needle, tmp_path, capsys):
    runs = tmp_path / "runs_here"
    assert main(["--config", str(SMOKE), "--seed", "0", "--runs-dir", str(runs), *extra]) == 2
    out = capsys.readouterr()
    assert out.out == ""
    assert out.err.startswith("error: ") and needle in out.err
    assert not runs.exists()


def test_cli_reports_a_missing_config_file(tmp_path, capsys):
    assert main(["--config", str(tmp_path / "nope.yaml"), "--seed", "0", "--runs-dir", str(tmp_path / "r")]) == 2
    assert "nope.yaml" in capsys.readouterr().err
    assert not (tmp_path / "r").exists()


def test_cli_refuses_an_existing_run(smoke_run, tmp_path, capsys):
    runs = tmp_path / "copy"
    shutil.copytree(smoke_run.runs, runs)
    before = snapshot(runs)
    assert main(["--config", str(SMOKE), "--seed", "0", "--runs-dir", str(runs)]) == 1
    out = capsys.readouterr()
    assert out.out == "" and "never overwritten" in out.err
    assert snapshot(runs) == before


@pytest.mark.parametrize("argv", [["--seed", "0"], ["--config", "x.yaml"], ["--config", "x.yaml", "--seed", "-1"],
                                  ["--config", "x.yaml", "--seed", "0.5"], ["--config", "x.yaml", "--seed", "0", "--run", "r"]])
def test_cli_usage_errors_exit_with_2(argv, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == 2
    capsys.readouterr()


# ============================================================ metrics writer
def test_metrics_writer_flushes_header_and_rows(tmp_path):
    path = tmp_path / "m.csv"
    with MetricsWriter(path, ["a", "b", "c"]) as writer:
        assert path.read_text(encoding="utf-8") == "a,b,c\n"
        writer.write({"c": None, "a": 3, "b": 0.1})
        assert path.read_text(encoding="utf-8") == "a,b,c\n3,0.1,\n"
        writer.write({"a": np.int64(7), "b": np.float64(1) / 3, "c": np.float32(0.5)})
        assert path.read_text(encoding="utf-8").splitlines()[-1] == f"7,{1 / 3!r},0.5"
        with pytest.raises(KeyError):
            writer.write({"a": 1, "b": 2})
        with pytest.raises(KeyError):
            writer.write({"a": 1, "b": 2, "c": 3, "d": 4})
    assert float(path.read_text(encoding="utf-8").splitlines()[-1].split(",")[1]) == 1 / 3  # nothing is rounded
    with pytest.raises(FileExistsError):
        MetricsWriter(path, ["a"])
    assert path.read_bytes().count(b"\n") == 3 and b"\r" not in path.read_bytes()


# ================================================================ acceptance
def test_shipped_8x8_config_ends_far_above_the_random_player(tmp_path):
    """M4 acceptance at the level of the entry point: a fifth of the shipped
    budget (100k of 500k env steps, about a second) already beats the random
    player by a wide margin under the same evaluation protocol."""
    cfg = load_config(BOARD_8, ["train.total_env_steps=100000"])
    assert cfg.train.eval_every == 25_000 and cfg.train.eval_episodes == 100
    run_dir = train(cfg, 0, tmp_path, quiet=True)
    final = EvalResult.load(run_dir / "final_eval.json")
    random_player = evaluate(make_policy("random", seed=cfg.train.eval_seed), cfg.env, cfg.train.eval_episodes,
                             cfg.train.eval_seed)
    assert [e.seed for e in final.episodes] == [e.seed for e in random_player.episodes]  # the same boards
    assert random_player.mean_score < 1.0
    assert final.mean_score > 8.0
    assert final.mean_score - final.ci95 > 10 * (random_player.mean_score + random_player.ci95)
    curve = [row["eval_mean_score"] for row in read_metrics(run_dir)]
    assert len(curve) == 5 and curve[0] < 1.0 and curve[-1] == final.mean_score

"""The command line of the behaviour analysis (M8, spec section 3.7):
``python -m snake_rl.analysis.behaviour`` with ``--player`` (a baseline and
a ``--config``, a run directory and ``--checkpoint``) and ``--exp`` (the
per-seed analyses plus the aggregate outputs), and its usage errors.

Run directories are hand-written (``tests.test_m8_behaviour.write_dqn_run``:
a validated ``config.yaml`` and tiny ``DQNAgent`` files); the experiment
YAML of a baseline is ``configs/smoke_tabular.yaml`` on a 6x6 board with a
starvation limit of 20. Every ``episodes.csv`` the CLI writes is compared
with ``episode_table(rollouts(...))`` of the network the flags select.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from snake_rl.agents.baselines import make_policy
from snake_rl.agents.dqn.agent import DQNAgent
from snake_rl.analysis.aggregate import mean_ci95
from snake_rl.analysis.behaviour import DEFAULT_EPISODES, DEFAULT_SEED, episode_table, main, rollouts
from snake_rl.config import config_to_dict, load_config
from snake_rl.evaluate import env_config_to_dict
from snake_rl.utils import REPO_ROOT
from tests.helpers import EnvConfig
from tests.test_m8_behaviour import SUMMARY_KEYS, TABLE_COLUMNS, write_dqn_run
from tests.test_plots import png_size

KEY_COLUMNS = ("seed", "score", "steps", "length", "cause", "truncated", "cut", "tail_without_food", "period", "loop_steps")


# ------------------------------------------------------------------- helpers
def write_experiment_yaml(path: Path, W: int = 6, starve_limit: int = 20) -> Path:
    """An experiment YAML whose ``env`` block is a 6x6 board with H = 20 (validated by ``load_config``)."""
    cfg = load_config(REPO_ROOT / "configs" / "smoke_tabular.yaml", [f"env.W={W}", f"env.starve_limit={starve_limit}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config_to_dict(cfg), sort_keys=False), encoding="utf-8")
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def printed_path(capsys) -> Path:
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert lines, "the CLI prints the path of summary.json"
    return Path(lines[-1].strip())


def assert_table_matches(csv_path: Path, policy, cfg: EnvConfig, n: int, seed: int, max_steps: int | None = None) -> pd.DataFrame:
    """The ``episodes.csv`` written by the CLI is the episode table of these rollouts."""
    table = episode_table(rollouts(policy, cfg, n, seed, max_steps=max_steps), cfg.reward)
    csv = pd.read_csv(csv_path)
    assert list(csv.columns) == list(TABLE_COLUMNS) and len(csv) == n
    for column in KEY_COLUMNS:
        assert csv[column].tolist() == table[column].tolist(), column
    return table


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and not (isinstance(value, float) and math.isnan(value))


# ------------------------------------------------------------------ --player
def test_player_baseline_with_config(tmp_path, capsys):
    config = write_experiment_yaml(tmp_path / "cfg" / "exp.yaml")
    cfg = EnvConfig(W=6, starve_limit=20)
    out = tmp_path / "analysis" / "bfs"
    argv = ["--player", "bfs", "--config", str(config), "--out-dir", str(out), "--episodes", "3", "--seed", "10000",
            "--record-top", "2", "--k-last", "5"]
    assert main(argv) == 0
    assert printed_path(capsys).resolve() == (out / "summary.json").resolve()
    summary = read_json(out / "summary.json")
    assert SUMMARY_KEYS <= set(summary)
    assert summary["player"] == "bfs" and summary["env"] == env_config_to_dict(cfg)
    assert summary["seed"] == 10_000 and summary["n_episodes"] == 3 and summary["max_steps"] is None
    assert summary["checkpoint"] is None and summary["source"] == "bfs"
    assert isinstance(summary["git"], str) and summary["git"]
    assert summary["hacking_cases"] == [] and not list(out.glob("hack_*"))
    table = assert_table_matches(out / "episodes.csv", make_policy("bfs"), cfg, 3, 10_000)
    assert set(table["cause"]) <= {"wall", "self"} and (out / "last_steps.png").is_file()
    assert summary["frac_wall"] + summary["frac_self"] == 1.0 and summary["loop_fraction"] == 0.0
    heat = np.load(out / "heatmap.npy")
    assert heat.shape == (6, 6) and abs(heat.sum() - 1.0) < 1e-12
    assert png_size(out / "heatmap.png")[0] > 0
    assert sorted(p.name for p in out.iterdir()) == ["episodes.csv", "heatmap.npy", "heatmap.png", "last_steps.png", "summary.json"]
    # --checkpoint final is the one value the flag may take with a baseline
    out2 = tmp_path / "analysis" / "bfs_final"
    assert main(["--player", "bfs", "--config", str(config), "--out-dir", str(out2), "--episodes", "2",
                 "--checkpoint", "final"]) == 0
    second = read_json(out2 / "summary.json")
    assert second["checkpoint"] is None and second["n_episodes"] == 2 and second["source"] == "bfs"


def test_player_defaults_are_100_episodes_from_seed_10000(tmp_path, capsys):
    config = write_experiment_yaml(tmp_path / "exp.yaml")
    out = tmp_path / "defaults"
    assert main(["--player", "bfs", "--config", str(config), "--out-dir", str(out)]) == 0
    summary = read_json(out / "summary.json")
    assert summary["n_episodes"] == DEFAULT_EPISODES == 100 and summary["seed"] == DEFAULT_SEED == 10_000
    assert summary["max_steps"] is None and len(summary["hacking_cases"]) <= 3  # --record-top defaults to 3
    csv = pd.read_csv(out / "episodes.csv")
    assert csv["seed"].tolist() == list(range(10_000, 10_100))


def test_player_run_directory_with_checkpoint(tmp_path, capsys):
    run = write_dqn_run(tmp_path / "runs" / "dqn_W6", 0, checkpoint_steps=(100, 500, 1000))
    cfg = load_config(run / "config.yaml").env
    assert cfg == EnvConfig(W=6, starve_limit=20)
    out = tmp_path / "analysis" / "early"
    assert main(["--player", str(run), "--checkpoint", "early", "--out-dir", str(out), "--episodes", "3"]) == 0
    assert printed_path(capsys).resolve() == (out / "summary.json").resolve()
    summary = read_json(out / "summary.json")
    assert summary["checkpoint"] == 100 and summary["source"] == str(run)
    assert summary["player"] == "dqn" and summary["env"] == env_config_to_dict(cfg)
    assert summary["seed"] == 10_000 and summary["n_episodes"] == 3
    early = DQNAgent.load(run / "checkpoints" / "step_00000100.pt", device="cpu")
    assert_table_matches(out / "episodes.csv", early, cfg, 3, 10_000)
    assert pd.read_csv(out / "episodes.csv")["q_max_mean"].notna().all()  # a DQN has action values
    assert is_number(summary["q_max_mean"])
    # the default checkpoint is the final network, model.pt, at the run's budget
    out = tmp_path / "analysis" / "final"
    assert main(["--player", str(run), "--out-dir", str(out), "--episodes", "3", "--max-steps", "7"]) == 0
    summary = read_json(out / "summary.json")
    assert summary["checkpoint"] == 1000 and summary["max_steps"] == 7
    final = DQNAgent.load(run / "model.pt", device="cpu")
    table = assert_table_matches(out / "episodes.csv", final, cfg, 3, 10_000, max_steps=7)
    assert (table["steps"] <= 7).all()
    # a decimal step picks that checkpoint
    out = tmp_path / "analysis" / "step500"
    assert main(["--player", str(run), "--checkpoint", "500", "--out-dir", str(out), "--episodes", "2"]) == 0
    assert read_json(out / "summary.json")["checkpoint"] == 500
    middle = DQNAgent.load(run / "checkpoints" / "step_00000500.pt", device="cpu")
    assert_table_matches(out / "episodes.csv", middle, cfg, 2, 10_000)
    # the same network as a bare *.pt spec (its run config supplies the environment)
    out = tmp_path / "analysis" / "pt"
    assert main(["--player", str(run / "checkpoints" / "step_00000500.pt"), "--out-dir", str(out), "--episodes", "2"]) == 0
    summary = read_json(out / "summary.json")
    assert summary["source"] == str(run / "checkpoints" / "step_00000500.pt") and summary["env"] == env_config_to_dict(cfg)
    assert_table_matches(out / "episodes.csv", middle, cfg, 2, 10_000)


# --------------------------------------------------------------------- --exp
def test_exp_analyses_every_seed_and_aggregates(tmp_path, capsys):
    exp = tmp_path / "runs" / "dqn_W6"
    for seed in (1, 0):  # creation order is not seed order
        write_dqn_run(exp, seed, checkpoint_steps=(100, 500, 1000))
    (exp / "figures").mkdir()  # not a seed directory: no config.yaml
    (exp / "notes.txt").write_text("not a directory", encoding="utf-8")
    cfg = load_config(exp / "0" / "config.yaml").env
    out = tmp_path / "analysis" / "dqn_W6"
    assert main(["--exp", str(exp), "--out-dir", str(out), "--episodes", "3", "--seed", "10000"]) == 0
    assert printed_path(capsys).resolve() == (out / "summary.json").resolve()
    per_seed = {}
    for seed in (0, 1):
        seed_dir = out / str(seed)
        s = read_json(seed_dir / "summary.json")
        per_seed[seed] = s
        assert SUMMARY_KEYS <= set(s) and s["checkpoint"] == 1000 and "source" in s
        assert Path(s["run"]).resolve() == (exp / str(seed)).resolve()
        assert s["n_episodes"] == 3 and s["seed"] == 10_000 and s["player"] == "dqn"
        agent = DQNAgent.load(exp / str(seed) / "model.pt", device="cpu")
        assert_table_matches(seed_dir / "episodes.csv", agent, cfg, 3, 10_000)
        assert (seed_dir / "heatmap.npy").is_file() and png_size(seed_dir / "heatmap.png")[0] > 0
    assert not (out / "figures").exists()
    heats = [np.load(out / str(seed) / "heatmap.npy") for seed in (0, 1)]
    mean_heat = np.load(out / "heatmap_mean.npy")
    np.testing.assert_allclose(mean_heat, (heats[0] + heats[1]) / 2, rtol=1e-12)
    assert abs(mean_heat.sum() - 1.0) < 1e-12 and png_size(out / "heatmap_mean.png")[0] > 0
    agg = read_json(out / "summary.json")
    assert agg["n_seeds"] == 2 and agg["seeds"] == [0, 1] and agg["exp"] == "dqn_W6" and agg["checkpoint"] == "final"
    for key in SUMMARY_KEYS:
        assert f"{key}_mean" in agg and f"{key}_ci95" in agg, key
        values = [per_seed[seed][key] for seed in (0, 1)]
        if all(is_number(v) for v in values):
            mean, ci = mean_ci95(values)
            assert agg[f"{key}_mean"] == pytest.approx(mean, rel=1e-12), key
            assert agg[f"{key}_ci95"] == pytest.approx(ci, rel=1e-12), key
    assert agg["n_episodes_mean"] == 3.0 and agg["n_episodes_ci95"] == 0.0
    assert agg["steps_mean_mean"] == pytest.approx((per_seed[0]["steps_mean"] + per_seed[1]["steps_mean"]) / 2, rel=1e-12)
    # --checkpoint picks the same checkpoint of every seed and is recorded in the aggregate
    out = tmp_path / "analysis" / "dqn_W6_mid"
    assert main(["--exp", str(exp), "--checkpoint", "mid", "--out-dir", str(out), "--episodes", "2"]) == 0
    agg = read_json(out / "summary.json")
    assert agg["checkpoint"] == "mid" and agg["seeds"] == [0, 1]
    for seed in (0, 1):
        assert read_json(out / str(seed) / "summary.json")["checkpoint"] == 500
        middle = DQNAgent.load(exp / str(seed) / "checkpoints" / "step_00000500.pt", device="cpu")
        assert_table_matches(out / str(seed) / "episodes.csv", middle, cfg, 2, 10_000)


# ------------------------------------------------------------- usage errors
def test_usage_errors_exit_2(tmp_path, capsys):
    config = write_experiment_yaml(tmp_path / "exp.yaml")
    run = write_dqn_run(tmp_path / "runs" / "dqn_W6", 0, checkpoint_steps=(100, 500, 1000))
    out = tmp_path / "out"
    for argv in (
        ["--player", "bfs", "--exp", str(run.parent), "--config", str(config), "--out-dir", str(out)],  # both
        ["--out-dir", str(out)],  # neither
        ["--player", "bfs", "--config", str(config), "--checkpoint", "early", "--out-dir", str(out)],  # alias, baseline
        ["--player", "bfs", "--config", str(config), "--checkpoint", "100", "--out-dir", str(out)],  # step, baseline
        ["--player", "bfs", "--out-dir", str(out)],  # a baseline needs --config
        ["--player", str(run), "--config", str(config), "--out-dir", str(out)],  # a run has its own config
        ["--exp", str(run.parent), "--config", str(config), "--out-dir", str(out)],  # so has an experiment
        ["--player", str(run / "model.pt"), "--checkpoint", "early", "--out-dir", str(out)],  # a *.pt is one network
        ["--player", "bfs", "--config", str(config)],  # --out-dir is required
    ):
        with pytest.raises(SystemExit) as exc:
            main(argv)
        assert exc.value.code == 2, argv
        assert capsys.readouterr().err != "", argv
    assert not out.exists()


# ------------------------------------------------------------ entry point
def test_module_entry_point(tmp_path):
    config = write_experiment_yaml(tmp_path / "exp.yaml")
    out = tmp_path / "entry"
    proc = subprocess.run(
        [sys.executable, "-m", "snake_rl.analysis.behaviour", "--player", "bfs", "--config", str(config),
         "--out-dir", str(out), "--episodes", "2", "--seed", "10000"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert Path(lines[-1].strip()).resolve() == (out / "summary.json").resolve()
    summary = read_json(out / "summary.json")
    assert summary["n_episodes"] == 2 and summary["player"] == "bfs" and summary["seed"] == 10_000
    proc = subprocess.run(
        [sys.executable, "-m", "snake_rl.analysis.behaviour", "--player", "bfs", "--out-dir", str(tmp_path / "x")],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 2 and proc.stderr.strip() != "" and not (tmp_path / "x").exists()

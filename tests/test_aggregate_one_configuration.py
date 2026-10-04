"""An experiment directory must hold seeds of ONE configuration: a run started with another
``--set`` under the same name is not a replicate and must not be averaged into the row."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import yaml

from snake_rl.analysis.aggregate import AggregateWarning, learning_curve, load_runs, main, summarise_experiment
from tests.test_aggregate import write_run

EPISODES = [(1, 5, "wall"), (3, 5, "self")]
CURVE = [(0, 0.0), (100, 2.0)]


def edit_config(run_dir: Path, dotted: str, value) -> None:
    path = run_dir / "config.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    node = config
    *parents, last = dotted.split(".")
    for part in parents:
        node = node[part]
    node[last] = value
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def two_runs(tmp_path: Path) -> Path:
    exp = tmp_path / "exp"
    write_run(exp, 0, EPISODES, CURVE)
    write_run(exp, 1, EPISODES, CURVE)
    return exp


@pytest.mark.parametrize(
    "key, value",
    [("agent.gamma", 0.5), ("env.W", 6), ("env.reward.variant", "R1"), ("train.total_env_steps", 400),
     ("train.eval_episodes", 10), ("agent.kind", "dqn")],
)
def test_runs_of_two_configurations_are_refused(tmp_path, key, value):
    exp = two_runs(tmp_path)
    edit_config(exp / "1", key, value)
    for call in (load_runs, summarise_experiment, learning_curve):
        with pytest.raises(ValueError, match=key.replace(".", r"\.")):
            call(exp)
    with pytest.raises(ValueError, match="do not share one configuration"):
        main(["--exp", str(exp)])


def test_run_block_and_exp_name_may_differ(tmp_path):
    exp = two_runs(tmp_path)
    edit_config(exp / "1", "run.git", "fffffff+dirty")
    edit_config(exp / "1", "exp", "renamed")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert [r.seed for r in load_runs(exp)] == [0, 1]


def test_unfinished_run_of_another_configuration_only_warns(tmp_path):
    exp = two_runs(tmp_path)
    write_run(exp, 2, EPISODES, CURVE, finished=False)
    edit_config(exp / "2", "agent.gamma", 0.5)
    with pytest.warns(AggregateWarning, match="unfinished"):
        assert [r.seed for r in load_runs(exp)] == [0, 1]

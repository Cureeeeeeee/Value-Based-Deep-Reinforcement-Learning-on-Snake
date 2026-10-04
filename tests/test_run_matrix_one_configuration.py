"""New seeds under an existing experiment name must have its configuration."""

from __future__ import annotations

import pytest

from snake_rl.config import load_config
from snake_rl.train import train
from tests.test_run_matrix import BASE, CONFIG, TINY, rm, run_main


@pytest.fixture
def two_seeds(tmp_path):
    cfg = load_config(CONFIG, TINY)
    for seed in (0, 1):
        train(cfg, seed, tmp_path, quiet=True)
    return tmp_path


def test_new_seeds_of_another_configuration_are_refused(two_seeds):
    with pytest.raises(ValueError, match=r"another configuration \(agent\.gamma: 0\.99 there, 0\.5 requested\)"):
        rm.plan_runs(CONFIG, [2, 3], sets=[*TINY, "agent.gamma=0.5"], runs_dir=two_seeds)
    code, out, err = run_main("--config", CONFIG, "--seeds", 2, 3, "--runs-dir", two_seeds, "--set", *TINY, "agent.gamma=0.5")
    assert code == 2 and out == "" and "agent.gamma" in err
    assert sorted(p.name for p in (two_seeds / BASE).iterdir()) == ["0", "1"]  # nothing was started


def test_new_seeds_of_the_same_configuration_are_planned(two_seeds):
    runs = rm.plan_runs(CONFIG, [0, 2, 3], sets=TINY, runs_dir=two_seeds)
    assert [(run.seed, run.status) for run in runs] == [(0, rm.DONE), (2, rm.PENDING), (3, rm.PENDING)]


def test_another_name_keeps_the_configurations_apart(two_seeds):
    runs = rm.plan_runs(CONFIG, [2, 3], sets=[*TINY, "agent.gamma=0.5"], exp_prefix="other", runs_dir=two_seeds)
    assert [run.status for run in runs] == [rm.PENDING, rm.PENDING]

"""``scripts/run_matrix.py`` ignores ``agent.keep_checkpoints`` when it compares
a planned run with a recorded one (M10, spec section 2).

The matrix of the report is launched with ``--set agent.keep_checkpoints=false``
(24-47 MB of weights per run otherwise), while the seeds trained before M10
kept their checkpoints. Whether weight files are kept decides nothing about
the training, so such a run is ``done`` (or ``incomplete`` / ``pending``) and
not a configuration clash; any other difference is still refused.

The recorded run directories are written by hand from
``configs/dqn_A_8x8.yaml`` (``config_to_dict(load_config(...))`` plus a
``run`` block), so no process is started and nothing is trained.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from snake_rl.analysis.aggregate import IGNORED_SETTINGS
from snake_rl.config import config_to_dict, load_config
from tests.test_aggregate import FIVE_SEEDS, make_result
from tests.test_run_matrix import REPO, rm, run_main

CONFIG = REPO / "configs" / "dqn_A_8x8.yaml"
EXP = "dqn_A_W8"  # exp of CONFIG
NO_CKPT = "agent.keep_checkpoints=false"


def write_recorded_run(runs_dir: Path, seed: int, overrides=(), *, finished: bool = True,
                       drop_keep_checkpoints: bool = False) -> Path:
    """A run directory the way ``snake_rl.train`` records it, trained with ``overrides``."""
    config = config_to_dict(load_config(CONFIG, list(overrides)))
    if drop_keep_checkpoints:  # a run trained before M7, when the key did not exist
        del config["agent"]["keep_checkpoints"]
    config["run"] = {"seed": seed, "env_seed": 100_000 * (seed + 1), "git": "abc1234"}
    run_dir = runs_dir / config["exp"] / str(seed)
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (run_dir / "metrics.csv").write_text("env_steps,eval_mean_score\n0,0.0\n", encoding="utf-8")
    if finished:
        make_result(FIVE_SEEDS[1], policy="dqn").save(run_dir / "final_eval.json")
    return run_dir


def test_the_relaxed_key_is_the_one_of_aggregate():
    assert IGNORED_SETTINGS == ("agent.keep_checkpoints",)  # the tuple both the script and aggregate ignore


@pytest.mark.parametrize("recorded, requested", [("true", "false"), ("false", "true"), ("absent", "false")])
def test_existing_status_ignores_keep_checkpoints(tmp_path, recorded, requested):
    runs_dir = tmp_path / "runs"
    kept = [] if recorded == "absent" else [f"agent.keep_checkpoints={recorded}"]
    done = write_recorded_run(runs_dir, 0, kept, drop_keep_checkpoints=recorded == "absent")
    incomplete = write_recorded_run(runs_dir, 1, kept, finished=False, drop_keep_checkpoints=recorded == "absent")
    cfg = load_config(CONFIG, [f"agent.keep_checkpoints={requested}"])
    assert rm.existing_status(done, cfg) == rm.DONE
    assert rm.existing_status(incomplete, cfg) == rm.INCOMPLETE
    assert rm.existing_status(runs_dir / EXP / "2", cfg) == rm.PENDING


def test_a_planned_batch_without_checkpoints_continues_an_experiment_that_kept_them(tmp_path):
    runs_dir = tmp_path / "runs"
    write_recorded_run(runs_dir, 0)  # keep_checkpoints: true, the value of the config file
    write_recorded_run(runs_dir, 2, finished=False)
    runs = rm.plan_runs(CONFIG, [0, 1, 2], sets=[NO_CKPT], runs_dir=runs_dir)
    assert [(run.exp, run.seed, run.status) for run in runs] == [
        (EXP, 0, rm.DONE), (EXP, 1, rm.PENDING), (EXP, 2, rm.INCOMPLETE)]
    pending = runs[1].command
    assert pending[pending.index("--set") + 1] == NO_CKPT  # the new seed is trained without checkpoints
    assert runs[1].run_dir == runs_dir / EXP / "1"


def test_the_other_direction_is_fine_too(tmp_path):
    runs_dir = tmp_path / "runs"
    write_recorded_run(runs_dir, 3, [NO_CKPT])
    runs = rm.plan_runs(CONFIG, [3, 4], runs_dir=runs_dir)  # the config file keeps checkpoints
    assert [(run.seed, run.status) for run in runs] == [(3, rm.DONE), (4, rm.PENDING)]


def test_any_other_difference_still_stops_the_batch(tmp_path):
    runs_dir = tmp_path / "runs"
    write_recorded_run(runs_dir, 0)
    with pytest.raises(ValueError, match=r"agent\.gamma: 0\.9 there, 0\.5 requested") as exc:
        rm.plan_runs(CONFIG, [0, 1], sets=[NO_CKPT, "agent.gamma=0.5"], runs_dir=runs_dir)
    assert "keep_checkpoints" not in str(exc.value)  # the ignored key is not reported either
    with pytest.raises(ValueError, match=r"agent\.double"):
        rm.plan_runs(CONFIG, [0, 1], sets=["agent.double=true"], runs_dir=runs_dir)
    cfg = load_config(CONFIG, ["train.total_env_steps=256000", NO_CKPT])
    with pytest.raises(ValueError, match=r"train\.total_env_steps"):
        rm.existing_status(runs_dir / EXP / "0", cfg)


def test_dry_run_lists_the_done_seed_and_prints_the_missing_one(tmp_path):
    runs_dir = tmp_path / "runs"
    write_recorded_run(runs_dir, 0)
    code, out, err = run_main("--config", CONFIG, "--seeds", 0, 1, "--runs-dir", runs_dir, "--set", NO_CKPT, "--dry-run")
    assert code == 0
    assert out.splitlines() == [rm.format_command(
        rm.plan_runs(CONFIG, [1], sets=[NO_CKPT], runs_dir=runs_dir)[0].command)]
    assert [line.rsplit(": ", 1)[1] for line in err.splitlines()] == ["done"]
    assert NO_CKPT in out and "--exp dqn_A_W8" in out

    code, out, err = run_main("--config", CONFIG, "--seeds", 0, 1, "--runs-dir", runs_dir, "--set", NO_CKPT,
                              "agent.gamma=0.5", "--dry-run")
    assert code == 2 and out == ""
    assert "agent.gamma" in err and "--exp-prefix" in err

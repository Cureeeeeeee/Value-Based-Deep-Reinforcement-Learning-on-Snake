"""Cross-seed aggregation (``snake_rl/analysis/aggregate.py``).

Run directories are synthetic and live in ``tmp_path``; they are written in
the file formats of ``snake_rl.train``: ``config.yaml`` with a ``run`` block,
``final_eval.json`` through ``EvalResult.save`` and ``metrics.csv`` with the
exact header of the training loop.

The five-seed fixture has UNEQUAL episode counts (6, 2, 2, 2, 2) and seed 10
next to seed 2, so that every wrong way of aggregating gives a different
number from the right one:

=====================================  =========  ==========================
quantity                               right      wrong alternatives
=====================================  =========  ==========================
mean score (mean of 5 seed means)      6.0        72 / 14 = 5.143 (pooled)
95 % half-width (t, ddof=1, 5 means)   3.6195     2.5555 (z = 1.96),
                                                  3.2374 (ddof = 0),
                                                  2.063 (14 pooled episodes)
median (pooled episodes)               4.5        5.0 (of seed means),
                                                  5.8 (mean of seed medians)
max (pooled episodes)                  12         10 (of seed means)
mean steps (pooled episodes)           500 / 14   46.0 (mean of seed means)
fraction wall (pooled episodes)        7 / 14     0.3 (mean of seed fractions)
seed order                             0 1 2 3 10 0 1 10 2 3 (lexicographic)
=====================================  =========  ==========================
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from snake_rl.agents.baselines import make_policy
from snake_rl.analysis.aggregate import (
    CURVE_COLUMNS,
    AggregateWarning,
    RunResult,
    learning_curve,
    load_runs,
    main,
    mean_ci95,
    results_table,
    summarise_baseline,
    summarise_experiment,
    t_critical_95,
)
from snake_rl.env import EnvConfig
from snake_rl.evaluate import EpisodeRecord, EvalResult, env_config_to_dict, evaluate
from snake_rl.utils import REPO_ROOT

#: Exact header written by the training loop.
METRICS_HEADER = (
    "env_steps,vec_steps,epsilon,eval_mean_score,eval_ci95,eval_median_score,eval_max_score,"
    "eval_mean_steps,eval_frac_wall,eval_frac_self,eval_frac_starve,eval_frac_win,"
    "train_episodes,train_mean_score,mean_abs_td,cells_visited,wall_time_s"
)
ENV = env_config_to_dict(EnvConfig(W=8))
EVAL_SEED = 10_000

#: Keys every summary row must have (``n_aborted`` / ``aborted``: M6, D5).
SUMMARY_KEYS = {
    "label", "n_seeds", "seeds", "n_episodes", "mean_score", "ci95", "ci_kind", "per_seed_mean_scores",
    "median_score", "max_score", "mean_steps", "cause_fractions", "n_aborted", "aborted",
}

#: seed -> final-evaluation episodes as (score, steps, cause). Seed means 3, 5, 4, 8, 10.
FIVE_SEEDS = {
    0: [(0, 10, "wall"), (1, 10, "wall"), (1, 10, "wall"), (2, 10, "wall"), (2, 10, "wall"), (12, 10, "wall")],
    1: [(4, 20, "wall"), (6, 40, "self")],
    2: [(3, 30, "self"), (5, 30, "self")],
    3: [(7, 50, "self"), (9, 70, "starve")],
    10: [(9, 100, "starve"), (11, 100, "win")],
}
#: seed -> eval_mean_score at env_steps 0, 100, 200.
FIVE_CURVES = {0: [0.0, 1.0, 3.0], 1: [0.0, 2.0, 5.0], 2: [0.0, 3.0, 4.0], 3: [0.0, 4.0, 8.0], 10: [0.0, 5.0, 10.0]}
STEPS = [0, 100, 200]

T4 = 2.776  # two-sided 95 % Student t, 4 degrees of freedom
CI_FIVE = T4 * math.sqrt(8.5 / 5)  # seed means 3 5 4 8 10: sum of squares 34, variance 34 / 4


# ------------------------------------------------------------------- builders
def make_result(episodes, policy: str = "tabular_q") -> EvalResult:
    """``episodes`` are ``(score, steps, cause)`` triples; episode i gets seed EVAL_SEED + i."""
    records = [
        EpisodeRecord(seed=EVAL_SEED + i, score=score, steps=steps, length=score + 3, cause=cause)
        for i, (score, steps, cause) in enumerate(episodes)
    ]
    return EvalResult(policy=policy, env=ENV, seed=EVAL_SEED, episodes=records, git="abc1234", wall_time_s=0.5)


def metrics_csv(curve, extra: bool = False) -> str:
    """``curve`` is a list of ``(env_steps, eval_mean_score)``; the other columns are decoys."""
    lines = [METRICS_HEADER + (",loss,q_mean" if extra else "")]
    for k, (env_steps, mean) in enumerate(curve):
        first = k == 0
        cells = [
            env_steps, env_steps // 64, 1.0 if first else 0.05, mean, 0.5, mean + 100.0, 99, 50.0,
            0.25, 0.25, 0.25, 0.25, 0 if first else 12, "" if first else 1.5, "" if first else 0.01, 10 * k, 0.1 * k,
        ]
        if extra:
            cells += ["" if first else 0.3, 1.25]
        lines.append(",".join(str(c) for c in cells))
    return "\n".join(lines) + "\n"


def write_run(
    exp_dir: Path,
    seed: int,
    episodes,
    curve,
    *,
    finished: bool = True,
    dir_name: str | None = None,
    extra: bool = False,
    run_block: bool = True,
) -> Path:
    """Write one run directory the way ``snake_rl.train`` does."""
    run_dir = exp_dir / (dir_name if dir_name is not None else str(seed))
    run_dir.mkdir(parents=True)
    config = {
        "exp": exp_dir.name,
        "env": ENV,
        "agent": {"kind": "tabular_q", "alpha": 0.1, "gamma": 0.99, "eps_start": 1.0, "eps_end": 0.05,
                  "eps_decay_fraction": 0.2, "q_init": 0.0},
        "train": {"n_envs": 64, "total_env_steps": 200, "eval_every": 100, "eval_episodes": 2,
                  "eval_seed": EVAL_SEED},
    }
    if run_block:
        config["run"] = {
            "seed": seed, "env_seed": 100_000 * (seed + 1), "git": "abc1234", "started": "2026-10-04T12:00:00+11:00",
            "python": "3.13.7", "numpy": "2.5.3", "platform": "test",
            "argv": ["--config", "x.yaml", "--seed", str(seed)],
        }
    if extra:  # what a later milestone may add
        config["agent"] = {"kind": "dqn", "lr": 1e-3, "double": True, "hidden": [128, 128]}
        config["train"]["batch_size"] = 256
        config["notes"] = "extra top-level key"
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (run_dir / "metrics.csv").write_text(metrics_csv(curve, extra), encoding="utf-8")
    if finished:
        make_result(episodes).save(run_dir / "final_eval.json")
    return run_dir


def make_five_seed_exp(root: Path, name: str = "five_seeds", extra: bool = False) -> Path:
    exp_dir = root / name
    for seed in (10, 3, 0, 2, 1):  # creation order is not seed order
        write_run(exp_dir, seed, FIVE_SEEDS[seed], list(zip(STEPS, FIVE_CURVES[seed])), extra=extra)
    return exp_dir


def write_aborted_run(exp_dir: Path, seed: int, curve, *, dir_name: str | None = None) -> Path:
    """An aborted DQN run the way ``snake_rl.train`` leaves it (M6, D4):
    ``config.yaml``, the ``metrics.csv`` rows written before the abort and
    ``aborted.json``; no ``final_eval.json``."""
    run_dir = write_run(exp_dir, seed, [], curve, finished=False, dir_name=dir_name)
    marker = {"error": "non-finite DQN update at gradient step 120: loss nan, gradient norm nan",
              "env_steps": 1472, "vec_steps": 184, "grad_steps": 119}
    (run_dir / "aborted.json").write_text(json.dumps(marker), encoding="utf-8")
    return run_dir


@pytest.fixture
def five(tmp_path: Path) -> Path:
    return make_five_seed_exp(tmp_path)


# ------------------------------------------------------------- t_critical_95
@pytest.mark.parametrize(
    "df, expected",
    [(1, 12.706), (2, 4.303), (4, 2.776), (9, 2.262), (30, 2.042), (31, 1.96), (120, 1.96), (10**6, 1.96)],
)
def test_t_critical_95_known_values(df, expected):
    assert t_critical_95(df) == expected


@pytest.mark.parametrize("df", [0, -1, -30])
def test_t_critical_95_rejects_df_below_one(df):
    with pytest.raises(ValueError, match="df"):
        t_critical_95(df)


def test_t_critical_95_accepts_numpy_integers():
    assert t_critical_95(np.int64(4)) == 2.776


_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(200)


def _prob_abs_t_below(t: float, df: int) -> float:
    """P(|T_df| <= t): Gauss-Legendre quadrature of the Student t density on [0, t]."""
    x = 0.5 * t * (_GL_NODES + 1.0)
    log_c = math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - 0.5 * math.log(df * math.pi)
    pdf = np.exp(log_c - (df + 1) / 2 * np.log1p(x * x / df))
    return float(t * (_GL_WEIGHTS * pdf).sum())


@pytest.mark.parametrize("df", range(1, 31))
def test_t_table_entry_is_the_rounded_975_quantile(df):
    """Every table entry, checked against the t distribution itself (no scipy)."""
    t = t_critical_95(df)
    assert _prob_abs_t_below(t - 0.0005, df) <= 0.95 <= _prob_abs_t_below(t + 0.0005, df)


def test_t_table_decreases_towards_the_normal_value():
    table = [t_critical_95(df) for df in range(1, 31)]
    assert all(a > b for a, b in zip(table, table[1:]))
    assert table[-1] > 1.96


# ------------------------------------------------------------------ mean_ci95
def test_mean_ci95_five_values_by_hand():
    # 3 5 4 8 10: mean 6; deviations -3 -1 -2 2 4; SS 34; s^2 = 34/4 = 8.5; se = sqrt(8.5/5)
    mean, half = mean_ci95([3, 5, 4, 8, 10])
    assert mean == 6.0
    assert half == pytest.approx(2.776 * math.sqrt(1.7), rel=1e-12)
    assert half == pytest.approx(3.6195, abs=5e-5)
    # 1 2 3 4 5: mean 3; SS 10; s^2 = 2.5; se = sqrt(0.5)
    mean, half = mean_ci95(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert mean == 3.0
    assert half == pytest.approx(1.9629, abs=5e-5)


def test_mean_ci95_two_values_uses_df_one():
    # 1 3: mean 2; s = sqrt(2); se = 1; half-width = t(1) = 12.706
    mean, half = mean_ci95((1, 3))
    assert mean == 2.0
    assert half == pytest.approx(12.706, rel=1e-12)


def test_mean_ci95_is_order_and_container_independent():
    expected = mean_ci95([3, 5, 4, 8, 10])
    assert mean_ci95(pd.Series([10, 8, 4, 5, 3])) == pytest.approx(expected)
    assert all(isinstance(v, float) for v in expected)


def test_mean_ci95_single_value_has_no_interval():
    mean, half = mean_ci95([7.5])
    assert mean == 7.5
    assert math.isnan(half)


def test_mean_ci95_empty_raises():
    with pytest.raises(ValueError):
        mean_ci95([])


# ------------------------------------------------------------------ load_runs
def test_load_runs_fields(five):
    runs = load_runs(five)
    assert len(runs) == 5 and all(isinstance(r, RunResult) for r in runs)
    r = runs[1]
    assert r.run_dir == five / "1" and r.seed == 1
    assert r.config["run"]["seed"] == 1 and r.config["agent"]["kind"] == "tabular_q"
    assert isinstance(r.final, EvalResult)
    assert [(e.score, e.steps, e.cause) for e in r.final.episodes] == FIVE_SEEDS[1]
    assert isinstance(r.metrics, pd.DataFrame)
    assert list(r.metrics.columns) == METRICS_HEADER.split(",")
    assert r.metrics["env_steps"].tolist() == STEPS
    assert r.metrics["eval_mean_score"].tolist() == FIVE_CURVES[1]
    # empty cells of the first row (no training episode, no update yet) read as missing
    assert math.isnan(r.metrics["train_mean_score"][0]) and math.isnan(r.metrics["mean_abs_td"][0])


def test_load_runs_sorts_by_seed_numerically(five):
    assert [r.seed for r in load_runs(five)] == [0, 1, 2, 3, 10]  # not 0 1 10 2 3


def test_load_runs_seed_comes_from_the_run_block(tmp_path):
    exp = tmp_path / "exp"
    write_run(exp, 7, FIVE_SEEDS[1], [(0, 0.0)], dir_name="a_renamed_directory")
    write_run(exp, 2, FIVE_SEEDS[2], [(0, 0.0)], dir_name="z")
    runs = load_runs(exp)
    assert [r.seed for r in runs] == [2, 7]
    assert [r.run_dir.name for r in runs] == ["z", "a_renamed_directory"]


def test_load_runs_without_run_block_raises(tmp_path):
    exp = tmp_path / "exp"
    write_run(exp, 0, FIVE_SEEDS[1], [(0, 0.0)], run_block=False)
    with pytest.raises(ValueError, match="run.seed"):
        load_runs(exp)


def test_load_runs_skips_unfinished_and_warns(five):
    write_run(five, 4, FIVE_SEEDS[1], [(0, 0.0), (100, 50.0)], finished=False)  # config.yaml only
    make_result(FIVE_SEEDS[1]).save(five / "5" / "final_eval.json")  # final_eval.json only
    (five / "plots").mkdir()  # not a run directory: no warning
    (five / "notes.txt").write_text("not a directory", encoding="utf-8")
    with pytest.warns(AggregateWarning, match=r"skipped 2 unfinished run\(s\): 4, 5$") as record:
        runs = load_runs(five)
    assert len(record) == 1
    assert [r.seed for r in runs] == [0, 1, 2, 3, 10]


def test_load_runs_is_silent_when_all_runs_are_finished(five):
    with warnings.catch_warnings():
        warnings.simplefilter("error", AggregateWarning)
        load_runs(five)


def test_load_runs_raises_when_no_run_is_finished(tmp_path):
    exp = tmp_path / "exp"
    write_run(exp, 0, FIVE_SEEDS[1], [(0, 0.0)], finished=False)
    with pytest.warns(AggregateWarning), pytest.raises(ValueError, match="no finished run"):
        load_runs(exp)
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no finished run"):
        load_runs(tmp_path / "empty")
    with pytest.raises(ValueError):
        load_runs(tmp_path / "does_not_exist")


def test_load_runs_rejects_a_seed_counted_twice(five):
    write_run(five, 3, FIVE_SEEDS[3], [(0, 0.0)], dir_name="3_copy")
    with pytest.raises(ValueError, match="duplicate seed"):
        load_runs(five)


# -------------------------------------------------------- aborted runs (M6, D5)
def test_load_runs_skips_aborted_runs_and_reports_them_apart_from_unfinished_ones(five):
    write_aborted_run(five, 4, [(0, 0.0), (100, 50.0)])  # config.yaml + metrics.csv + aborted.json
    write_run(five, 5, FIVE_SEEDS[1], [(0, 0.0), (100, 50.0)], finished=False)  # config.yaml only
    make_result(FIVE_SEEDS[1]).save(five / "6" / "final_eval.json")  # final_eval.json only
    write_aborted_run(five, 7, [(0, 0.0)])
    with pytest.warns(AggregateWarning) as record:
        runs = load_runs(five)
    assert [r.seed for r in runs] == [0, 1, 2, 3, 10]
    messages = sorted(str(w.message) for w in record)
    assert len(messages) == 2, messages
    assert messages[0] == f"{five}: skipped 2 aborted run(s) (aborted.json): 4, 7"
    assert messages[1] == f"{five}: skipped 2 unfinished run(s): 5, 6"  # the aborted ones are not unfinished


def test_load_runs_warns_once_about_aborted_runs_when_nothing_is_unfinished(five):
    write_aborted_run(five, 4, [(0, 0.0)])
    with pytest.warns(AggregateWarning, match=r"skipped 1 aborted run\(s\) \(aborted\.json\): 4$") as record:
        runs = load_runs(five)
    assert len(record) == 1 and "unfinished" not in str(record[0].message)
    assert [r.seed for r in runs] == [0, 1, 2, 3, 10]


def test_load_runs_raises_when_every_run_is_aborted(tmp_path):
    exp = tmp_path / "exp"
    write_aborted_run(exp, 0, [(0, 0.0)])
    write_aborted_run(exp, 1, [(0, 0.0), (100, 1.0)])
    with pytest.warns(AggregateWarning, match="aborted"), pytest.raises(ValueError, match="no finished run"):
        load_runs(exp)
    with pytest.warns(AggregateWarning), pytest.raises(ValueError, match="no finished run"):
        summarise_experiment(exp)
    with pytest.warns(AggregateWarning), pytest.raises(ValueError, match="no finished run"):
        learning_curve(exp)


def test_summaries_count_the_aborted_runs(five, tmp_path):
    reference = summarise_experiment(five)
    assert reference["n_aborted"] == 0 and reference["aborted"] == []
    write_aborted_run(five, 4, [(0, 0.0)])
    write_aborted_run(five, 7, [(0, 0.0)], dir_name="7_crashed")
    write_run(five, 8, FIVE_SEEDS[1], [(0, 0.0)], finished=False)  # unfinished: not aborted
    with pytest.warns(AggregateWarning):
        s = summarise_experiment(five)
    assert s["n_aborted"] == 2 and s["aborted"] == ["4", "7_crashed"]
    assert type(s["n_aborted"]) is int and all(isinstance(name, str) for name in s["aborted"])
    assert {k: v for k, v in s.items() if k not in ("n_aborted", "aborted")} == {
        k: v for k, v in reference.items() if k not in ("n_aborted", "aborted")}  # the statistics are untouched
    assert SUMMARY_KEYS <= s.keys()
    baseline = make_result(FIVE_SEEDS[1], "bfs").save(tmp_path / "bfs.json")
    row = summarise_baseline(baseline)
    assert row["n_aborted"] == 0 and row["aborted"] == [] and row.keys() == s.keys()


def test_results_table_seeds_cell_shows_the_aborted_runs_and_is_unchanged_without_them(five):
    plain = results_table([summarise_experiment(five)])
    write_aborted_run(five, 4, [(0, 0.0)])
    with pytest.warns(AggregateWarning):
        with_aborted = results_table([summarise_experiment(five)])
    lines = with_aborted.splitlines()
    assert lines[:2] == plain.splitlines()[:2]  # the header is the same
    assert lines[2] == ("| five_seeds | 5 (+1 aborted) | 2-6 | 6.00 +- 3.62 | seeds | 4.5 | 12 | 35.7 | "
                        "0.50 | 0.29 | 0.14 | 0.07 |")
    assert plain.splitlines()[2] == ("| five_seeds | 5 | 2-6 | 6.00 +- 3.62 | seeds | 4.5 | 12 | 35.7 | "
                                     "0.50 | 0.29 | 0.14 | 0.07 |")
    # Hand-built rows: without the key n_aborted = 0; with n_aborted = 0 the cell is the bare count.
    row = {
        "label": "tabular_q", "n_seeds": 5, "seeds": [0, 1, 2, 3, 4], "n_episodes": 100, "mean_score": 12.34,
        "ci95": 0.8, "ci_kind": "seeds", "per_seed_mean_scores": [11.5, 12.0, 12.2, 12.9, 13.1],
        "median_score": 12.5, "max_score": 31, "mean_steps": 140.2,
        "cause_fractions": {"wall": 0.25, "self": 0.7, "starve": 0.05, "win": 0.0},
    }
    bare = "| tabular_q | 5 | 100 | 12.34 +- 0.80 | seeds | 12.5 | 31 | 140.2 | 0.25 | 0.70 | 0.05 | 0.00 |"
    assert results_table([row]).splitlines()[2] == bare
    assert results_table([{**row, "n_aborted": 0, "aborted": []}]).splitlines()[2] == bare
    assert results_table([{**row, "n_aborted": 3, "aborted": ["5", "6", "7"]}]).splitlines()[2] == bare.replace(
        "| 5 |", "| 5 (+3 aborted) |")


def test_learning_curve_skips_aborted_runs(five):
    reference = learning_curve(five)
    write_aborted_run(five, 4, [(0, 77.0), (100, 77.0), (200, 77.0), (300, 77.0)])
    with pytest.warns(AggregateWarning, match="aborted") as record:
        curve = learning_curve(five)
    assert len(record) == 1  # the aborted run neither contributes values nor removes steps
    pd.testing.assert_frame_equal(curve, reference)
    pd.testing.assert_frame_equal(curve, expected_five_curve(), rtol=1e-12)


def test_cli_table_shows_aborted_runs(five, tmp_path, capsys):
    write_aborted_run(five, 4, [(0, 0.0)])
    out = tmp_path / "table.md"
    with pytest.warns(AggregateWarning, match="aborted"):
        assert main(["--exp", str(five), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "| five_seeds | 5 (+1 aborted) | 2-6 |" in printed
    assert out.read_text(encoding="utf-8") == printed


def without_episodes(path: Path) -> Path:
    """Rewrite an ``EvalResult`` JSON file so that its episode list is empty."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["episodes"] = []
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_a_result_file_without_episodes_is_an_error_not_a_nan_row(five, tmp_path):
    """An empty episode list has no mean score. It must stop the aggregation;
    as a row it would put nan into the table and into the mean over seeds."""
    without_episodes(five / "2" / "final_eval.json")
    with pytest.raises(ValueError, match="no episodes"):
        load_runs(five)
    with pytest.raises(ValueError, match="no episodes"):
        summarise_experiment(five)
    with pytest.raises(ValueError, match="no episodes"):
        learning_curve(five)
    baseline = without_episodes(make_result(FIVE_SEEDS[1], "bfs").save(tmp_path / "bfs.json"))
    with pytest.raises(ValueError, match="no episodes"):
        summarise_baseline(baseline)


# ------------------------------------------------------- summarise_experiment
def test_summarise_experiment_five_seeds_by_hand(five):
    s = summarise_experiment(five)
    assert SUMMARY_KEYS <= s.keys()
    assert s["label"] == "five_seeds"
    assert s["n_seeds"] == 5
    assert s["seeds"] == [0, 1, 2, 3, 10]
    assert s["per_seed_mean_scores"] == [3.0, 5.0, 4.0, 8.0, 10.0]
    # mean of the seed means, not of the 14 pooled episodes (72 / 14 = 5.14)
    assert s["mean_score"] == pytest.approx(6.0, rel=1e-12)
    # Student t over the 5 seed means: 2.776 * sqrt(8.5 / 5)
    assert s["ci95"] == pytest.approx(CI_FIVE, rel=1e-12)
    assert s["ci95"] == pytest.approx(3.6195, abs=5e-5)
    assert s["ci_kind"] == "seeds"
    # pooled episodes, sorted: 0 1 1 2 2 3 | 4 5 | 6 7 9 9 11 12 -> median (4 + 5) / 2
    assert s["median_score"] == 4.5
    assert s["max_score"] == 12 and isinstance(s["max_score"], int)
    # steps: 6 * 10 + 60 + 60 + 120 + 200 = 500 over 14 episodes
    assert s["mean_steps"] == pytest.approx(500 / 14, rel=1e-12)
    # causes: wall 6 + 1, self 1 + 2 + 1, starve 1 + 1, win 1
    assert s["cause_fractions"] == pytest.approx({"wall": 7 / 14, "self": 4 / 14, "starve": 2 / 14, "win": 1 / 14})
    assert list(s["cause_fractions"]) == ["wall", "self", "starve", "win"]
    assert sum(s["cause_fractions"].values()) == pytest.approx(1.0, rel=1e-12)
    # seeds disagree on the episode count: no single per-seed number
    assert s["n_episodes"] is None
    assert s["per_seed_n_episodes"] == [6, 2, 2, 2, 2]


def test_summarise_experiment_equal_episode_counts_and_label(tmp_path):
    exp = tmp_path / "two"
    write_run(exp, 0, [(1, 5, "wall"), (2, 5, "wall"), (3, 5, "self")], [(0, 0.0)])
    write_run(exp, 1, [(5, 9, "self"), (5, 9, "self"), (8, 9, "win")], [(0, 0.0)])
    s = summarise_experiment(exp, label="tabular Q, 8x8")
    assert s["label"] == "tabular Q, 8x8"
    assert s["n_episodes"] == 3 and s["per_seed_n_episodes"] == [3, 3]
    # seed means 2 and 6: mean 4; s = sqrt(8); se = 2; t(1) = 12.706
    assert s["mean_score"] == pytest.approx(4.0)
    assert s["ci95"] == pytest.approx(2 * 12.706, rel=1e-12)
    assert s["median_score"] == 4.0 and s["max_score"] == 8 and s["mean_steps"] == pytest.approx(7.0)


def test_summarise_experiment_default_label_is_the_directory_name(five, monkeypatch):
    assert summarise_experiment(str(five) + "/")["label"] == "five_seeds"
    monkeypatch.chdir(five)
    assert summarise_experiment(".")["label"] == "five_seeds"
    assert summarise_experiment(Path("../five_seeds"))["label"] == "five_seeds"


def test_summarise_experiment_ignores_unfinished_runs(five):
    reference = summarise_experiment(five)
    write_run(five, 4, [(50, 500, "win")] * 2, [(0, 0.0)], finished=False)
    with pytest.warns(AggregateWarning, match="unfinished"):
        assert summarise_experiment(five) == reference


def test_summarise_experiment_single_seed_has_no_interval(tmp_path):
    exp = tmp_path / "one_seed"
    write_run(exp, 0, FIVE_SEEDS[1], [(0, 0.0), (100, 5.0)])
    s = summarise_experiment(exp)
    assert s["n_seeds"] == 1 and s["seeds"] == [0] and s["mean_score"] == 5.0
    assert math.isnan(s["ci95"]) and s["ci_kind"] == "seeds"
    row = results_table([s]).splitlines()[2]
    assert row == "| one_seed | 1 | 2 | 5.00 +- n/a | seeds | 5.0 | 6 | 30.0 | 0.50 | 0.50 | 0.00 | 0.00 |"
    assert "nan" not in results_table([s]).lower()


def test_extra_metrics_columns_and_config_keys_are_tolerated(tmp_path):
    plain = make_five_seed_exp(tmp_path / "a")
    extra = make_five_seed_exp(tmp_path / "b", extra=True)
    runs = load_runs(extra)
    assert runs[0].config["notes"] == "extra top-level key" and runs[0].config["agent"]["kind"] == "dqn"
    assert list(runs[0].metrics.columns) == METRICS_HEADER.split(",") + ["loss", "q_mean"]
    assert summarise_experiment(extra) == summarise_experiment(plain)
    pd.testing.assert_frame_equal(learning_curve(extra), learning_curve(plain))


# --------------------------------------------------------- summarise_baseline
def test_summarise_baseline_equals_eval_result_summary(tmp_path):
    # a real file: the scalar evaluation protocol on a small board
    result = evaluate(make_policy("random", seed=123), EnvConfig(W=6), n_episodes=30, seed=123)
    path = result.save(tmp_path / "baselines" / "random_W6_relative.json")
    expected = EvalResult.load(path).summary()
    row = summarise_baseline(path)
    assert row.keys() == summarise_experiment(make_five_seed_exp(tmp_path)).keys()
    assert SUMMARY_KEYS <= row.keys()
    for key in ("n_episodes", "mean_score", "ci95", "median_score", "max_score", "mean_steps", "cause_fractions"):
        assert row[key] == expected[key], key
    assert row["ci_kind"] == "episodes"
    assert row["n_seeds"] == 1 and row["seeds"] == []
    assert row["per_seed_mean_scores"] == [expected["mean_score"]]
    assert row["label"] == "random_W6_relative"
    assert summarise_baseline(path, label="random")["label"] == "random"


def test_summarise_baseline_recomputes_from_the_episode_records(tmp_path):
    path = make_result([(2, 10, "wall"), (4, 20, "self"), (6, 30, "self"), (8, 40, "starve")], "bfs").save(
        tmp_path / "bfs.json"
    )
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["summary"]["mean_score"] = 999.0  # a hand-edited summary block must not reach a table
    doc["summary"]["ci95"] = 0.0
    path.write_text(json.dumps(doc), encoding="utf-8")
    row = summarise_baseline(path)
    # 2 4 6 8: mean 5; SS 20; s = sqrt(20 / 3); normal approximation 1.96 * s / sqrt(4)
    assert row["mean_score"] == 5.0
    assert row["ci95"] == pytest.approx(1.96 * math.sqrt(20 / 3) / 2, rel=1e-12)
    assert row["ci95"] == pytest.approx(2.5303, abs=5e-5)
    assert row["cause_fractions"] == {"wall": 0.25, "self": 0.5, "starve": 0.25, "win": 0.0}


def test_seed_ci_and_episode_ci_are_different_things(tmp_path):
    """The same four numbers 2 4 6 8 as seed means and as episode scores."""
    exp = tmp_path / "learner"
    for seed, m in enumerate([2, 4, 6, 8]):
        write_run(exp, seed, [(m, 10, "wall"), (m, 10, "wall")], [(0, 0.0)])
    baseline = make_result([(m, 10, "wall") for m in (2, 4, 6, 8)], "bfs").save(tmp_path / "bfs.json")
    learner, player = summarise_experiment(exp), summarise_baseline(baseline)
    assert learner["mean_score"] == player["mean_score"] == 5.0
    s = math.sqrt(20 / 3)
    assert learner["ci95"] == pytest.approx(3.182 * s / 2, rel=1e-12)  # Student t, 3 df, over 4 seeds
    assert player["ci95"] == pytest.approx(1.96 * s / 2, rel=1e-12)  # normal approximation over 4 episodes
    lines = results_table([player, learner]).splitlines()
    assert lines[0].split(" | ")[4] == "CI over"
    assert lines[2] == "| bfs | 1 | 4 | 5.00 +- 2.53 | episodes | 5.0 | 8 | 10.0 | 1.00 | 0.00 | 0.00 | 0.00 |"
    assert lines[3] == "| learner | 4 | 2 | 5.00 +- 4.11 | seeds | 5.0 | 8 | 10.0 | 1.00 | 0.00 | 0.00 | 0.00 |"


# -------------------------------------------------------------- results_table
def test_results_table_exact_format():
    rows = [
        {
            "label": "bfs", "n_seeds": 1, "seeds": [], "n_episodes": 100, "mean_score": 30.61, "ci95": 1.5959,
            "ci_kind": "episodes", "per_seed_mean_scores": [30.61], "median_score": 31.0, "max_score": 58,
            "mean_steps": 255.79, "cause_fractions": {"wall": 0.16, "self": 0.84, "starve": 0.0, "win": 0.0},
        },
        {
            "label": "tabular_q", "n_seeds": 5, "seeds": [0, 1, 2, 3, 4], "n_episodes": 100, "mean_score": 12.34,
            "ci95": 0.8, "ci_kind": "seeds", "per_seed_mean_scores": [11.5, 12.0, 12.2, 12.9, 13.1],
            "median_score": 12.5, "max_score": 31, "mean_steps": 140.2,
            "cause_fractions": {"wall": 0.25, "self": 0.7, "starve": 0.05, "win": 0.0},
        },
        {
            "label": "one_seed", "n_seeds": 1, "seeds": [0], "n_episodes": 100, "mean_score": 9.0,
            "ci95": float("nan"), "ci_kind": "seeds", "per_seed_mean_scores": [9.0], "median_score": 9.0,
            "max_score": 20, "mean_steps": 99.0,
            "cause_fractions": {"wall": 0.5, "self": 0.5, "starve": 0.0, "win": 0.0},
        },
    ]
    assert results_table(rows) == (
        "| label | seeds | episodes | score mean +- 95% CI | CI over | median | max | mean steps "
        "| wall | self | starve | win |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
        "| bfs | 1 | 100 | 30.61 +- 1.60 | episodes | 31.0 | 58 | 255.8 | 0.16 | 0.84 | 0.00 | 0.00 |\n"
        "| tabular_q | 5 | 100 | 12.34 +- 0.80 | seeds | 12.5 | 31 | 140.2 | 0.25 | 0.70 | 0.05 | 0.00 |\n"
        "| one_seed | 1 | 100 | 9.00 +- n/a | seeds | 9.0 | 20 | 99.0 | 0.50 | 0.50 | 0.00 | 0.00 |"
    )


def test_results_table_of_the_five_seed_experiment(five):
    lines = results_table([summarise_experiment(five)]).splitlines()
    assert len(lines) == 3
    # unequal episode counts are shown as a range
    assert lines[2] == "| five_seeds | 5 | 2-6 | 6.00 +- 3.62 | seeds | 4.5 | 12 | 35.7 | 0.50 | 0.29 | 0.14 | 0.07 |"


def test_results_table_without_rows_is_just_the_header():
    lines = results_table([]).splitlines()
    assert len(lines) == 2 and lines[0].count("|") == lines[1].count("|") == 13


def test_results_table_escapes_a_pipe_in_a_label(five):
    """A label is free text; an unescaped ``|`` would add a column to its row."""
    line = results_table([summarise_experiment(five, label="alpha=0.1 | gamma=0.9")]).splitlines()[2]
    assert line.startswith("| alpha=0.1 \\| gamma=0.9 | 5 | 2-6 | 6.00 +- 3.62 | seeds | ")
    assert line.replace("\\|", "").count("|") == 13


# ------------------------------------------------------------- learning_curve
def expected_five_curve() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "env_steps": np.array(STEPS, dtype=np.int64),
            "n_seeds": 5,
            "mean": [0.0, 3.0, 6.0],
            # step 100: 1 2 3 4 5 -> s^2 = 2.5; step 200: 3 5 4 8 10 -> s^2 = 8.5
            "ci95": [0.0, T4 * math.sqrt(2.5 / 5), T4 * math.sqrt(8.5 / 5)],
            "min": [0.0, 1.0, 3.0],
            "max": [0.0, 5.0, 10.0],
        }
    )


def test_learning_curve_five_seeds_by_hand(five):
    curve = learning_curve(five)
    assert tuple(curve.columns) == CURVE_COLUMNS == ("env_steps", "n_seeds", "mean", "ci95", "min", "max")
    pd.testing.assert_frame_equal(curve, expected_five_curve(), rtol=1e-12)
    assert curve["ci95"].tolist() == pytest.approx([0.0, 1.9629, 3.6195], abs=5e-5)


def test_learning_curve_uses_only_steps_present_in_all_finished_runs(tmp_path):
    exp = tmp_path / "ragged"
    for seed in FIVE_SEEDS:
        curve = list(zip(STEPS, FIVE_CURVES[seed]))
        if seed == 0:
            curve.append((300, 99.0))  # one run evaluated longer
        if seed == 3:
            curve.insert(1, (50, 42.0))  # one run evaluated more often
        write_run(exp, seed, FIVE_SEEDS[seed], curve)
    # an unfinished run must neither contribute values nor remove steps
    write_run(exp, 4, FIVE_SEEDS[1], [(0, 77.0)], finished=False)
    with pytest.warns(AggregateWarning) as record:
        curve = learning_curve(exp)
    messages = sorted(str(w.message) for w in record)
    assert len(messages) == 2
    assert "2 env_steps value(s) are not present in all 5 finished runs" in messages[0]
    assert "unfinished run(s): 4" in messages[1]
    pd.testing.assert_frame_equal(curve, expected_five_curve(), rtol=1e-12)


def test_learning_curve_single_seed(tmp_path):
    exp = tmp_path / "one_seed"
    write_run(exp, 0, FIVE_SEEDS[1], [(0, 0.5), (100, 2.5), (200, 5.0)])
    curve = learning_curve(exp)
    assert curve["env_steps"].tolist() == STEPS and curve["n_seeds"].tolist() == [1, 1, 1]
    assert curve["mean"].tolist() == curve["min"].tolist() == curve["max"].tolist() == [0.5, 2.5, 5.0]
    assert curve["ci95"].isna().all()


def test_learning_curve_needs_the_curve_columns(tmp_path):
    exp = tmp_path / "broken"
    run_dir = write_run(exp, 0, FIVE_SEEDS[1], [(0, 0.0)])
    (run_dir / "metrics.csv").write_text("env_steps,loss\n0,1.0\n", encoding="utf-8")
    assert summarise_experiment(exp)["mean_score"] == 5.0  # the table does not need the curve
    with pytest.raises(ValueError, match="eval_mean_score"):
        learning_curve(exp)


def test_learning_curve_rejects_a_metrics_file_with_repeated_env_steps(tmp_path):
    """Two rows for one ``env_steps`` (two runs appended to one file) have no
    single value to average: an error that names the file, not a guess."""
    exp = tmp_path / "appended_twice"
    write_run(exp, 0, FIVE_SEEDS[1], [(0, 0.0), (100, 2.0), (100, 3.0)])
    write_run(exp, 1, FIVE_SEEDS[2], [(0, 0.0), (100, 4.0)])
    with pytest.raises(ValueError, match=r"metrics\.csv: duplicate env_steps"):
        learning_curve(exp)
    assert summarise_experiment(exp)["per_seed_mean_scores"] == [5.0, 4.0]  # the table does not read the curve


# ------------------------------------------------ real run directories (seam)
def test_aggregate_reads_what_train_writes(tmp_path, monkeypatch):
    """Every fixture above is a hand-written copy of the run-directory format.
    This is the real thing: two seeds trained by ``snake_rl.train``, then
    aggregated, with every number recomputed here from the files on disk."""
    from snake_rl import evaluate as evaluate_module
    from snake_rl import train as train_module
    from snake_rl.config import load_config

    monkeypatch.setattr(train_module, "git_hash", lambda: "testhash")  # two git processes per call
    monkeypatch.setattr(evaluate_module, "git_hash", lambda: "testhash")
    cfg = load_config(
        REPO_ROOT / "configs" / "smoke_tabular.yaml",
        ["exp=seam", "env.starve_limit=20", "train.n_envs=8", "train.total_env_steps=2400", "train.eval_every=800",
         "train.eval_episodes=6"],
    )
    seeds = (2, 11)  # written in the other order; "11" sorts before "2" as text
    for seed in reversed(seeds):
        train_module.train(cfg, seed, tmp_path, quiet=True)
    exp = tmp_path / "seam"
    assert (exp / "2" / "metrics.csv").read_text(encoding="utf-8").splitlines()[0] == METRICS_HEADER

    runs = load_runs(exp)
    assert [r.seed for r in runs] == [2, 11] and [r.run_dir.name for r in runs] == ["2", "11"]
    assert [r.config["run"]["env_seed"] for r in runs] == [300_000, 1_200_000]
    assert all(r.final.policy == "tabular_q" and r.final.seed == EVAL_SEED for r in runs)

    episodes = {s: json.loads((exp / str(s) / "final_eval.json").read_text(encoding="utf-8"))["episodes"] for s in seeds}
    assert all(len(episodes[s]) == 6 for s in seeds)
    means = [sum(e["score"] for e in episodes[s]) / 6 for s in seeds]
    assert means[0] != means[1]
    pooled = episodes[2] + episodes[11]

    s = summarise_experiment(exp)
    assert s["label"] == "seam" and s["seeds"] == [2, 11] and s["n_episodes"] == 6 and s["ci_kind"] == "seeds"
    assert s["per_seed_mean_scores"] == means
    assert s["mean_score"] == pytest.approx(sum(means) / 2, rel=1e-12)
    # two seeds: sd = |a - b| / sqrt(2), so t(1) * sd / sqrt(2) = 12.706 * |a - b| / 2
    assert s["ci95"] == pytest.approx(12.706 * abs(means[0] - means[1]) / 2, rel=1e-12)
    assert s["max_score"] == max(e["score"] for e in pooled)
    assert s["mean_steps"] == pytest.approx(sum(e["steps"] for e in pooled) / 12, rel=1e-12)
    for cause in ("wall", "self", "starve", "win"):
        assert s["cause_fractions"][cause] == sum(e["cause"] == cause for e in pooled) / 12

    curve = learning_curve(exp)
    assert curve["env_steps"].tolist() == [0, 800, 1600, 2400] and curve["n_seeds"].tolist() == [2] * 4
    assert curve["mean"].iloc[0] == 0.0  # the untrained table scores nothing on either seed
    # The last evaluation of the curve is the final evaluation of the table.
    assert curve["mean"].iloc[-1] == pytest.approx(s["mean_score"], rel=1e-12)
    assert curve["ci95"].iloc[-1] == pytest.approx(s["ci95"], rel=1e-12)
    assert (curve["min"].iloc[-1], curve["max"].iloc[-1]) == (min(means), max(means))


# ------------------------------------------------------------------------ CLI
def write_baselines(root: Path) -> tuple[Path, Path]:
    bfs = make_result([(20, 200, "self"), (30, 300, "self"), (40, 250, "wall")], "bfs")
    rnd = make_result([(0, 12, "wall"), (1, 30, "wall"), (0, 9, "self")], "random")
    return bfs.save(root / "bl" / "bfs_W8.json"), rnd.save(root / "bl" / "random_W8.json")


def test_cli_writes_table_and_curves(tmp_path, capsys):
    bfs, rnd = write_baselines(tmp_path)
    five = make_five_seed_exp(tmp_path / "runs")
    two = tmp_path / "runs" / "two_seeds"
    write_run(two, 0, [(1, 5, "wall"), (3, 5, "self")], [(0, 0.0), (100, 2.0)])
    write_run(two, 1, [(5, 9, "self"), (7, 9, "win")], [(0, 1.0), (100, 6.0)])
    out = tmp_path / "tables" / "nested" / "part_c.md"
    curves_dir = tmp_path / "tables" / "curves" / "csv"

    argv = ["--baseline", str(bfs), str(rnd), "--exp", str(five), str(two), "--out", str(out),
            "--curves-dir", str(curves_dir)]
    assert main(argv) == 0

    # baselines first in the order given, then the experiments
    table = results_table(
        [summarise_baseline(bfs), summarise_baseline(rnd), summarise_experiment(five), summarise_experiment(two)]
    )
    assert [line.split(" | ")[0] for line in table.splitlines()[2:]] == [
        "| bfs_W8", "| random_W8", "| five_seeds", "| two_seeds",
    ]
    assert capsys.readouterr().out == table + "\n"
    assert out.read_bytes() == (table + "\n").encode("utf-8")
    assert sorted(p.name for p in curves_dir.iterdir()) == ["five_seeds_curve.csv", "two_seeds_curve.csv"]
    pd.testing.assert_frame_equal(pd.read_csv(curves_dir / "five_seeds_curve.csv"), learning_curve(five), rtol=1e-12)
    pd.testing.assert_frame_equal(pd.read_csv(curves_dir / "two_seeds_curve.csv"), learning_curve(two), rtol=1e-12)
    assert (curves_dir / "five_seeds_curve.csv").read_text(encoding="utf-8").splitlines()[0] == ",".join(CURVE_COLUMNS)

    # regenerating overwrites the table instead of appending or failing
    assert main(argv) == 0
    assert out.read_bytes() == (table + "\n").encode("utf-8")


def test_cli_flags_can_be_repeated_and_out_is_optional(tmp_path, capsys):
    bfs, rnd = write_baselines(tmp_path)
    assert main(["--baseline", str(rnd), "--baseline", str(bfs)]) == 0
    assert capsys.readouterr().out == results_table([summarise_baseline(rnd), summarise_baseline(bfs)]) + "\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bl"]  # nothing else was written


def test_cli_without_inputs_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
    assert "nothing to aggregate" in capsys.readouterr().err


def test_cli_refuses_two_experiments_whose_curve_files_would_collide(tmp_path, capsys):
    a = make_five_seed_exp(tmp_path / "a", name="same")
    b = make_five_seed_exp(tmp_path / "b", name="same")
    with pytest.raises(SystemExit) as exc:
        main(["--exp", str(a), str(b), "--curves-dir", str(tmp_path / "curves")])
    assert exc.value.code == 2
    assert not (tmp_path / "curves").exists()


def test_cli_fails_loudly_when_an_experiment_has_no_finished_run(tmp_path):
    bfs, _ = write_baselines(tmp_path)
    exp = tmp_path / "running"
    write_run(exp, 0, FIVE_SEEDS[1], [(0, 0.0)], finished=False)
    out = tmp_path / "table.md"
    with pytest.warns(AggregateWarning), pytest.raises(ValueError, match="no finished run"):
        main(["--baseline", str(bfs), "--exp", str(exp), "--out", str(out)])
    assert not out.exists()  # no partial table


def test_module_entry_point(five):
    proc = subprocess.run(
        [sys.executable, "-m", "snake_rl.analysis.aggregate", "--exp", str(five)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == results_table([summarise_experiment(five)])

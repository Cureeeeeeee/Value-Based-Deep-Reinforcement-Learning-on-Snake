"""``aggregate.value_curve`` (M7, spec D8): the overestimation curve across
seeds, from the value columns of ``metrics.csv`` or from the ``values.csv``
of ``scripts/backfill_values.py``.

Synthetic DQN run directories in ``tmp_path``, written in the file formats
of ``snake_rl.train`` (the M7 ``metrics.csv`` header; the M6 header for runs
trained before M7) and of the backfill script. Three seeds over env_steps
0, 100, 200 with hand-picked numbers::

    seed  value_visited   return_visited     (value_start = visited + 10, return_start = visited + 1)
    0     0  1  3         0  0.5  1
    1     0  2  5         0  1    2
    2     0  3  4         0  1.5  6

so that at step 200: values 3 5 4 (mean 4, variance 1), returns 1 2 6 (mean
3, variance 7) and per-seed gaps 2 3 -2 (mean 1, variance 7) - the gap
interval comes from the per-seed differences, not from the two other
intervals (``sqrt(1 + 7) != sqrt(7)``). Student t with 2 degrees of freedom:
4.303.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from snake_rl.analysis.aggregate import VALUE_CURVE_COLUMNS, AggregateWarning, value_curve
from snake_rl.env import EnvConfig
from snake_rl.evaluate import env_config_to_dict
from snake_rl.utils import REPO_ROOT
from tests.test_aggregate import EVAL_SEED, FIVE_SEEDS, make_result, write_run

#: The M7 header of a DQN metrics.csv (spec D6) and the M6 one (a run trained before M7).
DQN_HEADER = (
    "env_steps,vec_steps,grad_steps,epsilon,eval_mean_score,eval_ci95,eval_median_score,eval_max_score,"
    "eval_mean_steps,eval_frac_wall,eval_frac_self,eval_frac_starve,eval_frac_win,eval_value_start,eval_return_start,"
    "eval_value_visited,eval_return_visited,train_episodes,train_mean_score,train_truncated,loss,mean_q,mean_target,"
    "grad_norm,buffer_size,time_env_s,time_act_s,time_update_s,time_eval_s,wall_time_s"
)
DQN_HEADER_M6 = DQN_HEADER.replace("eval_value_start,eval_return_start,eval_value_visited,eval_return_visited,", "")
#: The columns of values.csv (spec D7).
VALUES_HEADER = "env_steps,eval_mean_score,eval_value_start,eval_return_start,eval_value_visited,eval_return_visited"
VALUE_COLUMNS = ("eval_value_start", "eval_return_start", "eval_value_visited", "eval_return_visited")
ENV = env_config_to_dict(EnvConfig(W=8))
STEPS = (0, 100, 200)
#: seed -> per step (value_visited, return_visited).
VISITED = {
    0: [(0.0, 0.0), (1.0, 0.5), (3.0, 1.0)],
    1: [(0.0, 0.0), (2.0, 1.0), (5.0, 2.0)],
    2: [(0.0, 0.0), (3.0, 1.5), (4.0, 6.0)],
}
START_SHIFT = (10.0, 1.0)
T2 = 4.303


# ------------------------------------------------------------------- builders
def points_of(seed: int, steps=STEPS, visited=None) -> list[tuple]:
    """``(env_steps, value_start, return_start, value_visited, return_visited)`` rows of one run."""
    visited = VISITED[seed] if visited is None else visited
    return [(s, vv + START_SHIFT[0], rv + START_SHIFT[1], vv, rv) for s, (vv, rv) in zip(steps, visited)]


def metrics_text(points, header: str = "m7", blank=()) -> str:
    """A DQN ``metrics.csv``; every cell but the step and value columns is a decoy. ``blank`` lists
    ``(env_steps, column)`` cells left empty (a missing value)."""
    columns = (DQN_HEADER if header == "m7" else DQN_HEADER_M6).split(",")
    lines = [",".join(columns)]
    for k, (steps, vs, rs, vv, rv) in enumerate(points):
        first = k == 0
        cells = {name: "0.1" for name in columns}
        cells.update({
            "env_steps": str(steps), "vec_steps": str(steps // 8), "grad_steps": str(2 * k),
            "epsilon": "1.0" if first else "0.05", "eval_mean_score": repr(1.5 * k), "eval_ci95": "0.5",
            "eval_median_score": "1.0", "eval_max_score": "9", "eval_mean_steps": "50.0", "eval_frac_wall": "0.25",
            "eval_frac_self": "0.25", "eval_frac_starve": "0.25", "eval_frac_win": "0.25",
            "train_episodes": "0" if first else "12", "train_mean_score": "" if first else "1.5", "train_truncated": "0",
            "loss": "" if first else "0.3", "mean_q": "" if first else "1.25", "mean_target": "" if first else "1.0",
            "grad_norm": "" if first else "0.7", "buffer_size": str(10 * k), "wall_time_s": repr(0.1 * k),
        })
        if header == "m7":
            cells.update({"eval_value_start": repr(float(vs)), "eval_return_start": repr(float(rs)),
                          "eval_value_visited": repr(float(vv)), "eval_return_visited": repr(float(rv))})
            for blank_steps, column in blank:
                if blank_steps == steps:
                    cells[column] = ""
        lines.append(",".join(cells[name] for name in columns))
    return "\n".join(lines) + "\n"


def values_text(points) -> str:
    """A ``values.csv`` as the backfill script writes it."""
    lines = [VALUES_HEADER]
    for k, (steps, vs, rs, vv, rv) in enumerate(points):
        lines.append(f"{steps},{1.5 * k!r},{float(vs)!r},{float(rs)!r},{float(vv)!r},{float(rv)!r}")
    return "\n".join(lines) + "\n"


def write_dqn_run(exp_dir: Path, seed: int, points, *, header: str = "m7", blank=(), values=None, finished: bool = True,
                  aborted: bool = False, dir_name: str | None = None) -> Path:
    """One DQN run directory the way ``snake_rl.train`` (and the backfill script) writes it."""
    run_dir = exp_dir / (dir_name if dir_name is not None else str(seed))
    run_dir.mkdir(parents=True)
    config = {
        "exp": exp_dir.name,
        "env": ENV,
        "agent": {"kind": "dqn", "gamma": 0.9, "lr": 2.5e-4, "hidden": [16], "double": False},
        "train": {"n_envs": 8, "total_env_steps": 200, "eval_every": 100, "eval_episodes": 2, "eval_seed": EVAL_SEED},
        "run": {"seed": seed, "env_seed": 100_000 * (seed + 1), "git": "abc1234", "started": "2026-10-20T12:00:00+11:00",
                "python": "3.13.7", "numpy": "2.5.3", "platform": "test", "argv": ["--config", "x.yaml", "--seed", str(seed)],
                "torch": "2.9.0", "device": "cpu", "cuda_device": None, "torch_threads": 1},
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (run_dir / "metrics.csv").write_text(metrics_text(points, header, blank), encoding="utf-8")
    if values is not None:
        (run_dir / "values.csv").write_text(values_text(values), encoding="utf-8")
    if finished:
        make_result(FIVE_SEEDS[1], policy="dqn").save(run_dir / "final_eval.json")
    if aborted:
        marker = {"error": "non-finite DQN update at gradient step 120: loss nan, gradient norm nan",
                  "env_steps": 1472, "vec_steps": 184, "grad_steps": 119}
        (run_dir / "aborted.json").write_text(json.dumps(marker), encoding="utf-8")
    return run_dir


def make_three_seed_exp(root: Path, name: str = "three_seeds", **kwargs) -> Path:
    exp = root / name
    for seed in (2, 0, 1):  # creation order is not seed order
        write_dqn_run(exp, seed, points_of(seed), **kwargs)
    return exp


def expected_frame(which: str = "visited") -> pd.DataFrame:
    dv, dr = START_SHIFT if which == "start" else (0.0, 0.0)
    return pd.DataFrame({
        "env_steps": np.array(STEPS, dtype=np.int64),
        "n_seeds": 3,
        "value_mean": [0.0 + dv, 2.0 + dv, 4.0 + dv],
        # step 100: 1 2 3 -> variance 1; step 200: 3 5 4 -> variance 1
        "value_ci95": [0.0, T2 * math.sqrt(1.0 / 3), T2 * math.sqrt(1.0 / 3)],
        "return_mean": [0.0 + dr, 1.0 + dr, 3.0 + dr],
        # step 100: 0.5 1 1.5 -> sd 0.5; step 200: 1 2 6 -> variance 7
        "return_ci95": [0.0, T2 * 0.5 / math.sqrt(3), T2 * math.sqrt(7.0 / 3)],
        # per-seed gaps: step 100: 0.5 1 1.5; step 200: 2 3 -2 -> variance 7
        "gap_mean": [0.0 + dv - dr, 1.0 + dv - dr, 1.0 + dv - dr],
        "gap_ci95": [0.0, T2 * 0.5 / math.sqrt(3), T2 * math.sqrt(7.0 / 3)],
    })


@pytest.fixture
def three(tmp_path: Path) -> Path:
    return make_three_seed_exp(tmp_path)


# ------------------------------------------------------------------- the frame
def test_value_curve_from_metrics_by_hand(three):
    curve = value_curve(three)
    assert isinstance(curve, pd.DataFrame)
    assert tuple(curve.columns) == VALUE_CURVE_COLUMNS == (
        "env_steps", "n_seeds", "value_mean", "value_ci95", "return_mean", "return_ci95", "gap_mean", "gap_ci95")
    assert curve["env_steps"].dtype == np.int64
    pd.testing.assert_frame_equal(curve, expected_frame(), rtol=1e-12)
    assert curve["value_ci95"].tolist() == pytest.approx([0.0, 2.4843, 2.4843], abs=5e-4)
    assert curve["gap_ci95"].tolist() == pytest.approx([0.0, 1.2422, 6.5727], abs=5e-4)
    pd.testing.assert_frame_equal(value_curve(three, "visited"), curve)
    pd.testing.assert_frame_equal(value_curve(str(three), which="visited"), curve)


def test_value_curve_start(three):
    curve = value_curve(three, which="start")
    pd.testing.assert_frame_equal(curve, expected_frame("start"), rtol=1e-12)
    visited = value_curve(three)
    assert (curve["value_mean"] - visited["value_mean"]).tolist() == pytest.approx([10.0] * 3)
    assert (curve["return_mean"] - visited["return_mean"]).tolist() == pytest.approx([1.0] * 3)
    pd.testing.assert_series_equal(curve["value_ci95"], visited["value_ci95"])


def test_gap_statistics_come_from_the_per_seed_differences(three):
    curve = value_curve(three)
    assert (curve["gap_mean"] - (curve["value_mean"] - curve["return_mean"])).abs().max() < 1e-12
    gap_200 = curve["gap_ci95"].iloc[2]
    assert gap_200 == pytest.approx(T2 * math.sqrt(7.0 / 3), rel=1e-12)
    assert gap_200 != pytest.approx(math.hypot(curve["value_ci95"].iloc[2], curve["return_ci95"].iloc[2]), rel=1e-3)
    assert gap_200 != pytest.approx(abs(curve["value_ci95"].iloc[2] - curve["return_ci95"].iloc[2]), rel=1e-3)
    assert gap_200 != pytest.approx(curve["value_ci95"].iloc[2] + curve["return_ci95"].iloc[2], rel=1e-3)


@pytest.mark.parametrize("which", ["foo", "", "Visited", "mean", "gap"])
def test_which_is_visited_or_start(three, which):
    with pytest.raises(ValueError):
        value_curve(three, which)
    with pytest.raises(ValueError):
        value_curve(three, which=which)


# ------------------------------------------------------------------- sources
def test_value_curve_from_values_csv_when_metrics_lack_the_columns(tmp_path):
    """Runs trained before M7 (the M6 header) with the values.csv of the backfill script."""
    exp = make_three_seed_exp(tmp_path, header="m6", values=None)
    for seed in VISITED:
        (exp / str(seed) / "values.csv").write_text(values_text(points_of(seed)), encoding="utf-8")
    pd.testing.assert_frame_equal(value_curve(exp), expected_frame(), rtol=1e-12)
    pd.testing.assert_frame_equal(value_curve(exp, "start"), expected_frame("start"), rtol=1e-12)


def test_metrics_wins_over_values_csv(tmp_path):
    exp = tmp_path / "both"
    for seed in VISITED:
        other = [(vv + 100.0, rv + 100.0) for vv, rv in VISITED[seed]]
        write_dqn_run(exp, seed, points_of(seed), values=points_of(seed, visited=other))
    pd.testing.assert_frame_equal(value_curve(exp), expected_frame(), rtol=1e-12)


def test_a_missing_value_cell_falls_back_to_values_csv_for_that_run(tmp_path):
    exp = tmp_path / "gappy"
    write_dqn_run(exp, 0, points_of(0), blank=((100, "eval_value_visited"),), values=points_of(0))  # from values.csv
    write_dqn_run(exp, 1, points_of(1))
    write_dqn_run(exp, 2, points_of(2), blank=((0, "eval_return_start"),), values=points_of(2))
    pd.testing.assert_frame_equal(value_curve(exp), expected_frame(), rtol=1e-12)
    # The blank start cell of seed 2 matters for which="start" only, and values.csv covers it too.
    pd.testing.assert_frame_equal(value_curve(exp, "start"), expected_frame("start"), rtol=1e-12)
    # Without values.csv the run with the blank cell has no value estimates.
    (exp / "0" / "values.csv").unlink()
    with pytest.raises(ValueError, match="no value estimates"):
        value_curve(exp)


def test_mixed_sources_across_the_runs_of_one_experiment(tmp_path):
    exp = tmp_path / "mixed"
    write_dqn_run(exp, 0, points_of(0), header="m6", values=points_of(0))
    write_dqn_run(exp, 1, points_of(1))
    write_dqn_run(exp, 2, points_of(2), header="m6", values=points_of(2))
    pd.testing.assert_frame_equal(value_curve(exp), expected_frame(), rtol=1e-12)


def test_without_either_source_is_an_error_naming_the_run_and_the_remedy(tmp_path):
    exp = tmp_path / "tabular"
    for seed in (0, 1):
        write_run(exp, seed, FIVE_SEEDS[1], [(0, 0.0), (100, 1.0)])  # a tabular metrics.csv: no value columns
    with pytest.raises(ValueError) as excinfo:
        value_curve(exp)
    message = str(excinfo.value)
    assert "no value estimates" in message and "backfill_values" in message
    assert str(exp / "0") in message or (exp / "0").as_posix() in message
    # The M7 header with blank cells and no values.csv is the same situation.
    exp = tmp_path / "blank"
    write_dqn_run(exp, 0, points_of(0), blank=((200, "eval_value_visited"),))
    with pytest.raises(ValueError, match="no value estimates"):
        value_curve(exp)
    pd.testing.assert_frame_equal(value_curve(exp, "start")[["value_mean"]], pd.DataFrame({"value_mean": [10.0, 11.0, 13.0]}))


# ------------------------------------------------------------------- the runs that count
def test_intersection_rule_and_the_warnings_of_learning_curve(tmp_path):
    exp = tmp_path / "ragged"
    write_dqn_run(exp, 0, points_of(0, steps=(0, 100, 200, 300), visited=[*VISITED[0], (9.0, 9.0)]))  # evaluated longer
    write_dqn_run(exp, 1, points_of(1))
    write_dqn_run(exp, 2, points_of(2, steps=(0, 50, 100, 200), visited=[VISITED[2][0], (7.0, 7.0), *VISITED[2][1:]]))  # more often
    write_dqn_run(exp, 3, points_of(1), finished=False)  # unfinished: neither values nor steps
    write_dqn_run(exp, 4, points_of(1), finished=False, aborted=True)
    with pytest.warns(AggregateWarning) as record:
        curve = value_curve(exp)
    messages = sorted(str(w.message) for w in record)
    assert len(messages) == 3, messages
    assert any("2 env_steps value(s) are not present in all 3 finished runs" in m for m in messages)
    assert any("aborted" in m and "4" in m for m in messages)
    assert any("unfinished" in m and "3" in m for m in messages)
    pd.testing.assert_frame_equal(curve, expected_frame(), rtol=1e-12)


def test_no_warning_when_every_run_is_finished_and_aligned(three):
    with warnings.catch_warnings():
        warnings.simplefilter("error", AggregateWarning)
        value_curve(three)


def test_single_seed_has_no_interval(tmp_path):
    exp = tmp_path / "one_seed"
    write_dqn_run(exp, 0, points_of(0))
    curve = value_curve(exp)
    assert curve["env_steps"].tolist() == list(STEPS) and curve["n_seeds"].tolist() == [1, 1, 1]
    assert curve["value_mean"].tolist() == [0.0, 1.0, 3.0] and curve["return_mean"].tolist() == [0.0, 0.5, 1.0]
    assert curve["gap_mean"].tolist() == [0.0, 0.5, 2.0]
    assert curve["value_ci95"].isna().all() and curve["return_ci95"].isna().all() and curve["gap_ci95"].isna().all()


def test_bad_directories(tmp_path):
    with pytest.raises(ValueError, match="not a directory"):
        value_curve(tmp_path / "no_such_experiment")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no finished run"):
        value_curve(tmp_path / "empty")


# ------------------------------------------------ real run directories (seam)
def test_value_curve_reads_what_train_writes(tmp_path, monkeypatch):
    """Two tiny DQN seeds trained by ``snake_rl.train``, aggregated, every number recomputed from the files."""
    from snake_rl import evaluate as evaluate_module
    from snake_rl import train as train_module
    from snake_rl.config import load_config

    monkeypatch.setattr(train_module, "git_hash", lambda: "testhash")
    monkeypatch.setattr(evaluate_module, "git_hash", lambda: "testhash")
    cfg = load_config(
        REPO_ROOT / "configs" / "smoke_8x8.yaml",
        ["exp=seam", "env.starve_limit=20", "train.n_envs=8", "train.total_env_steps=1600", "train.eval_every=800",
         "train.eval_episodes=4", "agent.hidden=[16]", "agent.batch_size=16", "agent.buffer_size=1000",
         "agent.learning_starts=1000", "agent.target_update_every=50"],
    )
    before = torch_threads()
    try:
        for seed in (1, 0):
            train_module.train(cfg, seed, tmp_path, quiet=True)
    finally:
        set_torch_threads(before)
    exp = tmp_path / "seam"
    metrics = {seed: pd.read_csv(exp / str(seed) / "metrics.csv", float_precision="round_trip") for seed in (0, 1)}
    assert list(metrics[0].columns) == DQN_HEADER.split(",")
    for which, prefix in (("visited", "visited"), ("start", "start")):
        curve = value_curve(exp, which)
        assert curve["env_steps"].tolist() == [0, 800, 1600] and curve["n_seeds"].tolist() == [2, 2, 2]
        for j, steps in enumerate((0, 800, 1600)):
            v = [float(metrics[s].loc[metrics[s]["env_steps"] == steps, f"eval_value_{prefix}"].item()) for s in (0, 1)]
            r = [float(metrics[s].loc[metrics[s]["env_steps"] == steps, f"eval_return_{prefix}"].item()) for s in (0, 1)]
            assert curve["value_mean"].iloc[j] == pytest.approx(sum(v) / 2, rel=1e-12)
            assert curve["return_mean"].iloc[j] == pytest.approx(sum(r) / 2, rel=1e-12)
            # two seeds: sd = |a - b| / sqrt(2), so t(1) * sd / sqrt(2) = 12.706 * |a - b| / 2
            assert curve["value_ci95"].iloc[j] == pytest.approx(12.706 * abs(v[0] - v[1]) / 2, rel=1e-12)
            assert curve["return_ci95"].iloc[j] == pytest.approx(12.706 * abs(r[0] - r[1]) / 2, rel=1e-12)
            assert curve["gap_mean"].iloc[j] == pytest.approx((v[0] - r[0] + v[1] - r[1]) / 2, rel=1e-12)
            assert curve["gap_ci95"].iloc[j] == pytest.approx(12.706 * abs((v[0] - r[0]) - (v[1] - r[1])) / 2, rel=1e-12)
    assert value_curve(exp, "visited")["value_mean"].tolist() != value_curve(exp, "start")["value_mean"].tolist()


def torch_threads() -> int:
    import torch

    return torch.get_num_threads()


def set_torch_threads(n: int) -> None:
    import torch

    torch.set_num_threads(n)

"""Last-steps summary, tail validation, the new table column and the
``keep_checkpoints`` relaxation of ``snake_rl/analysis/aggregate.py`` (M10,
spec sections 1.1, 1.2, 1.3 and 1.6).

Synthetic run directories from ``tests.test_aggregate``. The five-seed
fixture has rows at env_steps 0, 100, 200::

    seed 0: 0 1 3    seed 1: 0 2 5    seed 2: 0 3 4    seed 3: 0 4 8    seed 10: 0 5 10

The tail by steps is the rows with ``env_steps >= last - last_steps``
(INCLUSIVE; ``last`` = 200 here)::

    last_steps   rows selected   = last_k   per-seed means     mean   half-width (t(4) = 2.776)
    99           200             1          3 5 4 8 10         6.0    2.776 * sqrt(8.5 / 5)    = 3.6195
    100          100, 200        2          2 3.5 3.5 6 7.5    4.5    2.776 * sqrt(4.875 / 5)  = 2.7411
    199          100, 200        2          (the same)
    200          0, 100, 200     3          4/3 7/3 7/3 4 5    3.0    2.776 * sqrt(13 / 6 / 5) = 1.8274
    1000         0, 100, 200     3          (the same)

The "long" experiment has two seeds over env_steps 0, 100, 200, 300, 400::

    seed 0: 0 9 1 2 6    seed 1: 0 9 3 4 5

    last_steps   rows selected    per-seed means   mean   half-width (t(1) = 12.706)
    200          200, 300, 400    3.0  4.0         3.5    12.706 * 1.0 / 2 = 6.353
    199          300, 400         4.0  4.5         4.25   12.706 * 0.5 / 2 = 3.1765
    1            400              6.0  5.0         5.5    12.706 * 1.0 / 2 = 6.353

The DQN-shaped experiment evaluates every 12 800 steps up to 512 000 (41
rows; the score of row k is k for seed 0 and 2k for seed 1). ``last_steps =
51200`` selects the five rows 460 800 .. 512 000 (k = 36 .. 40), the rows
of ``last_k = 5``: per-seed means 38 and 76, mean 57, half-width
12.706 * 38 / 2 = 241.414; ``last_steps = 51199`` selects four rows
(k = 37 .. 40): 38.5 and 77.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import pytest
import yaml

from snake_rl.analysis.aggregate import (
    IGNORED_SETTINGS,
    AggregateWarning,
    learning_curve,
    load_runs,
    results_table,
    summarise_baseline,
    summarise_experiment,
)
from tests.test_aggregate import (
    CI_FIVE,
    FIVE_CURVES,
    FIVE_SEEDS,
    STEPS,
    T4,
    make_five_seed_exp,
    write_aborted_run,
    write_baselines,
    write_run,
)
from tests.test_aggregate_lastk import BFS_ROW, FIVE_ROW, HEAD_12, RULE_12

LASTSTEPS_KEYS = {"last_steps", "laststeps_mean_score", "laststeps_ci95", "laststeps_per_seed", "laststeps_n_rows"}
LASTK_KEYS = {"last_k", "lastk_mean_score", "lastk_ci95", "lastk_per_seed"}
T1 = 12.706  # two-sided 95 % Student t, 1 degree of freedom
DQN_STEPS = list(range(0, 512_001, 12_800))  # 41 evaluations, eval_every 12 800 (configs/dqn_A_8x8.yaml)


@pytest.fixture
def five(tmp_path: Path) -> Path:
    return make_five_seed_exp(tmp_path)


@pytest.fixture
def long(tmp_path: Path) -> Path:
    exp = tmp_path / "long"
    write_run(exp, 0, FIVE_SEEDS[1], [(0, 0.0), (100, 9.0), (200, 1.0), (300, 2.0), (400, 6.0)])
    write_run(exp, 1, FIVE_SEEDS[2], [(0, 0.0), (100, 9.0), (200, 3.0), (300, 4.0), (400, 5.0)])
    return exp


@pytest.fixture
def dqn_shaped(tmp_path: Path) -> Path:
    exp = tmp_path / "dqn_shaped"
    write_run(exp, 0, FIVE_SEEDS[1], [(s, float(k)) for k, s in enumerate(DQN_STEPS)])
    write_run(exp, 1, FIVE_SEEDS[2], [(s, 2.0 * k) for k, s in enumerate(DQN_STEPS)])
    return exp


def edit_config(run_dir: Path, dotted: str, value) -> None:
    """Set (or delete, with ``value = None`` and ``"-"`` prefix) one dotted key of a run's config.yaml."""
    path = run_dir / "config.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    delete = dotted.startswith("-")
    *parents, last = dotted.lstrip("-").split(".")
    node = config
    for part in parents:
        node = node.setdefault(part, {})
    if delete:
        node.pop(last, None)
    else:
        node[last] = value
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


# ------------------------------------------------------- summarise_experiment
def test_last_steps_100_selects_the_last_two_rows_by_hand(five):
    s = summarise_experiment(five, last_steps=100)
    assert s["last_steps"] == 100 and isinstance(s["last_steps"], int)
    assert s["laststeps_n_rows"] == [2, 2, 2, 2, 2] and all(type(n) is int for n in s["laststeps_n_rows"])
    assert s["laststeps_per_seed"] == [2.0, 3.5, 3.5, 6.0, 7.5]  # seed order 0 1 2 3 10
    assert all(isinstance(v, float) for v in s["laststeps_per_seed"])
    assert s["laststeps_mean_score"] == pytest.approx(4.5, rel=1e-12)
    # deviations -2.5 -1 -1 1.5 3: SS 19.5, s^2 = 19.5 / 4; Student t over 5 seeds
    assert s["laststeps_ci95"] == pytest.approx(T4 * math.sqrt(4.875 / 5), rel=1e-12)
    assert s["laststeps_ci95"] == pytest.approx(2.7411, abs=5e-5)


def test_last_steps_99_is_the_last_row_only(five):
    s = summarise_experiment(five, last_steps=99)
    assert s["laststeps_n_rows"] == [1] * 5
    assert s["laststeps_per_seed"] == [FIVE_CURVES[seed][-1] for seed in (0, 1, 2, 3, 10)] == [3.0, 5.0, 4.0, 8.0, 10.0]
    assert s["laststeps_mean_score"] == pytest.approx(6.0, rel=1e-12)
    assert s["laststeps_ci95"] == pytest.approx(CI_FIVE, rel=1e-12)
    assert s["laststeps_ci95"] == pytest.approx(3.6195, abs=5e-5)


def test_last_steps_1000_selects_every_row(five):
    s = summarise_experiment(five, last_steps=1000)  # the evaluation of the untrained agent included
    assert s["laststeps_n_rows"] == [3] * 5
    assert s["laststeps_per_seed"] == pytest.approx([4 / 3, 7 / 3, 7 / 3, 4.0, 5.0], rel=1e-12)
    assert s["laststeps_mean_score"] == pytest.approx(3.0, rel=1e-12)
    assert s["laststeps_ci95"] == pytest.approx(T4 * math.sqrt(13 / 6 / 5), rel=1e-12)
    assert s["laststeps_ci95"] == pytest.approx(1.8274, abs=5e-5)


def test_the_tail_boundary_is_inclusive(five):
    """``env_steps >= last - last_steps``: 200 reaches row 0, 199 does not."""
    assert summarise_experiment(five, last_steps=200)["laststeps_n_rows"] == [3] * 5
    assert summarise_experiment(five, last_steps=199)["laststeps_n_rows"] == [2] * 5
    assert summarise_experiment(five, last_steps=1)["laststeps_n_rows"] == [1] * 5
    assert summarise_experiment(five, last_steps=199)["laststeps_mean_score"] == pytest.approx(4.5, rel=1e-12)


@pytest.mark.parametrize("last_steps, last_k", [(99, 1), (100, 2), (199, 2), (200, 3), (1000, 3)])
def test_last_steps_gives_the_numbers_of_the_matching_last_k(five, last_steps, last_k):
    by_steps = summarise_experiment(five, last_steps=last_steps)
    by_k = summarise_experiment(five, last_k=last_k)
    assert by_steps["laststeps_per_seed"] == pytest.approx(by_k["lastk_per_seed"], rel=1e-12)
    assert by_steps["laststeps_mean_score"] == pytest.approx(by_k["lastk_mean_score"], rel=1e-12)
    assert by_steps["laststeps_ci95"] == pytest.approx(by_k["lastk_ci95"], rel=1e-12)
    assert by_steps["laststeps_n_rows"] == [last_k] * 5


def test_last_steps_on_the_long_experiment_by_hand(long):
    s = summarise_experiment(long, last_steps=200)  # rows 200, 300, 400 (inclusive)
    assert s["laststeps_n_rows"] == [3, 3]
    assert s["laststeps_per_seed"] == [3.0, 4.0]  # (1 + 2 + 6) / 3 and (3 + 4 + 5) / 3; the 9.0 at step 100 is not in
    assert s["laststeps_mean_score"] == pytest.approx(3.5, rel=1e-12)
    assert s["laststeps_ci95"] == pytest.approx(T1 * 1.0 / 2, rel=1e-12)
    s = summarise_experiment(long, last_steps=199)  # rows 300, 400
    assert s["laststeps_n_rows"] == [2, 2] and s["laststeps_per_seed"] == [4.0, 4.5]
    assert s["laststeps_mean_score"] == pytest.approx(4.25, rel=1e-12)
    assert s["laststeps_ci95"] == pytest.approx(T1 * 0.5 / 2, rel=1e-12)
    assert s["laststeps_ci95"] == pytest.approx(3.1765, abs=5e-5)
    s = summarise_experiment(long, last_steps=1)  # the last row always qualifies
    assert s["laststeps_n_rows"] == [1, 1] and s["laststeps_per_seed"] == [6.0, 5.0]
    assert s["laststeps_mean_score"] == pytest.approx(5.5, rel=1e-12)
    assert s["laststeps_ci95"] == pytest.approx(T1 * 1.0 / 2, rel=1e-12)


def test_dqn_worked_example_51200_steps_are_the_last_five_evaluations(dqn_shaped):
    """eval_every 12 800: last_steps = 51200 selects 460 800 .. 512 000, the rows of last_k = 5."""
    s = summarise_experiment(dqn_shaped, last_steps=51_200)
    assert s["laststeps_n_rows"] == [5, 5]
    assert s["laststeps_per_seed"] == [38.0, 76.0]  # mean of k = 36 .. 40, and twice that
    assert s["laststeps_mean_score"] == pytest.approx(57.0, rel=1e-12)
    assert s["laststeps_ci95"] == pytest.approx(T1 * 38.0 / 2, rel=1e-12)
    assert s["laststeps_ci95"] == pytest.approx(241.414, abs=5e-4)
    k5 = summarise_experiment(dqn_shaped, last_k=5)
    assert s["laststeps_per_seed"] == k5["lastk_per_seed"]
    assert s["laststeps_mean_score"] == k5["lastk_mean_score"] and s["laststeps_ci95"] == k5["lastk_ci95"]
    # one step less and the row at 460 800 drops out
    s = summarise_experiment(dqn_shaped, last_steps=51_199)
    assert s["laststeps_n_rows"] == [4, 4] and s["laststeps_per_seed"] == [38.5, 77.0]


def test_last_steps_only_adds_keys(five):
    plain = summarise_experiment(five)
    assert not LASTSTEPS_KEYS & plain.keys()
    assert not LASTK_KEYS & plain.keys()
    with_s = summarise_experiment(five, label="x", last_steps=100)
    assert with_s.keys() == plain.keys() | LASTSTEPS_KEYS
    assert not LASTK_KEYS & with_s.keys()
    assert {k: v for k, v in with_s.items() if k not in LASTSTEPS_KEYS} == {**plain, "label": "x"}
    assert summarise_experiment(five, None, None, None) == plain
    assert summarise_experiment(five, None, None, 100) == summarise_experiment(five, last_steps=100)


def test_last_steps_single_seed_has_no_interval(tmp_path):
    exp = tmp_path / "one_seed"
    write_run(exp, 0, FIVE_SEEDS[1], [(0, 0.5), (100, 2.5), (200, 5.0)])
    s = summarise_experiment(exp, last_steps=100)
    assert s["laststeps_per_seed"] == [3.75] and s["laststeps_mean_score"] == 3.75 and s["laststeps_n_rows"] == [2]
    assert math.isnan(s["laststeps_ci95"])
    table = results_table([s])
    assert table.splitlines()[0] == HEAD_12 + " last-100 steps mean +- 95% CI |"
    assert table.splitlines()[2] == (
        "| one_seed | 1 | 2 | 5.00 +- n/a | seeds | 5.0 | 6 | 30.0 | 0.50 | 0.50 | 0.00 | 0.00 | 3.75 +- n/a |"
    )
    assert "nan" not in table.lower()


@pytest.mark.parametrize("last_steps", [0, -1, -100, True, False, 100.0, "100"])
def test_last_steps_below_one_or_not_an_integer_raises(five, last_steps):
    with pytest.raises(ValueError, match="last_steps"):
        summarise_experiment(five, last_steps=last_steps)


def test_last_steps_is_checked_before_any_file_is_read(tmp_path):
    with pytest.raises(ValueError, match="last_steps"):
        summarise_experiment(tmp_path / "does_not_exist", last_steps=0)
    with pytest.raises(ValueError, match="last_steps"):
        summarise_experiment(tmp_path / "does_not_exist", last_steps=2.5)


def test_last_k_and_last_steps_are_alternatives(five, tmp_path):
    with pytest.raises(ValueError, match="last_k and last_steps are alternatives"):
        summarise_experiment(five, last_k=1, last_steps=99)
    with pytest.raises(ValueError) as exc:
        summarise_experiment(tmp_path / "does_not_exist", last_k=2, last_steps=100)  # checked before any file is read
    assert "last_k" in str(exc.value) and "last_steps" in str(exc.value)


def test_last_steps_needs_the_columns_and_finite_values(tmp_path):
    exp = tmp_path / "broken"
    run_dir = write_run(exp, 0, FIVE_SEEDS[1], [(0, 0.0)])
    metrics = run_dir / "metrics.csv"
    metrics.write_text("env_steps,loss\n0,1.0\n", encoding="utf-8")
    assert summarise_experiment(exp)["mean_score"] == 5.0  # the plain row does not read the curve
    with pytest.raises(ValueError, match="eval_mean_score") as exc:
        summarise_experiment(exp, last_steps=1)
    assert str(metrics) in str(exc.value)
    metrics.write_text("vec_steps,eval_mean_score\n0,1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="env_steps") as exc:
        summarise_experiment(exp, last_steps=1)
    assert str(metrics) in str(exc.value)
    # an empty cell inside the tail must not put nan into a table; outside the tail it does not matter
    metrics.write_text("env_steps,eval_mean_score\n0,\n100,1.0\n200,2.0\n", encoding="utf-8")
    assert summarise_experiment(exp, last_steps=100)["laststeps_per_seed"] == [1.5]
    with pytest.raises(ValueError) as exc:
        summarise_experiment(exp, last_steps=200)
    assert str(metrics) in str(exc.value)


def test_last_steps_ignores_unfinished_and_aborted_runs(five):
    reference = summarise_experiment(five, last_steps=100)
    write_run(five, 4, [(50, 500, "win")] * 2, [(0, 0.0)], finished=False)  # one row only, and not finished
    write_aborted_run(five, 7, [(0, 0.0), (100, 99.0)])
    with pytest.warns(AggregateWarning) as record:
        s = summarise_experiment(five, last_steps=100)
    assert len(record) == 2
    assert {k: v for k, v in s.items() if k not in ("n_aborted", "aborted")} == {
        k: v for k, v in reference.items() if k not in ("n_aborted", "aborted")}
    assert s["n_aborted"] == 1 and s["aborted"] == ["7"]


# --------------------------------------- the tails of the seeds must agree (1.2)
def test_last_k_tails_must_cover_the_same_env_steps(tmp_path):
    exp = tmp_path / "shifted"
    write_run(exp, 0, FIVE_SEEDS[1], [(0, 0.0), (100, 1.0), (200, 2.0)], dir_name="run_a")
    write_run(exp, 3, FIVE_SEEDS[2], [(0, 0.0), (150, 3.0), (200, 4.0)], dir_name="run_b")
    assert summarise_experiment(exp, last_k=1)["lastk_per_seed"] == [2.0, 4.0]  # both tails are [200]
    with pytest.raises(ValueError) as exc:
        summarise_experiment(exp, last_k=2)  # [100, 200] against [150, 200]
    message = str(exc.value)
    assert str(exp) in message
    assert "[100, 200]" in message and "[150, 200]" in message
    assert "run_a" in message and "run_b" in message  # the two run directories that differ
    assert summarise_experiment(exp)["per_seed_mean_scores"] == [5.0, 4.0]  # the plain row does not read the curve


def test_last_k_tails_of_different_length_are_refused(tmp_path):
    """Shorter runs used to contribute an earlier tail; now the seeds are compared over the same steps or not at all."""
    exp = tmp_path / "ragged"
    write_run(exp, 0, FIVE_SEEDS[1], [(0, 0.0), (100, 1.0), (200, 2.0)])
    write_run(exp, 1, FIVE_SEEDS[2], [(0, 0.0), (100, 1.0)], dir_name="seed_b")
    with pytest.raises(ValueError) as exc:
        summarise_experiment(exp, last_k=2)  # [100, 200] against [0, 100]
    assert str(exp) in str(exc.value) and "seed_b" in str(exc.value)
    assert "[100, 200]" in str(exc.value) and "[0, 100]" in str(exc.value)


def test_last_steps_tails_must_cover_the_same_env_steps(tmp_path):
    exp = tmp_path / "shifted"
    write_run(exp, 0, FIVE_SEEDS[1], [(0, 0.0), (100, 1.0), (200, 2.0)])
    write_run(exp, 3, FIVE_SEEDS[2], [(0, 0.0), (150, 3.0), (200, 4.0)])
    assert summarise_experiment(exp, last_steps=49)["laststeps_per_seed"] == [2.0, 4.0]  # rows >= 151: [200] and [200]
    with pytest.raises(ValueError):
        summarise_experiment(exp, last_steps=50)  # rows >= 150: [200] against [150, 200] (inclusive)
    with pytest.raises(ValueError) as exc:
        summarise_experiment(exp, last_steps=100)  # [100, 200] against [150, 200]
    message = str(exc.value)
    assert str(exp) in message
    assert "[100, 200]" in message and "[150, 200]" in message
    # a denser run: the same steps within the window, so the same tail ...
    exp = tmp_path / "denser"
    write_run(exp, 0, FIVE_SEEDS[1], [(0, 0.0), (100, 1.0), (200, 2.0)])
    write_run(exp, 1, FIVE_SEEDS[2], [(0, 0.0), (50, 9.0), (100, 3.0), (200, 4.0)])
    assert summarise_experiment(exp, last_steps=100)["laststeps_per_seed"] == [1.5, 3.5]
    # ... until the window reaches the extra row
    with pytest.raises(ValueError) as exc:
        summarise_experiment(exp, last_steps=150)  # [100, 200] against [50, 100, 200]
    assert "[100, 200]" in str(exc.value) and "[50, 100, 200]" in str(exc.value)


def test_equal_tails_keep_every_last_k_number(five):
    s = summarise_experiment(five, last_k=2)
    assert s["lastk_per_seed"] == [2.0, 3.5, 3.5, 6.0, 7.5]
    assert s["lastk_mean_score"] == pytest.approx(4.5, rel=1e-12)
    assert s["lastk_ci95"] == pytest.approx(2.7411, abs=5e-5)
    assert summarise_experiment(five, last_k=3)["lastk_mean_score"] == pytest.approx(3.0, rel=1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("error", AggregateWarning)
        summarise_experiment(five, last_k=1)
        summarise_experiment(five, last_steps=51_200)


# -------------------------------------------------------------- results_table
#: Twelve standard cells of three hand-built rows (the rows of ``test_results_table_exact_format``).
ROWS = [
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
LINES_12 = [
    "| bfs | 1 | 100 | 30.61 +- 1.60 | episodes | 31.0 | 58 | 255.8 | 0.16 | 0.84 | 0.00 | 0.00 |",
    "| tabular_q | 5 | 100 | 12.34 +- 0.80 | seeds | 12.5 | 31 | 140.2 | 0.25 | 0.70 | 0.05 | 0.00 |",
    "| one_seed | 1 | 100 | 9.00 +- n/a | seeds | 9.0 | 20 | 99.0 | 0.50 | 0.50 | 0.00 | 0.00 |",
]


def laststeps(row: dict, S: int, mean: float, ci: float, per_seed=None) -> dict:
    per_seed = [mean] if per_seed is None else per_seed
    return {**row, "last_steps": S, "laststeps_mean_score": mean, "laststeps_ci95": ci,
            "laststeps_per_seed": per_seed, "laststeps_n_rows": [5] * len(per_seed)}


def test_table_gains_the_last_steps_column_exact_format():
    rows = [ROWS[0], laststeps(ROWS[1], 51_200, 12.5, 0.75, [11.5, 12.0, 12.2, 12.9, 13.9]),
            laststeps(ROWS[2], 51_200, 9.25, float("nan"))]
    assert results_table(rows) == "\n".join([
        HEAD_12 + " last-51200 steps mean +- 95% CI |",
        RULE_12 + "---|",
        LINES_12[0] + " n/a |",  # a baseline has no evaluation curve
        LINES_12[1] + " 12.50 +- 0.75 |",
        LINES_12[2] + " 9.25 +- n/a |",  # one seed: no interval, never "nan"
    ])
    assert all(line.count("|") == 14 for line in results_table(rows).splitlines())
    # a row that says "no last-steps" explicitly is the same as a row without the key
    assert results_table([{**r, "last_steps": None} for r in ROWS]) == results_table(ROWS)
    assert results_table(ROWS) == "\n".join([HEAD_12, RULE_12, *LINES_12])


def test_table_of_real_rows_with_last_steps(five, tmp_path):
    bfs, _ = write_baselines(tmp_path)
    with pytest.warns(AggregateWarning, match="same conditions"):  # 3 episodes against 2-6: spec 1.4
        table = results_table([summarise_baseline(bfs), summarise_experiment(five, last_steps=100)])
    assert table == "\n".join([
        HEAD_12 + " last-100 steps mean +- 95% CI |",
        RULE_12 + "---|",
        BFS_ROW + " n/a |",
        FIVE_ROW + " 4.50 +- 2.74 |",
    ])
    # the twelve existing columns keep their position and content
    assert [line.split(" | ")[:11] for line in table.splitlines()[2:]] == [
        line.split(" | ")[:11] for line in (BFS_ROW, FIVE_ROW)]


def test_table_shows_na_for_an_experiment_row_without_last_steps(five):
    with warnings.catch_warnings():
        warnings.simplefilter("error", AggregateWarning)  # the same experiment twice: evaluated alike
        lines = results_table(
            [summarise_experiment(five, label="plain"), summarise_experiment(five, label="s99", last_steps=99)]
        ).splitlines()
    assert lines[0].endswith("| win | last-99 steps mean +- 95% CI |")
    assert lines[2].endswith("| 0.07 | n/a |") and lines[3].endswith("| 0.07 | 6.00 +- 3.62 |")


def test_table_refuses_rows_with_different_last_steps(five):
    rows = [summarise_experiment(five, label="a", last_steps=99), summarise_experiment(five, label="b", last_steps=100)]
    with pytest.raises(ValueError, match="last_steps"):
        results_table(rows)
    with pytest.raises(ValueError):
        results_table([laststeps(ROWS[1], 100, 1.0, 0.1), laststeps(ROWS[2], 200, 1.0, 0.1)])


def test_table_refuses_rows_with_last_k_next_to_rows_with_last_steps(five):
    rows = [summarise_experiment(five, label="k", last_k=2), summarise_experiment(five, label="s", last_steps=100)]
    with pytest.raises(ValueError) as exc:
        results_table(rows)
    assert "last_k" in str(exc.value) and "last_steps" in str(exc.value)
    with pytest.raises(ValueError) as exc:
        results_table(rows[::-1])
    assert "last_k" in str(exc.value) and "last_steps" in str(exc.value)
    # hand-built rows: the same numbers under the two keys are still two different columns
    with pytest.raises(ValueError):
        results_table([{**ROWS[1], "last_k": 2, "lastk_mean_score": 4.5, "lastk_ci95": 2.74, "lastk_per_seed": [4.5]},
                       laststeps(ROWS[2], 100, 4.5, 2.74)])


def test_table_refuses_a_row_carrying_both_keys(five):
    both = {**summarise_experiment(five, last_k=2), **laststeps(ROWS[1], 100, 4.5, 2.74)}
    both["label"] = "both"
    with pytest.raises(ValueError) as exc:
        results_table([both])
    assert "last_k" in str(exc.value) and "last_steps" in str(exc.value)
    with pytest.raises(ValueError):
        results_table([{**laststeps(ROWS[1], 100, 4.5, 2.74), "last_k": 2, "lastk_mean_score": 4.5,
                        "lastk_ci95": 2.74, "lastk_per_seed": [4.5]}])


def test_table_refuses_duplicate_labels(five, tmp_path):
    with pytest.raises(ValueError, match="labels must be unique") as exc:
        results_table([summarise_experiment(five), summarise_experiment(five)])
    assert "five_seeds" in str(exc.value)
    with pytest.raises(ValueError, match="labels must be unique") as exc:  # also without a tail column
        results_table([ROWS[1], ROWS[0], ROWS[1]])
    assert "tabular_q" in str(exc.value)
    with pytest.raises(ValueError, match="labels must be unique"):  # also for a hand-built pair with a tail column
        results_table([laststeps(ROWS[1], 100, 1.0, 0.1), laststeps(ROWS[1], 100, 2.0, 0.2)])
    with pytest.raises(ValueError, match="labels must be unique"):  # the raw labels are compared, before escaping
        results_table([{**ROWS[0], "label": "a | b"}, {**ROWS[1], "label": "a | b"}])
    # distinct labels are fine, also when they differ only in case or by a pipe
    rows = [{**ROWS[0], "label": "a"}, {**ROWS[1], "label": "A"}, {**ROWS[2], "label": "a | b"}]
    assert len(results_table(rows).splitlines()) == 5
    bfs, rnd = write_baselines(tmp_path)
    with pytest.raises(ValueError, match="labels must be unique"):
        results_table([summarise_baseline(bfs, label="baseline"), summarise_baseline(rnd, label="baseline")])


# -------------------------------- agent.keep_checkpoints is not a setting (1.6)
EPISODES = [(1, 5, "wall"), (3, 5, "self")]
CURVE = [(0, 0.0), (100, 2.0)]


def two_runs(tmp_path: Path) -> Path:
    exp = tmp_path / "exp"
    write_run(exp, 0, EPISODES, CURVE)
    write_run(exp, 1, EPISODES, CURVE)
    return exp


def test_ignored_settings_is_the_public_tuple():
    assert IGNORED_SETTINGS == ("agent.keep_checkpoints",)
    assert isinstance(IGNORED_SETTINGS, tuple)


@pytest.mark.parametrize(
    "first, second",
    [(True, False), (False, True), ("absent", False), ("absent", True), (True, "absent")],
    ids=["true-false", "false-true", "absent-false", "absent-true", "true-absent"],
)
def test_runs_that_differ_only_in_keep_checkpoints_are_one_configuration(tmp_path, first, second):
    exp = two_runs(tmp_path)
    for run_dir, value in ((exp / "0", first), (exp / "1", second)):
        if value != "absent":
            edit_config(run_dir, "agent.keep_checkpoints", value)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert [r.seed for r in load_runs(exp)] == [0, 1]
        s = summarise_experiment(exp)
        assert s["n_seeds"] == 2 and s["per_seed_mean_scores"] == [2.0, 2.0]
        assert learning_curve(exp)["mean"].tolist() == [0.0, 2.0]


@pytest.mark.parametrize("key, value", [("agent.gamma", 0.5), ("env.W", 6), ("train.eval_episodes", 10)])
def test_any_other_difference_still_raises(tmp_path, key, value):
    exp = two_runs(tmp_path)
    edit_config(exp / "1", key, value)
    for call in (load_runs, summarise_experiment, learning_curve):
        with pytest.raises(ValueError, match=key.replace(".", r"\.")):
            call(exp)


def test_keep_checkpoints_is_not_named_when_another_key_differs(tmp_path):
    exp = two_runs(tmp_path)
    edit_config(exp / "0", "agent.keep_checkpoints", True)
    edit_config(exp / "1", "agent.keep_checkpoints", False)
    edit_config(exp / "1", "agent.gamma", 0.5)
    with pytest.raises(ValueError, match="do not share one configuration") as exc:
        load_runs(exp)
    message = str(exc.value).replace(str(exp), "")  # the tmp_path carries this test's name
    assert "agent.gamma" in message and "keep_checkpoints" not in message

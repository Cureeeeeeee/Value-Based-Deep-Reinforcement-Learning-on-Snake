"""Last-k summary of ``snake_rl/analysis/aggregate.py`` (``last_k`` / ``--last-k``).

Synthetic run directories from ``tests.test_aggregate``. The five-seed
fixture has the evaluation curves (rows at env_steps 0, 100, 200)::

    seed 0: 0 1 3    seed 1: 0 2 5    seed 2: 0 3 4    seed 3: 0 4 8    seed 10: 0 5 10

=========  =======================  ======  ============================  =====================
last_k     per-seed means           mean    half-width (t(4) = 2.776)     wrong alternative
=========  =======================  ======  ============================  =====================
1          3 5 4 8 10               6.0     2.776 * sqrt(8.5 / 5) = 3.62  -
2          2 3.5 3.5 6 7.5          4.5     2.776 * sqrt(4.875 / 5)       1.944 (10 pooled rows,
                                            = 2.7411                      t(9)); mean 1.5 (first
                                                                          two rows)
3          4/3 7/3 7/3 4 5          3.0     2.776 * sqrt((13 / 6) / 5)    -
                                            = 1.8274
=========  =======================  ======  ============================  =====================
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from snake_rl.analysis.aggregate import (
    AggregateWarning,
    main,
    results_table,
    summarise_baseline,
    summarise_experiment,
)
from snake_rl.utils import REPO_ROOT
from tests.test_aggregate import (
    CI_FIVE,
    FIVE_CURVES,
    FIVE_SEEDS,
    STEPS,
    T4,
    make_five_seed_exp,
    write_baselines,
    write_run,
)

LASTK_KEYS = {"last_k", "lastk_mean_score", "lastk_ci95", "lastk_per_seed"}

HEAD_12 = (
    "| label | seeds | episodes | score mean +- 95% CI | CI over | median | max | mean steps "
    "| wall | self | starve | win |"
)
RULE_12 = "|---|---|---|---|---|---|---|---|---|---|---|---|"
FIVE_ROW = "| five_seeds | 5 | 2-6 | 6.00 +- 3.62 | seeds | 4.5 | 12 | 35.7 | 0.50 | 0.29 | 0.14 | 0.07 |"
BFS_ROW = "| bfs_W8 | 1 | 3 | 30.00 +- 11.32 | episodes | 30.0 | 40 | 250.0 | 0.33 | 0.67 | 0.00 | 0.00 |"


@pytest.fixture
def five(tmp_path: Path) -> Path:
    return make_five_seed_exp(tmp_path)


# ------------------------------------------------------- summarise_experiment
def test_last_k_two_by_hand(five):
    s = summarise_experiment(five, last_k=2)
    assert s["last_k"] == 2 and isinstance(s["last_k"], int)
    assert s["lastk_per_seed"] == [2.0, 3.5, 3.5, 6.0, 7.5]  # seed order 0 1 2 3 10
    assert all(isinstance(v, float) for v in s["lastk_per_seed"])
    # mean of the five per-seed means; the first two rows would give 1.5
    assert s["lastk_mean_score"] == pytest.approx(4.5, rel=1e-12)
    # deviations -2.5 -1 -1 1.5 3: SS 19.5, s^2 = 19.5 / 4; Student t over 5 seeds, not over 10 rows (1.944)
    assert s["lastk_ci95"] == pytest.approx(T4 * math.sqrt(4.875 / 5), rel=1e-12)
    assert s["lastk_ci95"] == pytest.approx(2.7411, abs=5e-5)


def test_last_k_three_by_hand(five):
    s = summarise_experiment(five, last_k=3)  # all rows, the evaluation of the untrained agent included
    assert s["lastk_per_seed"] == pytest.approx([4 / 3, 7 / 3, 7 / 3, 4.0, 5.0], rel=1e-12)
    assert s["lastk_mean_score"] == pytest.approx(3.0, rel=1e-12)
    # deviations -5/3 -2/3 -2/3 1 2: SS 26/3, s^2 = 13/6
    assert s["lastk_ci95"] == pytest.approx(T4 * math.sqrt(13 / 6 / 5), rel=1e-12)
    assert s["lastk_ci95"] == pytest.approx(1.8274, abs=5e-5)


def test_last_k_one_is_the_last_row_of_the_curve(five):
    s = summarise_experiment(five, last_k=1)
    assert s["lastk_per_seed"] == [FIVE_CURVES[seed][-1] for seed in (0, 1, 2, 3, 10)] == [3.0, 5.0, 4.0, 8.0, 10.0]
    assert s["lastk_mean_score"] == pytest.approx(6.0, rel=1e-12)
    assert s["lastk_ci95"] == pytest.approx(CI_FIVE, rel=1e-12)


def test_last_k_takes_the_last_rows_in_file_order(tmp_path):
    exp = tmp_path / "long"
    write_run(exp, 0, FIVE_SEEDS[1], [(0, 0.0), (100, 9.0), (200, 1.0), (300, 2.0), (400, 6.0)])
    write_run(exp, 1, FIVE_SEEDS[2], [(0, 0.0), (100, 9.0), (200, 3.0), (300, 4.0), (400, 5.0)])
    s = summarise_experiment(exp, last_k=3)
    assert s["lastk_per_seed"] == [3.0, 4.0]  # (1 + 2 + 6) / 3 and (3 + 4 + 5) / 3; the 9.0 at step 100 is not in
    assert s["lastk_mean_score"] == pytest.approx(3.5, rel=1e-12)
    # two seeds: t(1) * |a - b| / 2
    assert s["lastk_ci95"] == pytest.approx(12.706 * 1.0 / 2, rel=1e-12)


def test_last_k_only_adds_keys(five):
    plain = summarise_experiment(five)
    assert not LASTK_KEYS & plain.keys()
    with_k = summarise_experiment(five, label="x", last_k=2)
    assert with_k.keys() == plain.keys() | LASTK_KEYS
    assert {k: v for k, v in with_k.items() if k not in LASTK_KEYS} == {**plain, "label": "x"}
    assert summarise_experiment(five, None, None) == plain


def test_last_k_single_seed_has_no_interval(tmp_path):
    exp = tmp_path / "one_seed"
    write_run(exp, 0, FIVE_SEEDS[1], [(0, 0.5), (100, 2.5), (200, 5.0)])
    s = summarise_experiment(exp, last_k=2)
    assert s["lastk_per_seed"] == [3.75] and s["lastk_mean_score"] == 3.75
    assert math.isnan(s["lastk_ci95"])
    row = results_table([s]).splitlines()[2]
    assert row == "| one_seed | 1 | 2 | 5.00 +- n/a | seeds | 5.0 | 6 | 30.0 | 0.50 | 0.50 | 0.00 | 0.00 | 3.75 +- n/a |"
    assert "nan" not in results_table([s]).lower()


@pytest.mark.parametrize("last_k", [0, -1, -5, True, 2.0, "2"])
def test_last_k_below_one_or_not_an_integer_raises(five, last_k):
    with pytest.raises(ValueError, match="last_k"):
        summarise_experiment(five, last_k=last_k)


def test_last_k_is_checked_before_any_file_is_read(tmp_path):
    with pytest.raises(ValueError, match="last_k must be an integer >= 1"):
        summarise_experiment(tmp_path / "does_not_exist", last_k=0)


def test_last_k_larger_than_the_number_of_rows_raises(five):
    assert summarise_experiment(five, last_k=len(STEPS))["last_k"] == 3
    with pytest.raises(ValueError, match=r"metrics\.csv: 3 evaluation row\(s\), fewer than last_k = 4"):
        summarise_experiment(five, last_k=4)


def test_last_k_names_the_run_that_is_too_short(tmp_path):
    exp = tmp_path / "ragged"
    write_run(exp, 0, FIVE_SEEDS[1], [(0, 0.0), (100, 1.0), (200, 2.0)])
    write_run(exp, 1, FIVE_SEEDS[2], [(0, 0.0), (100, 1.0)])
    with pytest.raises(ValueError) as exc:
        summarise_experiment(exp, last_k=3)
    assert str(exp / "1" / "metrics.csv") in str(exc.value) and "2 evaluation row(s)" in str(exc.value)
    # M10: the last two rows of seed 0 are at env_steps 100, 200, of seed 1 at 0, 100 - the seeds would be
    # compared over different steps, which is refused (before M10 this returned [1.5, 0.5]).
    with pytest.raises(ValueError, match="same steps"):
        summarise_experiment(exp, last_k=2)


def test_last_k_needs_the_score_column(tmp_path):
    exp = tmp_path / "broken"
    run_dir = write_run(exp, 0, FIVE_SEEDS[1], [(0, 0.0)])
    (run_dir / "metrics.csv").write_text("env_steps,loss\n0,1.0\n", encoding="utf-8")
    assert summarise_experiment(exp)["mean_score"] == 5.0  # the plain row does not read the curve
    with pytest.raises(ValueError, match="eval_mean_score"):
        summarise_experiment(exp, last_k=1)
    (run_dir / "metrics.csv").write_text("env_steps,eval_mean_score\n0,1.0\n100,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a finite number"):
        summarise_experiment(exp, last_k=1)  # an empty cell must not put nan into a table


def test_last_k_ignores_unfinished_runs(five):
    reference = summarise_experiment(five, last_k=2)
    write_run(five, 4, [(50, 500, "win")] * 2, [(0, 0.0)], finished=False)  # one row only, and not finished
    with pytest.warns(AggregateWarning, match="unfinished"):
        assert summarise_experiment(five, last_k=2) == reference


# -------------------------------------------------------------- results_table
def test_default_table_is_byte_identical_to_the_table_before_last_k():
    """The rows and the exact string of ``test_results_table_exact_format``."""
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
    # a row that says "no last-k" explicitly is the same as a row without the key
    assert results_table([{**r, "last_k": None} for r in rows]) == results_table(rows)


def test_table_of_real_rows_without_last_k_is_unchanged(five, tmp_path):
    bfs, _ = write_baselines(tmp_path)
    table = results_table([summarise_baseline(bfs), summarise_experiment(five)])
    assert table == "\n".join([HEAD_12, RULE_12, BFS_ROW, FIVE_ROW])


def test_table_gains_one_column_when_a_row_carries_last_k(five, tmp_path):
    bfs, _ = write_baselines(tmp_path)
    table = results_table([summarise_baseline(bfs), summarise_experiment(five, last_k=2)])
    assert table == "\n".join(
        [
            HEAD_12 + " last-2 evals mean +- 95% CI |",
            RULE_12 + "---|",
            BFS_ROW + " n/a |",  # a baseline has no evaluation curve
            FIVE_ROW + " 4.50 +- 2.74 |",
        ]
    )
    assert all(line.count("|") == 14 for line in table.splitlines())
    # the twelve existing columns keep their position and content
    assert [line.split(" | ")[:11] for line in table.splitlines()[2:]] == [
        line.split(" | ")[:11] for line in (BFS_ROW, FIVE_ROW)
    ]


def test_table_shows_na_for_an_experiment_row_without_last_k(five):
    lines = results_table(
        [summarise_experiment(five, label="plain"), summarise_experiment(five, label="k3", last_k=3)]
    ).splitlines()
    assert lines[0].endswith("| win | last-3 evals mean +- 95% CI |")
    assert lines[2].endswith("| 0.07 | n/a |") and lines[3].endswith("| 0.07 | 3.00 +- 1.83 |")


def test_table_refuses_rows_with_different_last_k(five):
    rows = [summarise_experiment(five, label="a", last_k=1), summarise_experiment(five, label="b", last_k=2)]
    with pytest.raises(ValueError, match="different last_k"):
        results_table(rows)


# ------------------------------------------------------------------------ CLI
def test_cli_last_k_adds_the_column(five, tmp_path, capsys):
    bfs, rnd = write_baselines(tmp_path)
    out = tmp_path / "tables" / "lastk.md"
    curves_dir = tmp_path / "curves"
    argv = ["--baseline", str(bfs), str(rnd), "--exp", str(five), "--out", str(out), "--curves-dir", str(curves_dir)]

    assert main(argv) == 0
    plain_table = capsys.readouterr().out
    assert plain_table == results_table(
        [summarise_baseline(bfs), summarise_baseline(rnd), summarise_experiment(five)]
    ) + "\n"
    assert plain_table.splitlines()[:2] == [HEAD_12, RULE_12] and plain_table.splitlines()[4] == FIVE_ROW
    plain_bytes, plain_curve = out.read_bytes(), (curves_dir / "five_seeds_curve.csv").read_bytes()
    assert plain_bytes == plain_table.encode("utf-8")

    assert main(argv + ["--last-k", "2"]) == 0
    table = capsys.readouterr().out
    assert table == results_table(
        [summarise_baseline(bfs), summarise_baseline(rnd), summarise_experiment(five, last_k=2)]
    ) + "\n"
    lines = table.splitlines()
    assert lines[0] == HEAD_12 + " last-2 evals mean +- 95% CI |"
    assert lines[2] == BFS_ROW + " n/a |" and lines[3].endswith("| n/a |") and lines[4] == FIVE_ROW + " 4.50 +- 2.74 |"
    assert out.read_bytes() == table.encode("utf-8")
    # the curve file does not depend on --last-k
    assert (curves_dir / "five_seeds_curve.csv").read_bytes() == plain_curve


def test_cli_last_k_with_baselines_only_adds_no_column(tmp_path, capsys):
    bfs, _ = write_baselines(tmp_path)
    assert main(["--baseline", str(bfs), "--last-k", "5"]) == 0
    assert capsys.readouterr().out == "\n".join([HEAD_12, RULE_12, BFS_ROW]) + "\n"


@pytest.mark.parametrize("value", ["0", "-3", "two", "1.5"])
def test_cli_rejects_a_bad_last_k(five, capsys, value):
    with pytest.raises(SystemExit) as exc:
        main(["--exp", str(five), "--last-k", value])
    assert exc.value.code == 2
    assert "--last-k" in capsys.readouterr().err


def test_cli_last_k_beyond_the_curve_writes_no_table(five, tmp_path):
    out = tmp_path / "table.md"
    with pytest.raises(ValueError, match="fewer than last_k = 4"):
        main(["--exp", str(five), "--last-k", "4", "--out", str(out)])
    assert not out.exists()


# ------------------------------------------------ real run directories (seam)
def test_last_k_reads_what_train_writes(tmp_path, monkeypatch):
    """Two seeds trained by ``snake_rl.train``: last-1 is the final evaluation,
    last-k is recomputed here from the ``metrics.csv`` text."""
    from snake_rl import evaluate as evaluate_module
    from snake_rl import train as train_module
    from snake_rl.config import load_config

    monkeypatch.setattr(train_module, "git_hash", lambda: "testhash")
    monkeypatch.setattr(evaluate_module, "git_hash", lambda: "testhash")
    cfg = load_config(
        REPO_ROOT / "configs" / "smoke_tabular.yaml",
        ["exp=seam_k", "env.starve_limit=20", "train.n_envs=8", "train.total_env_steps=2400", "train.eval_every=800",
         "train.eval_episodes=6"],
    )
    for seed in (2, 11):
        train_module.train(cfg, seed, tmp_path, quiet=True)
    exp = tmp_path / "seam_k"

    def column(seed: int) -> list[float]:
        header, *rows = (exp / str(seed) / "metrics.csv").read_text(encoding="utf-8").splitlines()
        at = header.split(",").index("eval_mean_score")
        return [float(row.split(",")[at]) for row in rows]

    scores = {seed: column(seed) for seed in (2, 11)}
    assert all(len(v) == 4 for v in scores.values())  # env_steps 0, 800, 1600, 2400

    last_1 = summarise_experiment(exp, last_k=1)
    assert last_1["lastk_per_seed"] == last_1["per_seed_mean_scores"]
    assert last_1["lastk_mean_score"] == pytest.approx(last_1["mean_score"], rel=1e-12)
    assert last_1["lastk_ci95"] == pytest.approx(last_1["ci95"], rel=1e-12)

    last_3 = summarise_experiment(exp, last_k=3)
    per_seed = [sum(scores[seed][-3:]) / 3 for seed in (2, 11)]
    assert last_3["lastk_per_seed"] == pytest.approx(per_seed, rel=1e-12)
    assert last_3["lastk_mean_score"] == pytest.approx(sum(per_seed) / 2, rel=1e-12)
    assert last_3["lastk_ci95"] == pytest.approx(12.706 * abs(per_seed[0] - per_seed[1]) / 2, rel=1e-12, abs=1e-12)
    with pytest.raises(ValueError, match="fewer than last_k = 5"):
        summarise_experiment(exp, last_k=5)

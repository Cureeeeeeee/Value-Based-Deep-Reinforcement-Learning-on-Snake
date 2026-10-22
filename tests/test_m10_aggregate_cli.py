"""The M10 additions to the aggregate CLI (spec 1.5: ``--last-steps``, glob
patterns, duplicate names) and the evaluation-conditions check (spec 1.4:
the ``evaluation`` key, ``comparability_notes`` and the one warning of
``results_table``).

Synthetic run directories and baseline files from ``tests.test_aggregate``.
Every fixture there is evaluated on the 8x8 board with ``eval_seed`` 10 000;
the five-seed experiment has unequal episode counts (``n_episodes`` None)
and the two baselines of ``write_baselines`` have 3 episodes each, so a
baseline next to the five-seed row differs in ``n_episodes`` and must warn.
Rows that agree on all three values (board, evaluation seed, episode count)
are built here from runs with two episodes per seed and a baseline with two
episodes. The last-steps numbers are those of
``tests/test_m10_aggregate_laststeps.py``: ``last_steps = 100`` on the
five-seed fixture is ``4.50 +- 2.74``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
import pytest

from snake_rl.analysis.aggregate import (
    AggregateWarning,
    comparability_notes,
    learning_curve,
    main,
    results_table,
    summarise_baseline,
    summarise_experiment,
)
from snake_rl.env import EnvConfig
from snake_rl.evaluate import EpisodeRecord, EvalResult, env_config_to_dict
from tests.test_aggregate import (
    EVAL_SEED,
    FIVE_SEEDS,
    make_five_seed_exp,
    make_result,
    write_baselines,
    write_run,
)
from tests.test_aggregate_lastk import BFS_ROW, FIVE_ROW, HEAD_12, RULE_12

TWO = [(1, 5, "wall"), (3, 5, "self")]  # two episodes: n_episodes 2
CURVE = [(0, 0.0), (100, 2.0)]
WARNING_START = "rows were not evaluated under the same conditions"


@pytest.fixture
def five(tmp_path: Path) -> Path:
    return make_five_seed_exp(tmp_path)


def result_with(episodes, policy: str = "bfs", W: int = 8, seed: int = EVAL_SEED) -> EvalResult:
    """``make_result`` with a choice of board and evaluation seed."""
    records = [EpisodeRecord(seed=seed + i, score=s, steps=st, length=s + 3, cause=c)
               for i, (s, st, c) in enumerate(episodes)]
    return EvalResult(policy=policy, env=env_config_to_dict(EnvConfig(W=W)), seed=seed, episodes=records,
                      git="abc1234", wall_time_s=0.5)


def write_exp(root: Path, name: str, seeds=(0, 1), episodes=TWO) -> Path:
    exp = root / name
    for seed in seeds:
        write_run(exp, seed, episodes, CURVE)
    return exp


def cli(argv: list[str], capsys) -> tuple[int, str, str]:
    """``main(argv)`` -> (exit code, stdout, stderr); a usage error is exit code 2."""
    try:
        code = main(argv)
    except SystemExit as exit_:
        code = exit_.code
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# ------------------------------------------------------------- --last-steps
def test_cli_last_steps_adds_the_column(five, tmp_path, capsys):
    out = tmp_path / "tables" / "laststeps.md"
    curves_dir = tmp_path / "curves"
    argv = ["--exp", str(five), "--out", str(out), "--curves-dir", str(curves_dir)]
    assert main(argv) == 0
    plain = capsys.readouterr().out
    assert plain == results_table([summarise_experiment(five)]) + "\n"
    assert plain.splitlines() == [HEAD_12, RULE_12, FIVE_ROW]
    plain_curve = (curves_dir / "five_seeds_curve.csv").read_bytes()

    assert main(argv + ["--last-steps", "100"]) == 0
    table = capsys.readouterr().out
    assert table == results_table([summarise_experiment(five, last_steps=100)]) + "\n"
    assert table.splitlines() == [
        HEAD_12 + " last-100 steps mean +- 95% CI |",
        RULE_12 + "---|",
        FIVE_ROW + " 4.50 +- 2.74 |",
    ]
    assert out.read_bytes() == table.encode("utf-8")
    assert (curves_dir / "five_seeds_curve.csv").read_bytes() == plain_curve  # the curve does not depend on the tail
    pd.testing.assert_frame_equal(pd.read_csv(curves_dir / "five_seeds_curve.csv"), learning_curve(five), rtol=1e-12)


def test_cli_last_steps_baseline_cell_is_na(five, tmp_path, capsys):
    bfs, _ = write_baselines(tmp_path)
    with pytest.warns(AggregateWarning, match=WARNING_START):  # 3 episodes next to 2-6 (spec 1.4)
        assert main(["--baseline", str(bfs), "--exp", str(five), "--last-steps", "99"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        HEAD_12 + " last-99 steps mean +- 95% CI |",
        RULE_12 + "---|",
        BFS_ROW + " n/a |",
        FIVE_ROW + " 6.00 +- 3.62 |",
    ]


def test_cli_last_steps_with_baselines_only_adds_no_column(tmp_path, capsys):
    bfs, _ = write_baselines(tmp_path)
    assert main(["--baseline", str(bfs), "--last-steps", "51200"]) == 0
    assert capsys.readouterr().out == "\n".join([HEAD_12, RULE_12, BFS_ROW]) + "\n"


def test_cli_last_k_and_last_steps_are_exclusive(five, tmp_path, capsys):
    out = tmp_path / "table.md"
    code, stdout, err = cli(["--exp", str(five), "--last-k", "2", "--last-steps", "100", "--out", str(out)], capsys)
    assert code == 2 and stdout == ""
    assert "--last-k" in err and "--last-steps" in err
    assert not out.exists()


@pytest.mark.parametrize("value", ["0", "-3", "two", "1.5"])
def test_cli_rejects_a_bad_last_steps(five, capsys, value):
    code, stdout, err = cli(["--exp", str(five), "--last-steps", value], capsys)
    assert code == 2 and stdout == ""
    assert "--last-steps" in err


def test_cli_last_steps_beyond_the_tail_agreement_writes_no_table(tmp_path):
    exp = tmp_path / "shifted"
    write_run(exp, 0, FIVE_SEEDS[1], [(0, 0.0), (100, 1.0), (200, 2.0)])
    write_run(exp, 1, FIVE_SEEDS[2], [(0, 0.0), (150, 3.0), (200, 4.0)])
    out = tmp_path / "table.md"
    with pytest.raises(ValueError, match=r"\[100, 200\]"):
        main(["--exp", str(exp), "--last-steps", "100", "--out", str(out)])
    assert not out.exists()


# ------------------------------------------------------------- glob patterns
@pytest.fixture
def three_exps(tmp_path: Path) -> Path:
    runs = tmp_path / "runs"
    for name in ("exp_3", "exp_1", "exp_2"):
        write_exp(runs, name)
    return runs


def labels_of(table: str) -> list[str]:
    return [line.split(" | ")[0].lstrip("| ") for line in table.splitlines()[2:]]


def test_cli_expands_an_exp_pattern_in_sorted_order(three_exps, capsys):
    runs = three_exps
    assert main(["--exp", str(runs / "exp_[12]")]) == 0
    table = capsys.readouterr().out
    assert labels_of(table) == ["exp_1", "exp_2"]
    assert table == results_table([summarise_experiment(runs / "exp_1"), summarise_experiment(runs / "exp_2")]) + "\n"
    assert main(["--exp", str(runs / "exp_*")]) == 0
    assert labels_of(capsys.readouterr().out) == ["exp_1", "exp_2", "exp_3"]
    assert main(["--exp", str(runs / "exp_?")]) == 0
    assert labels_of(capsys.readouterr().out) == ["exp_1", "exp_2", "exp_3"]


def test_cli_keeps_the_argument_order_and_sorts_within_a_pattern(three_exps, capsys):
    runs = three_exps
    assert main(["--exp", str(runs / "exp_3"), str(runs / "exp_[21]")]) == 0
    assert labels_of(capsys.readouterr().out) == ["exp_3", "exp_1", "exp_2"]  # argument order, matches sorted
    assert main(["--exp", str(runs / "exp_[2]"), "--exp", str(runs / "exp_[31]")]) == 0
    assert labels_of(capsys.readouterr().out) == ["exp_2", "exp_1", "exp_3"]


def test_cli_a_second_pattern_naming_the_same_experiment_is_a_duplicate(three_exps, capsys):
    runs = three_exps
    code, stdout, err = cli(["--exp", str(runs / "exp_3"), str(runs / "exp_[21]"), str(runs / "exp_[3]")], capsys)
    assert code == 2 and stdout == "" and "exp_3" in err


def test_cli_expands_a_baseline_pattern(tmp_path, capsys):
    bfs, rnd = write_baselines(tmp_path)
    assert main(["--baseline", str(tmp_path / "bl" / "*.json")]) == 0
    table = capsys.readouterr().out
    assert labels_of(table) == ["bfs_W8", "random_W8"]
    assert table == results_table([summarise_baseline(bfs), summarise_baseline(rnd)]) + "\n"
    assert main(["--baseline", str(tmp_path / "bl" / "random_W8.json"), str(tmp_path / "bl" / "bfs_W?.json")]) == 0
    assert labels_of(capsys.readouterr().out) == ["random_W8", "bfs_W8"]


def test_cli_pattern_without_a_match_is_a_usage_error(three_exps, tmp_path, capsys):
    pattern = str(three_exps / "nothing_*")
    out = tmp_path / "table.md"
    code, stdout, err = cli(["--exp", pattern, "--out", str(out)], capsys)
    assert code == 2 and stdout == ""
    assert names(pattern, err)
    assert not out.exists()
    pattern = str(tmp_path / "bl" / "*.json")  # no baselines directory at all
    code, stdout, err = cli(["--baseline", pattern], capsys)
    assert code == 2 and names(pattern, err)
    # a pattern among good arguments is still refused, before anything is written
    code, stdout, err = cli(["--exp", str(three_exps / "exp_1"), str(three_exps / "exp_[9]"), "--out", str(out)], capsys)
    assert code == 2 and names(str(three_exps / "exp_[9]"), err) and not out.exists()


def names(pattern: str, err: str) -> bool:
    """The error names the pattern, as typed or quoted (``repr`` doubles the backslashes of a Windows path)."""
    return pattern in err or repr(pattern) in err


def test_cli_plain_arguments_are_used_as_given(tmp_path, three_exps, capsys):
    """Without * ? [ nothing is expanded: a missing directory still fails in scan_experiment."""
    with pytest.raises(ValueError, match="not a directory"):
        main(["--exp", str(tmp_path / "runs" / "exp_9")])
    assert main(["--exp", str(three_exps / "exp_2")]) == 0
    assert labels_of(capsys.readouterr().out) == ["exp_2"]


# ---------------------------------------------------------- duplicate names
def test_cli_refuses_duplicate_experiment_names_before_anything_is_loaded(tmp_path, capsys):
    (tmp_path / "a" / "same").mkdir(parents=True)  # empty: loading it would raise "no finished run"
    (tmp_path / "b" / "same").mkdir(parents=True)
    out = tmp_path / "table.md"
    code, stdout, err = cli(["--exp", str(tmp_path / "a" / "same"), str(tmp_path / "b" / "same"), "--out", str(out)], capsys)
    assert code == 2 and stdout == ""
    assert "same" in err
    assert not out.exists()


def test_cli_refuses_duplicate_names_from_patterns_and_repeated_arguments(tmp_path, capsys):
    a = make_five_seed_exp(tmp_path / "a", name="same")
    b = make_five_seed_exp(tmp_path / "b", name="same")
    code, stdout, err = cli(["--exp", str(tmp_path / "a" / "*"), str(tmp_path / "b" / "*")], capsys)
    assert code == 2 and stdout == "" and "same" in err
    code, stdout, err = cli(["--exp", str(a), str(a)], capsys)  # the same directory twice
    assert code == 2 and stdout == "" and "same" in err
    code, stdout, err = cli(["--exp", str(a), "--exp", str(b)], capsys)  # also across repeated --exp
    assert code == 2 and stdout == "" and "same" in err
    assert main(["--exp", str(a)]) == 0  # alone it is fine
    assert labels_of(capsys.readouterr().out) == ["same"]


# ------------------------------------------------------- the evaluation key
def test_summary_rows_carry_the_evaluation_key(five, tmp_path):
    assert summarise_experiment(five)["evaluation"] == {"W": 8, "eval_seed": EVAL_SEED, "n_episodes": None}
    assert summarise_experiment(five, last_k=2)["evaluation"] == {"W": 8, "eval_seed": EVAL_SEED, "n_episodes": None}
    bfs, _ = write_baselines(tmp_path)
    assert summarise_baseline(bfs)["evaluation"] == {"W": 8, "eval_seed": EVAL_SEED, "n_episodes": 3}
    exp = write_exp(tmp_path, "two")
    assert summarise_experiment(exp)["evaluation"] == {"W": 8, "eval_seed": EVAL_SEED, "n_episodes": 2}
    ten = result_with(TWO, "random", W=10, seed=20_000).save(tmp_path / "random_W10.json")
    row = summarise_baseline(ten)
    assert row["evaluation"] == {"W": 10, "eval_seed": 20_000, "n_episodes": 2}
    assert all(type(v) is int for v in row["evaluation"].values())


def test_evaluation_comes_from_the_finished_runs_which_must_agree(tmp_path):
    exp = write_exp(tmp_path, "exp", seeds=(0, 1, 2))
    path = exp / "2" / "final_eval.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["env"]["W"] = 10  # config.yaml still agrees: only the evaluation record differs
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError):
        summarise_experiment(exp)
    doc["env"]["W"] = 8
    doc["seed"] = 20_000
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError):
        summarise_experiment(exp)
    doc["seed"] = EVAL_SEED
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert summarise_experiment(exp)["evaluation"] == {"W": 8, "eval_seed": EVAL_SEED, "n_episodes": 2}


# ------------------------------------------------------ comparability_notes
def test_comparability_notes_is_empty_when_every_row_agrees(tmp_path):
    a, b = write_exp(tmp_path, "a"), write_exp(tmp_path, "b", seeds=(0, 1, 2))
    two = result_with(TWO, "bfs").save(tmp_path / "bfs_two.json")
    assert comparability_notes([summarise_experiment(a), summarise_experiment(b), summarise_baseline(two)]) == []
    assert comparability_notes([summarise_experiment(a)]) == []
    assert comparability_notes([]) == []


def test_comparability_notes_name_both_labels_and_the_differing_value(five, tmp_path):
    bfs8 = result_with(TWO, "bfs").save(tmp_path / "bfs_W8.json")
    bfs10 = result_with(TWO, "bfs", W=10).save(tmp_path / "bfs_W10.json")
    seed20 = result_with(TWO, "bfs", seed=20_000).save(tmp_path / "bfs_seed20000.json")
    first = summarise_baseline(bfs8, label="first")

    notes = comparability_notes([first, summarise_baseline(bfs10, label="board ten")])
    assert len(notes) == 1 and all(isinstance(n, str) for n in notes)
    assert "first" in notes[0] and "board ten" in notes[0] and "10" in notes[0] and "8" in notes[0]

    notes = comparability_notes([first, summarise_baseline(seed20, label="other boards")])
    assert len(notes) == 1
    assert "first" in notes[0] and "other boards" in notes[0] and "20000" in notes[0] and "10000" in notes[0]

    notes = comparability_notes([first, summarise_experiment(five)])  # 2 episodes against 2-6
    assert len(notes) == 1
    assert "first" in notes[0] and "five_seeds" in notes[0]

    # one note per differing row, in row order; rows that agree with the first contribute nothing
    rows = [first, summarise_baseline(bfs8, label="same"), summarise_baseline(bfs10, label="ten"),
            summarise_baseline(seed20, label="twenty")]
    notes = comparability_notes(rows)
    assert len(notes) == 2
    assert "ten" in notes[0] and "twenty" in notes[1] and all("first" in n for n in notes)


def test_comparability_notes_skip_rows_without_the_key(tmp_path):
    bfs8 = result_with(TWO, "bfs").save(tmp_path / "bfs_W8.json")
    bfs10 = result_with(TWO, "bfs", W=10).save(tmp_path / "bfs_W10.json")
    hand = {"label": "hand-built", "n_seeds": 5, "seeds": [0, 1, 2, 3, 4], "n_episodes": 100, "mean_score": 12.34,
            "ci95": 0.8, "ci_kind": "seeds", "per_seed_mean_scores": [11.5, 12.0, 12.2, 12.9, 13.1],
            "median_score": 12.5, "max_score": 31, "mean_steps": 140.2,
            "cause_fractions": {"wall": 0.25, "self": 0.7, "starve": 0.05, "win": 0.0}}
    assert comparability_notes([hand, hand]) == []
    assert comparability_notes([hand, summarise_baseline(bfs8)]) == []
    # the reference is the first row THAT HAS the key
    notes = comparability_notes([hand, summarise_baseline(bfs8, label="eight"), summarise_baseline(bfs10, label="ten")])
    assert len(notes) == 1 and "eight" in notes[0] and "ten" in notes[0] and "hand-built" not in notes[0]


# --------------------------------------------- the warning of results_table
def test_results_table_warns_once_when_rows_differ(five, tmp_path):
    bfs, rnd = write_baselines(tmp_path)
    ten = result_with(TWO, "bfs", W=10).save(tmp_path / "bfs_W10.json")
    rows = [summarise_baseline(bfs), summarise_baseline(rnd), summarise_experiment(five), summarise_baseline(ten)]
    notes = comparability_notes(rows)
    assert len(notes) == 2  # five_seeds (episode count) and the 10x10 baseline (board, episode count)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        table = results_table(rows)
    found = [w for w in record if issubclass(w.category, AggregateWarning)]
    assert len(found) == 1
    message = str(found[0].message)
    assert message.startswith(WARNING_START)
    assert all(note in message for note in notes)
    assert len(table.splitlines()) == 6 and table.splitlines()[:2] == [HEAD_12, RULE_12]  # the table is still returned
    with pytest.warns(AggregateWarning, match=f"^{WARNING_START}"):
        results_table([summarise_experiment(five), summarise_baseline(bfs)])  # the order does not matter


def test_results_table_is_silent_when_rows_agree(tmp_path):
    a, b = write_exp(tmp_path, "a"), write_exp(tmp_path, "b", seeds=(0, 1, 2))
    two = result_with(TWO, "bfs").save(tmp_path / "bfs_two.json")
    with warnings.catch_warnings():
        warnings.simplefilter("error", AggregateWarning)
        table = results_table([summarise_experiment(a), summarise_experiment(b)])
        assert len(table.splitlines()) == 4
        table = results_table([summarise_baseline(two), summarise_experiment(a), summarise_experiment(b, last_k=1)])
        assert len(table.splitlines()) == 5
        results_table([summarise_experiment(a)])
        results_table([])


def test_cli_prints_the_table_and_exits_0_despite_the_warning(five, tmp_path, capsys):
    bfs, _ = write_baselines(tmp_path)
    out = tmp_path / "table.md"
    with pytest.warns(AggregateWarning, match=f"^{WARNING_START}") as record:
        assert main(["--baseline", str(bfs), "--exp", str(five), "--out", str(out)]) == 0
    found = [w for w in record if issubclass(w.category, AggregateWarning)]
    assert len(found) == 1
    printed = capsys.readouterr().out
    assert printed.splitlines() == [HEAD_12, RULE_12, BFS_ROW, FIVE_ROW]
    assert out.read_text(encoding="utf-8") == printed
    warned = str(found[0].message)
    assert "bfs_W8" in warned and "five_seeds" in warned


def test_cli_is_silent_when_its_rows_agree(tmp_path, capsys):
    a = write_exp(tmp_path / "runs", "a")
    two = result_with(TWO, "bfs").save(tmp_path / "bfs_two.json")
    with warnings.catch_warnings():
        warnings.simplefilter("error", AggregateWarning)
        assert main(["--baseline", str(two), "--exp", str(a), "--last-steps", "100"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 4 and lines[0] == HEAD_12 + " last-100 steps mean +- 95% CI |"

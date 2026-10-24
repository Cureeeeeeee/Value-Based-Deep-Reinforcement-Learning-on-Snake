"""The tables of the final report from the run directories
(``snake_rl/analysis/report_tables.py``).

Synthetic runs in ``tmp_path/runs``, written in the file formats of
``snake_rl.train``: the five-seed fixture of ``tests.test_aggregate`` (seed
means 3 5 4 8 10, 95 % half-width ``CI_FIVE``) under the experiment names the
tables look up, a second fixture with seed means 4 5 6 8 12 (paired
differences 1 0 2 0 2 against the first: mean 1, half-width 2.776 / sqrt 5 =
1.2415), DQN runs with value columns from ``tests.test_m7_value_curve``,
reference players as ``EvalResult`` files whose episode seeds are the
evaluation boards of the runs (``EVAL_SEED + i``), and behaviour analyses as
``summary.json`` + ``episodes.csv``. Every table function is checked for its
shape and a few hand-computed cells; the real report is checked by the
command line against ``runs/`` (``--report``), not here.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from snake_rl.analysis import report_tables as rt
from snake_rl.analysis.report_tables import Paired, SeedStats
from tests.test_aggregate import CI_FIVE, EVAL_SEED, FIVE_CURVES, FIVE_SEEDS, STEPS, T4, make_result, write_run
from tests.test_m7_value_curve import make_three_seed_exp

#: seed -> episodes with seed means 4 5 6 8 12 (seeds 1 and 3 as in FIVE_SEEDS).
SHIFTED = {
    0: [(1, 10, "wall"), (2, 10, "wall"), (2, 10, "wall"), (3, 10, "wall"), (3, 10, "wall"), (13, 10, "wall")],
    1: FIVE_SEEDS[1],
    2: [(5, 30, "self"), (7, 30, "self")],
    3: FIVE_SEEDS[3],
    10: [(11, 100, "starve"), (13, 100, "win")],
}
H_PAIRED = T4 / math.sqrt(5)  # paired differences 1 0 2 0 2: variance 1
CI_SHIFTED = T4 * math.sqrt(10 / 5)  # seed means 4 5 6 8 12: mean 7, sum of squares 40, variance 10


# ------------------------------------------------------------------- builders
def write_exp(runs: Path, name: str, episodes=FIVE_SEEDS) -> Path:
    exp = runs / name
    for seed in (10, 3, 0, 2, 1):
        write_run(exp, seed, episodes[seed], list(zip(STEPS, FIVE_CURVES[seed])))
    return exp


def write_bfs(runs: Path, W: int, scores=(1, 1, 1, 1, 1, 1)) -> Path:
    """BFS on the six evaluation boards EVAL_SEED + 0..5 (seed 0 of the fixture plays six)."""
    path = runs / "baselines" / f"bfs_W{W}_relative.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    make_result([(s, 10, "self") for s in scores], policy="bfs").save(path)
    return path


def write_behaviour(root: Path, name: str, summary: dict, episodes, seeds=(0,)) -> Path:
    """A behaviour analysis: ``episodes`` are ``(loop_steps, q_max_mean)`` rows written to every seed
    (``seeds=None``: a single player, one episodes.csv next to summary.json)."""
    out = root / name
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    targets = [out / str(s) for s in seeds] if seeds is not None else [out]
    for t in targets:
        t.mkdir(exist_ok=True)
        with (t / "episodes.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["seed", "score", "steps", "loop_steps", "q_max_mean"])
            w.writeheader()
            for k, (loop_steps, q) in enumerate(episodes):
                w.writerow({"seed": EVAL_SEED + k, "score": 1, "steps": 10, "loop_steps": loop_steps, "q_max_mean": q})
    return out


def dqn_summary(**over) -> dict:
    base = {"score_mean_mean": 20.5, "score_mean_ci95": 0.25, "steps_mean_mean": 187.84, "steps_mean_ci95": 5.3,
            "frac_self_mean": 0.94, "frac_wall_mean": 0.06, "frac_starve_mean": 0.0,
            "border_fraction_mean_mean": 0.22, "border_fraction_mean_ci95": 0.012,
            "loop_fraction_mean": 0.0, "loop_fraction_ci95": 0.0,
            "tail_without_food_mean_mean": 7.04, "tail_without_food_mean_ci95": 0.7}
    base.update(over)
    return base


def player_summary(**over) -> dict:
    base = {"score_mean": 30.61, "steps_mean": 255.79, "frac_self": 0.84, "frac_wall": 0.16, "frac_starve": 0.0,
            "frac_win": 0.0, "border_fraction_mean": 0.2714, "loop_fraction": 0.0, "tail_without_food_mean": 9.22}
    base.update(over)
    return base


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    r = tmp_path / "runs"
    r.mkdir()
    return r


# ----------------------------------------------------------------- formatting
def test_number_formats_use_the_report_conventions():
    assert rt.fmt_num(-1.5) == "−1.50"
    assert rt.fmt_num(2409.7, 1) == "2,409.7"
    assert rt.fmt_signed(0.004) == "+0.00"
    assert rt.fmt_signed(-0.004) == "−0.00"  # the report prints −0.00 for a tiny negative bound
    assert rt.fmt_interval(Paired(mean=1.714, lo=1.26, hi=2.15, npos=5, n=5)) == "+1.71 [+1.26, +2.15]"
    assert rt.fmt_interval(Paired(mean=146.2, lo=130.4, hi=161.9, npos=5, n=5), 0) == "+146 [+130, +162]"
    assert rt.fmt_causes([(0, 1, "wall"), (0, 1, "self"), (0, 1, "self")]) == "33 / 67 / 0"
    assert rt.fmt_ci([3, 5, 4, 8, 10]) == f"6.00 ± {CI_FIVE:.2f}"


def test_markdown_table_and_parse_round_trip():
    md = rt.markdown_table(["a", "b"], [["1", "x y"], ["2", ""]])
    assert md.splitlines()[1] == "|---|---|"
    assert rt.parse_table(md) == [["a", "b"], ["1", "x y"], ["2", ""]]


# -------------------------------------------------------------------- loading
def test_load_experiment_per_seed_statistics(runs):
    write_exp(runs, "dqn_A_W8")
    exp = rt.load_experiment(runs, "dqn_A_W8")
    assert [s.seed for s in exp.seeds] == [0, 1, 2, 3, 10]
    assert exp.per_seed("final") == [3, 5, 4, 8, 10]
    assert exp.per_seed("last5") == pytest.approx([4 / 3, 7 / 3, 7 / 3, 4, 5])  # three evaluations: all of them
    assert exp.seeds[0].frac("wall") == 1.0 and exp.seeds[3].frac("starve") == 0.5
    assert exp.seeds[1].food_per_100 == pytest.approx(100 * 10 / 60)
    assert exp.seeds[0].episode_seeds == tuple(EVAL_SEED + i for i in range(6))
    assert exp.seeds[0].values is None
    assert exp.seeds[0].settings["agent.alpha"] == 0.1
    assert rt.starving_snapshots(exp) == 10  # eval_frac_starve 0.25 on every row; the row at step 0 does not count
    assert rt.load_experiment(runs, "dqn_A_W8") is exp  # cached


def test_missing_experiment_is_a_dedicated_error(runs):
    with pytest.raises(rt.MissingExperiment):
        rt.load_experiment(runs, "dqn_A_W8")
    with pytest.raises(rt.MissingExperiment):
        rt.load_baseline(runs, "bfs", 8)
    assert not rt.has_experiment(runs, "dqn_A_W8")


# ----------------------------------------------------------------- statistics
def test_paired_difference_by_hand(runs):
    write_exp(runs, "a", SHIFTED)
    write_exp(runs, "b")
    p = rt.paired(rt.load_experiment(runs, "a"), rt.load_experiment(runs, "b"), "final")
    assert p.mean == pytest.approx(1.0)
    assert p.hi - p.mean == pytest.approx(H_PAIRED)
    assert p.npos == 3 and p.n == 5
    assert not p.excludes_zero
    assert rt.fmt_interval(p) == "+1.00 [−0.24, +2.24]"


def test_paired_needs_the_same_seeds(runs):
    write_exp(runs, "a")
    exp = runs / "b"
    for seed in (0, 1, 2, 3, 4):
        write_run(exp, seed, FIVE_SEEDS[1], list(zip(STEPS, FIVE_CURVES[1])))
    with pytest.raises(ValueError, match="same seeds"):
        rt.paired(rt.load_experiment(runs, "a"), rt.load_experiment(runs, "b"), "final")


def test_bfs_paired_on_the_same_boards(runs):
    write_exp(runs, "dqn_A_W8")
    write_bfs(runs, 8)
    p = rt.bfs_paired(rt.load_experiment(runs, "dqn_A_W8"), runs, 8)
    # per seed: mean over its boards of (score - 1): 2 4 3 7 9 -> the five-seed fixture shifted by -1
    assert p.mean == pytest.approx(5.0)
    assert p.hi - p.mean == pytest.approx(CI_FIVE)
    assert p.npos == 5


def test_bfs_paired_refuses_an_unknown_board(runs):
    write_exp(runs, "dqn_A_W8")
    write_bfs(runs, 8, scores=(1, 1))  # only two boards
    with pytest.raises(ValueError, match="no BFS episode"):
        rt.bfs_paired(rt.load_experiment(runs, "dqn_A_W8"), runs, 8)


def test_value_gap_and_ratio_by_hand(tmp_path):
    make_three_seed_exp(tmp_path, "dqn_B8_W8")
    exp = rt.load_experiment(tmp_path, "dqn_B8_W8")
    # per seed over the three evaluations: value means 4/3 7/3 7/3, return means 1/2 1 5/2
    assert exp.per_seed("gap_visited") == pytest.approx([4 / 3 - 0.5, 7 / 3 - 1, 7 / 3 - 2.5])
    assert exp.per_seed("gap_start") == pytest.approx([4 / 3 - 0.5 + 9, 7 / 3 - 1 + 9, 7 / 3 - 2.5 + 9])
    assert rt.value_ratio(exp) == pytest.approx(2.0 / (4 / 3))


def test_value_gap_needs_value_columns(runs):
    write_exp(runs, "dqn_A_W8")
    with pytest.raises(rt.MissingExperiment):
        rt.load_experiment(runs, "dqn_A_W8").per_seed("gap_visited")


# -------------------------------------------------------------------- tables
def test_table_representation_cells(runs):
    for W in (8, 10):
        write_exp(runs, f"dqn_A_W{W}", SHIFTED)
        for obs in ("B3", "B7", "B8"):
            write_exp(runs, f"dqn_{obs}_W{W}")
    rows = rt.parse_table(rt.table_representation(runs))
    assert rows[0] == ["Observation", "8 × 8, final", "8 × 8, last-5", "8 × 8, minus B8",
                       "10 × 10, final", "10 × 10, last-5", "10 × 10, minus B8"]
    assert [r[0].split(":")[0] for r in rows[1:]] == ["A", "B3", "B7", "B8"]
    assert rows[1][1] == f"7.00 ± {rt.fmt_num(CI_SHIFTED)}"
    assert rows[4][1] == f"6.00 ± {rt.fmt_num(CI_FIVE)}"
    assert rows[1][3] == "+1.00 [−0.24, +2.24] / +0.00 [+0.00, +0.00]"
    assert rows[4][3] == rows[4][6] == rt.DASH


def test_component_settings_from_the_configuration():
    def exp(**settings):
        s = SeedStats(seed=0, episode_seeds=(), episode_scores=(), episode_steps=(), episode_causes=(), curve=(),
                      starve_curve=(), env_steps=(), values=None, settings={"train.n_envs": 128, **settings})
        return rt.Experiment(name="x", seeds=(s,))
    assert rt._component_settings(exp(**{"agent.replay": True, "agent.grad_steps_per_vec_step": None,
                                         "agent.target_update_every": 1000})) == ("yes", 32, "1,000 (≈ 4k env steps)")
    assert rt._component_settings(exp(**{"agent.replay": False, "agent.grad_steps_per_vec_step": None,
                                         "agent.target_update_every": 1000})) == ("no", 1, "1,000 (= 128k env steps)")
    assert rt._component_settings(exp(**{"agent.replay": False, "agent.grad_steps_per_vec_step": None,
                                         "agent.target_update_every": 31})) == ("no", 1, "31 (≈ 4k env steps)")
    assert rt._component_settings(exp(**{"agent.replay": True, "agent.grad_steps_per_vec_step": 1,
                                         "agent.target_update_every": 1})) == ("yes", 1, "1")


def test_table_refinements_gap_columns(tmp_path):
    for _, base, variant in rt.REFINEMENT_ROWS:
        for name in (base, variant):
            if not (tmp_path / name).is_dir():
                make_three_seed_exp(tmp_path, name)
    rows = rt.parse_table(rt.table_refinements(tmp_path))
    assert len(rows) == 1 + len(rt.REFINEMENT_ROWS)
    gap = (4 / 3 - 0.5 + 7 / 3 - 1 + 7 / 3 - 2.5) / 3
    assert rows[1][4] == f"{rt.fmt_num(gap)} → {rt.fmt_num(gap)}"  # identical runs: no paired interval
    assert rows[1][5] == f"{rt.fmt_num(gap + 9)} → {rt.fmt_num(gap + 9)}"
    assert rows[1][3] == " / ".join(["+0.00 [+0.00, +0.00]"] * 3)


def test_table_behaviour_formatting(runs):
    b = runs / "behaviour"
    write_behaviour(b, "bfs_W10", player_summary(), [(0, "nan")], seeds=None)
    write_behaviour(b, "hamiltonian_W10", player_summary(score_mean=97.0, frac_win=1.0, steps_mean=2409.7,
                                                         border_fraction_mean=0.36, tail_without_food_mean=0.0),
                    [(0, "nan")], seeds=None)
    write_behaviour(b, "dqn_B8_W10_final", dqn_summary(), [(0, 1.0), (5, 2.0)])
    write_behaviour(b, "dqn_B8_W10_gamma099_early",
                    dqn_summary(frac_self_mean=0.31, frac_wall_mean=0.17, frac_starve_mean=0.52,
                                loop_fraction_mean=0.5, loop_fraction_ci95=0.1),
                    [(25, 3.0), (0, 1.0), (30, 5.0), (0, 2.0)])
    write_behaviour(b, "dqn_B8_W10_gamma099_mid", dqn_summary(frac_starve_mean=0.014, loop_fraction_mean=0.012,
                                                              loop_fraction_ci95=0.02), [(20, 9.65), (0, 6.96)])
    for name in ("dqn_B8_W10_gamma099_final", "dqn_B8_W10_gamma099_R1_final", "dqn_B8_W10_gamma099_R2_final",
                 "dqn_B8_W10_gamma099_R3_final"):
        write_behaviour(b, name, dqn_summary(), [(0, 1.0)])
    rows = rt.parse_table(rt.table_behaviour(runs))
    assert rows[0] == rt.BEHAVIOUR_HEADER
    assert rows[1] == ["BFS greedy", "30.61", "255.8", "0.84 / 0.16 / 0", "0.27", "0", rt.DASH, "9.2"]
    assert rows[2] == ["Hamiltonian cycle", "97 (all won)", "2,409.7", rt.DASH, "0.36", "0", rt.DASH, "0"]
    assert rows[3] == ["B8, γ = 0.9", "20.50 ± 0.25", "187.8 ± 5.3", "0.94 / 0.06 / 0", "0.22 ± 0.01", "0", rt.DASH,
                       "7.0 ± 0.7"]
    assert rows[4][3:7] == ["0.31 / 0.17 / 0.52", "0.22 ± 0.01", "0.50 ± 0.10 (2)", "4.00 / 1.50"]
    assert rows[5][3:7] == ["0.94 / 0.06 / 0.014", "0.22 ± 0.01", "0.012 (1)", "9.65 / 6.96"]


def test_table_behaviour_needs_every_analysis(runs):
    (runs / "behaviour").mkdir()
    with pytest.raises(rt.MissingExperiment, match="bfs_W10"):
        rt.table_behaviour(runs)


def test_table_sweep_cells(runs):
    sweep = runs / rt.SWEEP_DIR
    for alpha in rt.SWEEP_ALPHAS:
        for gamma in rt.SWEEP_GAMMAS:
            write_exp(sweep, f"tabular_q_W8__alpha={alpha}__gamma={gamma}")
    rows = rt.parse_table(rt.table_sweep(runs))
    assert rows[0] == ["α \\ γ", "0.8", "0.9", "0.95", "0.99"]
    assert [r[0] for r in rows[1:]] == list(rt.SWEEP_ALPHAS)
    assert rows[2][3] == f"6.00 ± {rt.fmt_num(CI_FIVE)} (10)"


def test_appendix_table_selects_the_report_columns(tmp_path):
    src = tmp_path / "summary_W8.md"
    src.write_text(
        "| label | seeds | episodes | score mean +- 95% CI | CI over | median | max | mean steps | wall | self | starve | win | last-5 evals mean +- 95% CI |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
        "| A (MLP) | 5 | 100 | 16.11 +- 0.88 | seeds | 16.0 | 31 | 104.9 | 0.21 | 0.79 | 0.00 | 0.00 | 16.07 +- 0.64 |\n",
        encoding="utf-8")
    rows = rt.parse_table(rt.appendix_table(src))
    assert rows[0] == [title for _, title in rt.APPENDIX_COLUMNS]
    assert rows[1] == ["A (MLP)", "16.11 ± 0.88", "16.07 ± 0.64", "104.9", "0.21", "0.79", "0.00"]


# ------------------------------------------------------- build, compare, CLI
def _representation_runs(runs: Path) -> None:
    for W in (8, 10):
        for obs in ("A", "B3", "B7", "B8"):
            write_exp(runs, f"dqn_{obs}_W{W}")


def test_build_skips_the_tables_whose_experiments_are_missing(runs):
    _representation_runs(runs)
    tables, skipped = rt.build(runs)
    assert set(tables) == {"Table 4"}
    assert set(skipped) == set(rt.TABLES) - {"Table 4"}
    assert "baselines" in skipped["Table 3"] and "behaviour" in skipped["Table 9"]
    assert "agent.replay" in skipped["Table 6"]  # the synthetic B8 runs are tabular configurations


def _report_with(name: str, markdown: str, path: Path) -> Path:
    path.write_text(f"## Results\n\nSome text.\n\n{name}. A caption.\n\n{markdown}\n\n"
                    "@figure runs/report/figures/x.png | runs/report/figures/y.png\nFigure 2. Two panels.\n",
                    encoding="utf-8")
    return path


def test_compare_with_report_and_the_report_readers(runs, tmp_path):
    _representation_runs(runs)
    md = rt.table_representation(runs)
    report = _report_with("Table 4", md, tmp_path / "report.md")
    assert rt.compare_with_report("Table 4", md, report) == []
    assert rt.compare_with_report("Table 5", md, report) == [f"Table 5 is not in {report}"]
    altered = md.replace("| A: 11 features, MLP | 6.00", "| A: 11 features, MLP | 6.01", 1)
    assert altered != md
    diffs = rt.compare_with_report("Table 4", altered, report)
    assert len(diffs) == 1 and "row 1 col 1" in diffs[0] and "'6.01" in diffs[0] and "'6.00" in diffs[0]
    assert rt.report_captions(report) == {"Table 4": "A caption.", "Figure 2": "Two panels."}
    assert rt.report_figures(report) == {"Figure 2": ["runs/report/figures/x.png", "runs/report/figures/y.png"]}


def test_cli_writes_the_tables_and_reports_differences(runs, tmp_path, capsys):
    _representation_runs(runs)
    out = tmp_path / "out"
    assert rt.main(["--runs-dir", str(runs), "--out-dir", str(out)]) == 0
    assert (out / "report_table_4.md").read_text(encoding="utf-8").rstrip("\n") == rt.table_representation(runs)
    assert "Table 3: skipped" in capsys.readouterr().out
    report = _report_with("Table 4", rt.table_representation(runs).replace("| 6.00 ±", "| 6.01 ±", 1), tmp_path / "r.md")
    assert rt.main(["--runs-dir", str(runs), "--out-dir", str(out), "--report", str(report)]) == 1
    assert "Table 4: 1 differences" in capsys.readouterr().out
    assert rt.main(["--runs-dir", str(tmp_path / "nowhere"), "--out-dir", str(out)]) == 2

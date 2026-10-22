"""Every table and figure of the report in one command
(``snake_rl/analysis/report.py``, M10 spec section 4): the ``ITEMS``
invariants, ``build`` on synthetic runs, and the CLI.

Synthetic runs live under ``tmp_path/runs/<matrix name>/<seed>/`` and are
written with ``tests.test_aggregate.write_run`` (three final-evaluation
episodes per seed, so that the experiment rows agree with the three-episode
baseline files on board, evaluation seed and episode count and the table
does not warn about comparability); ``values`` items use the DQN runs of
``tests.test_m7_value_curve`` (value columns). Figures are only checked
for existence and a PNG header; their contents are tested in
``tests/test_plots.py``. The evaluation curves have three rows (env_steps
0, 100, 200), so the tests pass ``last_k = 2`` or ``None``, never the
default 5.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import yaml

from snake_rl.analysis import matrix, report
from snake_rl.analysis.aggregate import AggregateWarning, results_table, summarise_baseline, summarise_experiment
from snake_rl.utils import REPO_ROOT, git_hash
from tests.test_aggregate import make_result, write_run
from tests.test_m7_value_curve import points_of, write_dqn_run

EPISODES = [(1, 5, "wall"), (3, 5, "self"), (5, 9, "win")]  # three per seed: n_episodes 3, as the baselines
BASELINE_EPISODES = {
    "random": [(0, 12, "wall"), (1, 30, "wall"), (0, 9, "self")],
    "bfs": [(20, 200, "self"), (30, 300, "self"), (40, 250, "wall")],
    "hamiltonian": [(61, 2000, "win"), (61, 2000, "win"), (61, 2000, "win")],
}
PNG = b"\x89PNG\r\n\x1a\n"
THREE = ("random", "bfs", "hamiltonian")

#: Item name -> experiment names in order (spec section 4); the summary items are every row of the board.
EXPERIMENTS_OF = {
    "partC_W8": ["tabular_q_W8", "tabular_q_W8_gamma0", "dqn_A_W8"],
    "partC_W10": ["tabular_q_W10", "tabular_q_W10_gamma0", "dqn_A_W10"],
    "obs_W8": ["dqn_A_W8", "dqn_B3_W8", "dqn_B7_W8", "dqn_B8_W8"],
    "obs_W10": ["dqn_A_W10", "dqn_B3_W10", "dqn_B7_W10", "dqn_B8_W10"],
    "gamma_W8": ["dqn_A_W8", "dqn_A_W8_gamma099", "dqn_B8_W8", "dqn_B8_W8_gamma099"],
    "gamma_W10": ["dqn_A_W10", "dqn_B8_W10", "dqn_B8_W10_gamma099"],
    "components_A_W8": ["dqn_A_W8", "dqn_A_W8_noreplay", "dqn_A_W8_noreplay_t31", "dqn_A_W8_notarget",
                        "dqn_A_W8_noreplay_notarget"],
    "components_B8_W8": ["dqn_B8_W8", "dqn_B8_W8_noreplay", "dqn_B8_W8_noreplay_t31", "dqn_B8_W8_notarget",
                         "dqn_B8_W8_noreplay_notarget"],
    "components_A_W10": ["dqn_A_W10", "dqn_A_W10_noreplay_t31", "dqn_A_W10_notarget"],
    "components_B8_W10": ["dqn_B8_W10", "dqn_B8_W10_noreplay", "dqn_B8_W10_noreplay_t31", "dqn_B8_W10_notarget"],
    "budget_W8": ["dqn_A_W8", "dqn_A_W8_G1", "dqn_A_W8_G4", "dqn_A_W8_G8_b1024", "dqn_B8_W8", "dqn_B8_W8_G1",
                  "dqn_B8_W8_G1_t31"],
    "budget_W10": ["dqn_B8_W10", "dqn_B8_W10_G1_t31"],
    "capacity_W8": ["dqn_A_W8", "dqn_A_W8_h16", "dqn_A_W8_linear"],
    "variants_A_W8": ["dqn_A_W8", "dqn_A_W8_double", "dqn_A_W8_dueling", "dqn_A_W8_per"],
    "variants_B8_W8": ["dqn_B8_W8", "dqn_B8_W8_double", "dqn_B8_W8_dueling", "dqn_B8_W8_per"],
    "values_double_A_W8": ["dqn_A_W8_gamma099", "dqn_A_W8_gamma099_double"],
    "values_double_B8_W8": ["dqn_B8_W8_gamma099", "dqn_B8_W8_gamma099_double"],
    "reward_A_W8_gamma09": ["dqn_A_W8", "dqn_A_W8_R1", "dqn_A_W8_R2", "dqn_A_W8_R3"],
    "reward_A_W8_gamma099": ["dqn_A_W8_gamma099", "dqn_A_W8_gamma099_R1", "dqn_A_W8_gamma099_R2",
                             "dqn_A_W8_gamma099_R3", "dqn_A_W8_gamma099_R3sym", "dqn_A_W8_gamma099_R3_H1000"],
    "reward_B8_W8_gamma09": ["dqn_B8_W8", "dqn_B8_W8_R1", "dqn_B8_W8_R2", "dqn_B8_W8_R3"],
    "reward_B8_W8_gamma099": ["dqn_B8_W8_gamma099", "dqn_B8_W8_gamma099_R1", "dqn_B8_W8_gamma099_R2",
                              "dqn_B8_W8_gamma099_R3", "dqn_B8_W8_gamma099_R3sym", "dqn_B8_W8_gamma099_R3_H1000"],
    "reward_B8_W10_gamma099": ["dqn_B8_W10_gamma099", "dqn_B8_W10_gamma099_R1", "dqn_B8_W10_gamma099_R2",
                               "dqn_B8_W10_gamma099_R3"],
    "probe_W8": ["dqn_B8_W8_gamma099", "dqn_B8_W8_gamma099_R3", "dqn_B8_W8_gamma099_R3_c01"],
    "summary_W8": [e.name for e in matrix.experiments(W=8)],
    "summary_W10": [e.name for e in matrix.experiments(W=10)],
}
#: Item name -> (table, curve, values, causes).
FLAGS = {
    **{name: (True, True, False, False) for name in EXPERIMENTS_OF},
    "gamma_W8": (True, True, True, False), "gamma_W10": (True, True, True, False),
    "values_double_A_W8": (False, False, True, False), "values_double_B8_W8": (False, False, True, False),
    **{name: (True, True, False, True) for name in EXPERIMENTS_OF if name.startswith(("reward_", "probe_"))},
    "summary_W8": (True, False, False, False), "summary_W10": (True, False, False, False),
}
#: Item name -> baselines where the specification lists them.
BASELINES_OF = {
    "partC_W8": THREE, "partC_W10": THREE, "summary_W8": THREE, "summary_W10": THREE,
    **{name: ("bfs",) for name in EXPERIMENTS_OF if name.startswith(("obs_", "gamma_", "reward_", "probe_"))},
}


# ------------------------------------------------------------------- builders
def curve_of(seed: int) -> list[tuple[int, float]]:
    return [(0, 0.0), (100, float(seed)), (200, 2.0 * seed + 1.0)]


def write_exp(runs: Path, name: str, seeds=(0, 1)) -> Path:
    for seed in seeds:
        write_run(runs / name, seed, EPISODES, curve_of(seed))
    return runs / name


def write_w8_baselines(runs: Path, names=THREE) -> dict[str, Path]:
    return {name: make_result(BASELINE_EPISODES[name], name).save(runs / "baselines" / f"{name}_W8_relative.json")
            for name in names}


def item(name: str) -> "report.ReportItem":
    (found,) = [i for i in report.ITEMS if i.name == name]
    return found


def aggregate_messages(record) -> list[str]:
    return [str(w.message) for w in record if issubclass(w.category, AggregateWarning)]


def expected_table(runs: Path, name: str, last_k: int | None) -> str:
    """``results_table`` of the baseline rows and the experiment rows that exist, with the item's labels."""
    it = item(name)
    baseline_labels = {b.name: b.label for b in matrix.baselines(W=it.W)}
    rows = []
    for b in it.baselines:
        path = runs / "baselines" / f"{b}_W{it.W}_relative.json"
        if path.is_file():
            rows.append(summarise_baseline(path, label=baseline_labels[b]))
    for exp_name, label in it.experiments:
        if (runs / exp_name).is_dir():
            rows.append(summarise_experiment(runs / exp_name, label=label, last_k=last_k))
    return results_table(rows)


def heading_name(line: str) -> str | None:
    """``## partC_W8`` or ``## partC_W8 - Part C`` -> ``partC_W8``; None for other lines."""
    match = re.match(r"## (\w+)", line)
    return match.group(1) if match else None


def section(index_text: str, name: str) -> list[str]:
    """The lines of ``## <name>`` up to the next ``## `` heading."""
    lines = index_text.splitlines()
    starts = [i for i, line in enumerate(lines) if heading_name(line) == name]
    assert len(starts) == 1, (name, starts)
    body = []
    for line in lines[starts[0] + 1:]:
        if line.startswith("## "):
            break
        body.append(line)
    return body


def cli(argv: list[str], capsys) -> tuple[int, str, str]:
    try:
        code = report.main(argv)
    except SystemExit as exit_:
        code = exit_.code
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.fixture
def partc(tmp_path: Path) -> Path:
    """tabular_q_W8 (5 seeds), tabular_q_W8_gamma0 (2 seeds), the three 8x8 baselines; dqn_A_W8 is missing."""
    runs = tmp_path / "runs"
    write_exp(runs, "tabular_q_W8", seeds=(0, 1, 2, 3, 4))
    write_exp(runs, "tabular_q_W8_gamma0", seeds=(0, 1))
    write_w8_baselines(runs)
    return runs


# --------------------------------------------------------------------- ITEMS
def test_item_names_are_the_specified_ones():
    assert isinstance(report.ITEMS, tuple)
    names = [i.name for i in report.ITEMS]
    assert len(set(names)) == len(names)
    assert set(names) == set(EXPERIMENTS_OF)
    assert all(isinstance(i, report.ReportItem) for i in report.ITEMS)


def test_item_experiments_are_matrix_rows_of_the_board():
    for it in report.ITEMS:
        assert it.W in (8, 10) and re.search(r"_W(\d+)", it.name).group(1) == str(it.W), it.name
        for exp_name, label in it.experiments:
            exp = matrix.experiment(exp_name)  # ValueError if not in the matrix
            assert exp.W == it.W, (it.name, exp_name)
            assert isinstance(label, str) and label
        labels = [label for _, label in it.experiments]
        assert len(set(labels)) == len(labels), it.name
        exp_names = [exp_name for exp_name, _ in it.experiments]
        assert len(set(exp_names)) == len(exp_names), it.name


def test_item_baselines_exist_for_the_board():
    known = {(b.name, b.W) for b in matrix.BASELINES}
    for it in report.ITEMS:
        assert isinstance(it.baselines, tuple)
        for name in it.baselines:
            assert name in THREE and (name, it.W) in known, (it.name, name)
        assert len(set(it.baselines)) == len(it.baselines)


def test_curve_items_have_at_most_eight_experiments():
    for it in report.ITEMS:
        if it.curve:
            assert 1 <= len(it.experiments) <= 8, it.name
        assert len(it.experiments) >= 1, it.name


def test_value_and_cause_items_have_only_dqn_experiments():
    for it in report.ITEMS:
        if it.values or it.causes:
            assert all(matrix.experiment(n).kind == "dqn" for n, _ in it.experiments), it.name


def test_every_matrix_experiment_appears_in_an_item():
    covered = {n for it in report.ITEMS for n, _ in it.experiments}
    assert covered == {e.name for e in matrix.MATRIX}
    for W in (8, 10):
        assert [n for n, _ in item(f"summary_W{W}").experiments] == [e.name for e in matrix.experiments(W=W)]
        assert [label for _, label in item(f"summary_W{W}").experiments] == [e.label for e in matrix.experiments(W=W)]


def test_items_by_hand():
    for name, experiments in EXPERIMENTS_OF.items():
        it = item(name)
        assert [n for n, _ in it.experiments] == experiments, name
        assert (it.table, it.curve, it.values, it.causes) == FLAGS[name], name
        if name in BASELINES_OF:
            assert it.baselines == BASELINES_OF[name], name
        assert it.title is None or isinstance(it.title, str)


def test_report_item_defaults():
    it = report.ReportItem(name="x", W=8, experiments=(("dqn_A_W8", "A"),))
    assert (it.baselines, it.table, it.curve, it.values, it.causes, it.title) == ((), True, True, False, False, None)
    with pytest.raises(AttributeError):
        it.name = "y"  # type: ignore[misc]


# --------------------------------------------------------------------- build
def test_build_partc_writes_table_and_curve_and_lists_the_missing_experiment(partc, tmp_path):
    runs, out = partc, tmp_path / "report"
    with pytest.warns(AggregateWarning) as record:
        results = report.build(runs, out, only=["partC_W8"], last_k=2)
    messages = aggregate_messages(record)
    assert any("partC_W8" in m and "dqn_A_W8" in m for m in messages), messages
    assert not any("same conditions" in m for m in messages)  # three episodes everywhere: the rows agree
    (r,) = [r for r in results if r.item.name == "partC_W8"]
    assert isinstance(r, report.ItemResult) and r.item == item("partC_W8")
    assert r.used == ("tabular_q_W8", "tabular_q_W8_gamma0") and r.missing == ("dqn_A_W8",)
    assert r.seeds == {"tabular_q_W8": 5, "tabular_q_W8_gamma0": 2}
    assert set(r.outputs) == {"table", "curve"}
    assert r.outputs["table"] == out / "tables" / "partC_W8.md"
    assert r.outputs["curve"] == out / "figures" / "partC_W8_curve.png"
    assert all(isinstance(p, Path) and p.is_file() for p in r.outputs.values())
    assert r.outputs["curve"].read_bytes()[:8] == PNG and r.outputs["curve"].stat().st_size > 1000
    table = r.outputs["table"].read_text(encoding="utf-8")
    assert table.rstrip("\n") == expected_table(runs, "partC_W8", 2)
    lines = table.rstrip("\n").splitlines()
    assert lines[0].endswith("| last-2 evals mean +- 95% CI |") and len(lines) == 2 + 3 + 2
    labels = dict(item("partC_W8").experiments)
    assert lines[5].startswith(f"| {labels['tabular_q_W8']} | 5 | 3 | ")
    assert lines[6].startswith(f"| {labels['tabular_q_W8_gamma0']} | 2 | 3 | ")
    # only the requested item was built
    assert sorted(p.name for p in (out / "tables").iterdir()) == ["partC_W8.md"]
    assert sorted(p.name for p in (out / "figures").iterdir()) == ["partC_W8_curve.png"]
    assert (out / "index.md").is_file() and (out / "manifest.json").is_file()


def test_build_index_content(partc, tmp_path):
    runs, out = partc, tmp_path / "report"
    with pytest.warns(AggregateWarning):
        report.build(runs, out, only=["partC_W8"], last_k=2)
    text = (out / "index.md").read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0] == "# Report"
    head = [line for line in lines[1:] if git_hash() in line]
    assert len(head) == 1, "one line with the runs directory, the git hash and last_k"
    assert str(runs) in head[0] or runs.as_posix() in head[0]
    rest = head[0].replace(git_hash(), "").replace(str(runs), "").replace(runs.as_posix(), "")
    assert "2" in rest  # last_k
    body = section(text, "partC_W8")
    assert any("tables/partC_W8.md" in line for line in body)
    assert any("figures/partC_W8_curve.png" in line for line in body)
    labels = dict(item("partC_W8").experiments)
    assert f"- {labels['tabular_q_W8']}: tabular_q_W8 (5 seeds)" in body
    assert f"- {labels['tabular_q_W8_gamma0']}: tabular_q_W8_gamma0 (2 seeds)" in body
    assert "- missing: dqn_A_W8" in body
    assert not any("skipped" in line for line in body)
    assert any(line.startswith("## Skipped") for line in lines)
    assert "partC_W8" not in "\n".join(section(text, "Skipped"))


def test_build_manifest_content(partc, tmp_path):
    runs, out = partc, tmp_path / "report"
    path = runs / "tabular_q_W8_gamma0" / "1" / "config.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    del config["run"]["git"]  # a run without a recorded hash
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.warns(AggregateWarning):
        report.build(runs, out, only=["partC_W8"], last_k=2)
    raw = (out / "manifest.json").read_text(encoding="utf-8")
    assert raw.startswith("{\n  ")  # indent=2
    m = json.loads(raw)
    assert m["runs_dir"] in (str(runs), runs.as_posix())
    assert m["git"] == git_hash() and m["last_k"] == 2
    items = {i["name"]: i for i in m["items"]}
    it = items["partC_W8"]
    assert it["W"] == 8
    assert it["outputs"] == {"table": "tables/partC_W8.md", "curve": "figures/partC_W8_curve.png"}
    assert it["missing"] == ["dqn_A_W8"]
    labels = dict(item("partC_W8").experiments)
    used = [e for e in it["experiments"] if e["seeds"]]
    assert [(e["name"], e["label"]) for e in used] == [
        ("tabular_q_W8", labels["tabular_q_W8"]), ("tabular_q_W8_gamma0", labels["tabular_q_W8_gamma0"])]
    assert used[0]["seeds"] == [0, 1, 2, 3, 4] and used[0]["git"] == ["abc1234"] * 5
    assert used[1]["seeds"] == [0, 1] and used[1]["git"] == ["abc1234", "unknown"]
    assert all(isinstance(s, int) for e in used for s in e["seeds"])


def test_build_without_last_k_has_no_tail_column(partc, tmp_path):
    runs, out = partc, tmp_path / "report"
    with pytest.warns(AggregateWarning):
        (r,) = [r for r in report.build(runs, out, only=["partC_W8"], last_k=None) if r.item.name == "partC_W8"]
    lines = r.outputs["table"].read_text(encoding="utf-8").rstrip("\n").splitlines()
    assert lines[0].count("|") == 13 and lines[0].endswith("| win |")
    assert "\n".join(lines) == expected_table(runs, "partC_W8", None)
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8"))["last_k"] is None


def test_build_skips_an_item_with_nothing_used(partc, tmp_path):
    runs, out = partc, tmp_path / "report"
    with pytest.warns(AggregateWarning) as record:
        results = report.build(runs, out, only=["partC_W8", "capacity_W8"], last_k=2)
    messages = aggregate_messages(record)
    assert any("capacity_W8" in m and all(n in m for n in EXPERIMENTS_OF["capacity_W8"]) for m in messages), messages
    by = {r.item.name: r for r in results}
    cap = by["capacity_W8"]
    assert cap.outputs == {} and cap.used == () and cap.seeds == {}
    assert cap.missing == tuple(EXPERIMENTS_OF["capacity_W8"])
    assert set(by["partC_W8"].outputs) == {"table", "curve"}
    assert not (out / "tables" / "capacity_W8.md").exists()
    assert not (out / "figures" / "capacity_W8_curve.png").exists()
    text = (out / "index.md").read_text(encoding="utf-8")
    assert "skipped: no finished run" in section(text, "capacity_W8")
    skipped = "\n".join(section(text, "Skipped"))
    assert "capacity_W8" in skipped and "partC_W8" not in skipped
    items = {i["name"]: i for i in json.loads((out / "manifest.json").read_text(encoding="utf-8"))["items"]}
    assert items["capacity_W8"]["outputs"] == {} and items["capacity_W8"]["missing"] == EXPERIMENTS_OF["capacity_W8"]


def test_build_leaves_out_a_missing_baseline_with_a_warning(tmp_path):
    runs, out = tmp_path / "runs", tmp_path / "report"
    write_exp(runs, "tabular_q_W8")
    write_w8_baselines(runs, names=("bfs",))
    with pytest.warns(AggregateWarning) as record:
        (r,) = [r for r in report.build(runs, out, only=["partC_W8"], last_k=2) if r.item.name == "partC_W8"]
    messages = aggregate_messages(record)
    assert any("random_W8_relative.json" in m for m in messages), messages
    assert any("hamiltonian_W8_relative.json" in m for m in messages), messages
    assert set(r.outputs) == {"table", "curve"}  # the item is still built
    lines = r.outputs["table"].read_text(encoding="utf-8").rstrip("\n").splitlines()
    assert len(lines) == 2 + 1 + 1
    assert lines[2].startswith("| BFS greedy | 1 | 3 | 30.00 +- ")
    assert "\n".join(lines) == expected_table(runs, "partC_W8", 2)
    assert r.outputs["curve"].read_bytes()[:8] == PNG


def test_build_values_item(tmp_path):
    runs, out = tmp_path / "runs", tmp_path / "report"
    for name in EXPERIMENTS_OF["values_double_A_W8"]:
        for seed in (2, 0, 1):
            write_dqn_run(runs / name, seed, points_of(seed))
    with warnings.catch_warnings():
        warnings.simplefilter("error", AggregateWarning)  # nothing is missing: no warning
        results = report.build(runs, out, only=["values_double_A_W8"], last_k=2)
    (r,) = [r for r in results if r.item.name == "values_double_A_W8"]
    assert set(r.outputs) == {"values"}
    assert r.outputs["values"] == out / "figures" / "values_double_A_W8_values.png"
    assert r.outputs["values"].read_bytes()[:8] == PNG and r.outputs["values"].stat().st_size > 1000
    assert r.used == tuple(EXPERIMENTS_OF["values_double_A_W8"]) and r.missing == ()
    assert r.seeds == {name: 3 for name in EXPERIMENTS_OF["values_double_A_W8"]}
    assert not (out / "tables" / "values_double_A_W8.md").exists()
    assert not (out / "figures" / "values_double_A_W8_curve.png").exists()
    items = {i["name"]: i for i in json.loads((out / "manifest.json").read_text(encoding="utf-8"))["items"]}
    assert items["values_double_A_W8"]["outputs"] == {"values": "figures/values_double_A_W8_values.png"}
    body = section((out / "index.md").read_text(encoding="utf-8"), "values_double_A_W8")
    assert any("figures/values_double_A_W8_values.png" in line for line in body)


def test_build_causes_item(tmp_path):
    runs, out = tmp_path / "runs", tmp_path / "report"
    write_exp(runs, "dqn_A_W8", seeds=(0, 1, 2))
    write_exp(runs, "dqn_A_W8_R1")
    write_w8_baselines(runs, names=("bfs",))
    with pytest.warns(AggregateWarning) as record:
        (r,) = [r for r in report.build(runs, out, only=["reward_A_W8_gamma09"], last_k=2)
                if r.item.name == "reward_A_W8_gamma09"]
    messages = aggregate_messages(record)
    assert any("reward_A_W8_gamma09" in m and "dqn_A_W8_R2" in m and "dqn_A_W8_R3" in m for m in messages), messages
    assert set(r.outputs) == {"table", "curve", "causes"}
    assert r.outputs["causes"] == out / "figures" / "reward_A_W8_gamma09_causes.png"
    assert r.outputs["curve"] == out / "figures" / "reward_A_W8_gamma09_curve.png"
    assert r.outputs["table"] == out / "tables" / "reward_A_W8_gamma09.md"
    for kind in ("curve", "causes"):
        assert r.outputs[kind].read_bytes()[:8] == PNG and r.outputs[kind].stat().st_size > 1000
    assert r.used == ("dqn_A_W8", "dqn_A_W8_R1") and r.missing == ("dqn_A_W8_R2", "dqn_A_W8_R3")
    assert r.seeds == {"dqn_A_W8": 3, "dqn_A_W8_R1": 2}
    assert r.outputs["table"].read_text(encoding="utf-8").rstrip("\n") == expected_table(runs, "reward_A_W8_gamma09", 2)


def test_build_every_item(partc, tmp_path):
    runs, out = partc, tmp_path / "report"
    with pytest.warns(AggregateWarning) as record:
        results = report.build(runs, out, last_k=None)
    assert [r.item.name for r in results] == [i.name for i in report.ITEMS]
    by = {r.item.name: r for r in results}
    assert set(by["partC_W8"].outputs) == {"table", "curve"}
    assert set(by["summary_W8"].outputs) == {"table"}
    built = {"partC_W8", "summary_W8"}
    assert all(r.outputs == {} and r.used == () for r in results if r.item.name not in built)
    assert sorted(p.name for p in (out / "tables").iterdir()) == ["partC_W8.md", "summary_W8.md"]
    assert sorted(p.name for p in (out / "figures").iterdir()) == ["partC_W8_curve.png"]
    summary = by["summary_W8"].outputs["table"].read_text(encoding="utf-8").rstrip("\n")
    assert summary == expected_table(runs, "summary_W8", None)
    assert len(summary.splitlines()) == 2 + 3 + 2
    assert by["summary_W8"].missing == tuple(n for n in EXPERIMENTS_OF["summary_W8"]
                                             if n not in ("tabular_q_W8", "tabular_q_W8_gamma0"))
    messages = aggregate_messages(record)
    assert any("partC_W8" in m and "dqn_A_W8" in m for m in messages)
    assert any("summary_W8" in m and "dqn_B8_W8_gamma099_R3_c01" in m for m in messages)
    text = (out / "index.md").read_text(encoding="utf-8")
    for it in report.ITEMS:
        section(text, it.name)  # exactly one section per item
    headings = [heading_name(line) for line in text.splitlines() if line.startswith("## ")]
    assert headings == [i.name for i in report.ITEMS] + ["Skipped"]  # in ITEMS order, the skipped list last
    skipped = "\n".join(section(text, "Skipped"))
    assert all(name in skipped for name in EXPERIMENTS_OF if name not in built)
    assert "partC_W8" not in skipped and "summary_W8" not in skipped
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert [i["name"] for i in m["items"]] == [i.name for i in report.ITEMS] and m["last_k"] is None


def test_build_creates_directories_and_overwrites(partc, tmp_path):
    runs, out = partc, tmp_path / "deep" / "nested" / "report"
    with pytest.warns(AggregateWarning):
        first = report.build(runs, out, only=["partC_W8"], last_k=2)
    (out / "tables" / "partC_W8.md").write_text("stale", encoding="utf-8")
    with pytest.warns(AggregateWarning):
        second = report.build(runs, out, only=["partC_W8"], last_k=2)
    r1, r2 = [[r for r in rs if r.item.name == "partC_W8"][0] for rs in (first, second)]
    assert r1.outputs == r2.outputs and r1.used == r2.used and r1.seeds == r2.seeds
    assert r2.outputs["table"].read_text(encoding="utf-8").rstrip("\n") == expected_table(runs, "partC_W8", 2)
    assert str(runs) in (out / "index.md").read_text(encoding="utf-8") or runs.as_posix() in (
        out / "index.md").read_text(encoding="utf-8")


def test_build_rejects_an_unknown_item_name(partc, tmp_path):
    out = tmp_path / "report"
    with pytest.raises(ValueError, match="nope"):
        report.build(partc, out, only=["partC_W8", "nope"], last_k=2)
    with pytest.raises(ValueError, match="partc_w8"):
        report.build(partc, out, only=("partc_w8",), last_k=2)  # names are case sensitive
    assert not (out / "tables").exists()


# ----------------------------------------------------------------------- CLI
def test_cli_builds_and_prints_one_line_per_item(partc, tmp_path, capsys):
    runs, out = partc, tmp_path / "report"
    with pytest.warns(AggregateWarning):
        code, stdout, err = cli(["--runs-dir", str(runs), "--out-dir", str(out), "--only", "partC_W8", "capacity_W8",
                                 "--last-k", "2"], capsys)
    assert code == 0
    lines = stdout.splitlines()
    assert "partC_W8: table, curve" in lines
    assert "capacity_W8: skipped" in lines
    assert any("index.md" in line for line in lines)
    assert (out / "tables" / "partC_W8.md").read_text(encoding="utf-8").rstrip("\n") == expected_table(runs, "partC_W8", 2)
    assert (out / "figures" / "partC_W8_curve.png").read_bytes()[:8] == PNG
    assert not (out / "tables" / "capacity_W8.md").exists()
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8"))["last_k"] == 2


def test_cli_last_k_zero_means_no_column(partc, tmp_path, capsys):
    runs, out = partc, tmp_path / "report"
    with pytest.warns(AggregateWarning):
        code, stdout, err = cli(["--runs-dir", str(runs), "--out-dir", str(out), "--only", "partC_W8", "--last-k", "0"],
                                capsys)
    assert code == 0
    lines = (out / "tables" / "partC_W8.md").read_text(encoding="utf-8").rstrip("\n").splitlines()
    assert lines[0].count("|") == 13 and "\n".join(lines) == expected_table(runs, "partC_W8", None)
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8"))["last_k"] is None


def test_cli_list_prints_the_items_and_builds_nothing(tmp_path, capsys):
    runs, out = tmp_path / "runs", tmp_path / "report"
    code, stdout, err = cli(["--list", "--runs-dir", str(runs), "--out-dir", str(out)], capsys)
    assert code == 0
    lines = stdout.splitlines()
    assert len(lines) == len(report.ITEMS)
    for line, it in zip(lines, report.ITEMS):
        assert line.startswith(f"{it.name}: "), line
        assert str(it.W) in line
        assert all(exp_name in line for exp_name, _ in it.experiments), line
    assert not out.exists() and not runs.exists()


def test_cli_unknown_only_name_is_a_usage_error(partc, tmp_path, capsys):
    out = tmp_path / "report"
    code, stdout, err = cli(["--runs-dir", str(partc), "--out-dir", str(out), "--only", "partC_W8", "nope"], capsys)
    assert code == 2 and stdout == "" and "nope" in err
    assert not out.exists()


def test_module_entry_point_list():
    proc = subprocess.run(
        [sys.executable, "-m", "snake_rl.analysis.report", "--list"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert [line.split(":")[0] for line in proc.stdout.strip().splitlines()] == [i.name for i in report.ITEMS]

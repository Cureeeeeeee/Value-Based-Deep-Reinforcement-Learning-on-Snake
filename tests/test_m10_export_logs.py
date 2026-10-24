"""``scripts/export_logs.py`` (M10 spec section 5): the light-weight logs of a
runs directory as one zip for the notebook.

A synthetic runs tree in ``tmp_path`` holds, next to the log files that must
be kept, every kind of file that must not be: weights (``.pt``), tables
(``.npz``), anything under a ``checkpoints`` directory, the top-level
directories of ``EXCLUDED_DIRS`` (also when they contain a file named like a
log), ``runs/behaviour`` unless asked for, and a nested baseline JSON. Every
file has a distinct size, so the byte count of the result is the sum of
the kept files only.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "export_logs.py"


def _load_script() -> ModuleType:
    """``scripts/`` is not a package: import the file by its path."""
    spec = importlib.util.spec_from_file_location("export_logs_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


el = _load_script()

#: Kept without --include-behaviour (paths under the runs directory).
KEPT = [
    "baselines/bfs_W8_relative.json",
    "baselines/random_W8_relative.json",
    "dqn_A_W8/0/config.yaml",
    "dqn_A_W8/0/final_eval.json",
    "dqn_A_W8/0/metrics.csv",
    "dqn_A_W8/0/values.csv",
    "dqn_A_W8/1/aborted.json",
    "dqn_A_W8/1/config.yaml",
    "dqn_A_W8/1/metrics.csv",
    "figures/partD_gui_agent.png",
    "sweep_tabular_W8/alpha=0.1/0/config.yaml",
    "sweep_tabular_W8/alpha=0.1/0/final_eval.json",
    "sweep_tabular_W8/alpha=0.1/0/metrics.csv",
]
#: Kept only with --include-behaviour.
BEHAVIOUR = [
    "behaviour/dqn_A_W8/heatmap.png",
    "behaviour/dqn_A_W8/loops.csv",
    "behaviour/dqn_A_W8/notes.md",
    "behaviour/dqn_A_W8/stats.json",
    "behaviour/dqn_A_W8/visits.npy",
]
#: Never kept.
DROPPED = [
    "README.md",
    "baselines/notes.txt",
    "baselines/old/bfs_W8_relative.json",
    "behaviour/dqn_A_W8/weights.pt",
    "behaviour/dqn_A_W8/readme.txt",
    "dqn_A_W8/0/model.pt",
    "dqn_A_W8/0/checkpoints/step_12800.pt",
    "dqn_A_W8/0/checkpoints/metrics.csv",
    "dqn_A_W8/0/checkpoints/config.yaml",
    "dqn_A_W8/3/notes.txt",
    "sweep_tabular_W8/alpha=0.1/0/q_table.npz",
    "_interrupted/dqn_A_W8/0/config.yaml",
    "_interrupted/dqn_A_W8/0/metrics.csv",
    "bench/update.csv",
    "figures/learning_curves.png",
    "recordings/dqn.gif",
    "report/index.md",
    "report/config.yaml",
    "report/tables/partC_W8.md",
    "review/notes.md",
    "tables/part_c.md",
]
EMPTY_DIRS = ["dqn_A_W8/2", "behaviour/empty"]
N_RUNS = 3  # dqn_A_W8/0, dqn_A_W8/1 (config.yaml), sweep_tabular_W8/alpha=0.1/0


def build_tree(root: Path) -> Path:
    for i, rel in enumerate([*KEPT, *BEHAVIOUR, *DROPPED]):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{rel}\n".encode("utf-8") * (i + 1))  # distinct sizes
    for rel in EMPTY_DIRS:
        (root / rel).mkdir(parents=True, exist_ok=True)
    return root


def expected_entries(name: str = "runs", behaviour: bool = False) -> list[str]:
    return sorted(f"{name}/{rel}" for rel in [*KEPT, *(BEHAVIOUR if behaviour else [])])


def size_of(root: Path, rels) -> int:
    return sum((root / rel).stat().st_size for rel in rels)


def names_in(out: Path) -> list[str]:
    with zipfile.ZipFile(out) as zf:
        return zf.namelist()


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    return build_tree(tmp_path / "runs")


def cli(argv: list[str], capsys) -> tuple[int, str, str]:
    try:
        code = el.main(argv)
    except SystemExit as exit_:
        code = exit_.code
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# ------------------------------------------------------------------ constants
def test_constants():
    assert el.LOG_FILES == ("config.yaml", "metrics.csv", "final_eval.json", "values.csv", "aborted.json")
    assert el.EXCLUDED_DIRS == ("_interrupted", "bench", "figures", "recordings", "report", "review", "tables")
    assert el.BEHAVIOUR_SUFFIXES == (".json", ".csv", ".png", ".md", ".npy")
    assert el.REPORT_FILES == ("figures/partD_gui_agent.png",)


# ---------------------------------------------------------------- export_logs
def test_export_keeps_the_log_files_and_nothing_else(runs, tmp_path):
    out = tmp_path / "runs_logs.zip"
    stats = el.export_logs(runs, out)
    names = names_in(out)
    assert names == expected_entries()
    assert not any(n.endswith((".pt", ".npz")) or "/checkpoints/" in n for n in names)
    report_files = {f"runs/{rel}" for rel in el.REPORT_FILES}
    assert not any(n.split("/")[1] in el.EXCLUDED_DIRS or n.split("/")[1] == "behaviour"
                   for n in names if n not in report_files)
    assert "runs/figures/learning_curves.png" not in names  # the rest of runs/figures is regenerable
    assert "runs/baselines/old/bfs_W8_relative.json" not in names and "runs/baselines/notes.txt" not in names
    assert stats == {"files": len(KEPT), "runs": N_RUNS, "bytes": size_of(runs, KEPT)}
    assert all(type(v) is int for v in stats.values())
    with zipfile.ZipFile(out) as zf:
        assert all(info.compress_type == zipfile.ZIP_DEFLATED for info in zf.infolist())
        assert sum(info.file_size for info in zf.infolist()) == stats["bytes"]
        for rel in KEPT:
            assert zf.read(f"runs/{rel}") == (runs / rel).read_bytes(), rel
        assert zf.testzip() is None


def test_include_behaviour(runs, tmp_path):
    out = tmp_path / "with_behaviour.zip"
    stats = el.export_logs(runs, out, include_behaviour=True)
    names = names_in(out)
    assert names == expected_entries(behaviour=True)
    assert "runs/behaviour/dqn_A_W8/weights.pt" not in names and "runs/behaviour/dqn_A_W8/readme.txt" not in names
    assert stats == {"files": len(KEPT) + len(BEHAVIOUR), "runs": N_RUNS, "bytes": size_of(runs, [*KEPT, *BEHAVIOUR])}
    assert el.export_logs(runs, out, include_behaviour=False)["files"] == len(KEPT)


def test_entries_are_relative_to_the_parent_of_runs_dir(tmp_path):
    other = build_tree(tmp_path / "logs_copy")
    out = tmp_path / "copy.zip"
    el.export_logs(other, out)
    assert names_in(out) == expected_entries("logs_copy")
    assert el.export_logs(str(other), str(out))["files"] == len(KEPT)  # str paths too
    target = tmp_path / "unzipped"
    with zipfile.ZipFile(out) as zf:
        zf.extractall(target)
    assert (target / "logs_copy" / "dqn_A_W8" / "0" / "metrics.csv").read_bytes() == (
        other / "dqn_A_W8" / "0" / "metrics.csv").read_bytes()
    assert sorted(p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()) == expected_entries("logs_copy")


def test_unzipping_at_the_repo_root_recreates_runs(runs, tmp_path):
    out = tmp_path / "runs_logs.zip"
    el.export_logs(runs, out)
    root = tmp_path / "checkout"
    root.mkdir()
    with zipfile.ZipFile(out) as zf:
        zf.extractall(root)
    assert sorted(p.name for p in (root / "runs").iterdir()) == ["baselines", "dqn_A_W8", "figures", "sweep_tabular_W8"]
    assert sorted(p.name for p in (root / "runs" / "dqn_A_W8" / "0").iterdir()) == [
        "config.yaml", "final_eval.json", "metrics.csv", "values.csv"]
    assert sorted(p.name for p in (root / "runs" / "dqn_A_W8" / "1").iterdir()) == [
        "aborted.json", "config.yaml", "metrics.csv"]


def test_order_is_sorted_and_deterministic(runs, tmp_path):
    a, b = tmp_path / "a.zip", tmp_path / "b.zip"
    el.export_logs(runs, a, include_behaviour=True)
    el.export_logs(runs, b, include_behaviour=True)
    assert names_in(a) == sorted(names_in(a)) == names_in(b)


def test_out_parent_is_created_and_an_existing_file_is_overwritten(runs, tmp_path):
    out = tmp_path / "deep" / "nested" / "logs.zip"
    assert not out.parent.exists()
    first = el.export_logs(runs, out)
    assert names_in(out) == expected_entries()
    out.write_bytes(b"not a zip any more")
    assert el.export_logs(runs, out) == first
    assert names_in(out) == expected_entries()
    with zipfile.ZipFile(out) as zf:
        assert zf.testzip() is None and len(zf.infolist()) == len(KEPT)  # not appended to


def test_runs_dir_must_be_a_directory(tmp_path):
    out = tmp_path / "x.zip"
    with pytest.raises(ValueError):
        el.export_logs(tmp_path / "missing", out)
    (tmp_path / "a_file").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        el.export_logs(tmp_path / "a_file", out)
    assert not out.exists()


def test_an_empty_runs_directory_gives_an_empty_archive(tmp_path):
    (tmp_path / "runs").mkdir()
    out = tmp_path / "empty.zip"
    assert el.export_logs(tmp_path / "runs", out) == {"files": 0, "runs": 0, "bytes": 0}
    assert names_in(out) == []


def test_only_directories_with_a_config_or_a_final_evaluation_count_as_runs(tmp_path):
    runs = tmp_path / "runs"
    for rel in ("a/0/config.yaml", "a/1/final_eval.json", "a/2/metrics.csv", "b/0/config.yaml", "b/0/final_eval.json",
                "baselines/bfs_W8_relative.json", "behaviour/x/stats.json"):
        (runs / rel).parent.mkdir(parents=True, exist_ok=True)
        (runs / rel).write_text(rel, encoding="utf-8")
    stats = el.export_logs(runs, tmp_path / "x.zip", include_behaviour=True)
    assert stats["runs"] == 3 and stats["files"] == 7  # a/0, a/1, b/0; a/2 has only metrics.csv


# ------------------------------------------------------------------------ CLI
def test_cli_writes_the_zip_and_prints_a_summary(runs, tmp_path, capsys):
    out = tmp_path / "logs" / "runs_logs.zip"
    code, stdout, err = cli(["--runs-dir", str(runs), "--out", str(out)], capsys)
    assert code == 0
    assert names_in(out) == expected_entries()
    stats = el.export_logs(runs, tmp_path / "again.zip")
    assert stdout.splitlines() == [
        f"{stats['files']} files from {stats['runs']} run directories, {stats['bytes'] / 1e6:.1f} MB uncompressed -> {out}"
    ]
    assert f"{len(KEPT)} files from {N_RUNS} run directories, 0.0 MB uncompressed -> " in stdout


def test_cli_list_prints_the_entries_and_writes_nothing(runs, tmp_path, capsys):
    out = tmp_path / "runs_logs.zip"
    code, stdout, err = cli(["--runs-dir", str(runs), "--out", str(out), "--list"], capsys)
    assert code == 0
    assert stdout.splitlines() == expected_entries()
    assert not out.exists()
    code, stdout, err = cli(["--runs-dir", str(runs), "--out", str(out), "--list", "--include-behaviour"], capsys)
    assert code == 0
    assert stdout.splitlines() == expected_entries(behaviour=True)
    assert not out.exists()


def test_cli_include_behaviour(runs, tmp_path, capsys):
    out = tmp_path / "runs_logs.zip"
    code, stdout, err = cli(["--runs-dir", str(runs), "--out", str(out), "--include-behaviour"], capsys)
    assert code == 0
    assert names_in(out) == expected_entries(behaviour=True)
    assert stdout.startswith(f"{len(KEPT) + len(BEHAVIOUR)} files from {N_RUNS} run directories, ")


def test_cli_defaults_are_runs_and_runs_logs_zip_in_the_working_directory(runs, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    code, stdout, err = cli([], capsys)
    assert code == 0
    assert names_in(tmp_path / "runs_logs.zip") == expected_entries()
    assert stdout.rstrip("\n").endswith("-> runs_logs.zip")


def test_cli_missing_runs_dir_fails(tmp_path, capsys):
    out = tmp_path / "x.zip"
    try:
        code = el.main(["--runs-dir", str(tmp_path / "missing"), "--out", str(out)])
    except (SystemExit, ValueError) as err:
        code = err.code if isinstance(err, SystemExit) else 1
    assert code not in (0, None)
    assert not out.exists()


def test_the_script_runs_as_a_script_from_any_directory(runs, tmp_path):
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--runs-dir", str(runs), "--list"], capture_output=True, text=True, cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines() == expected_entries()
    out = tmp_path / "from_script.zip"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--runs-dir", str(runs), "--out", str(out)], capture_output=True, text=True,
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert names_in(out) == expected_entries()
    assert proc.stdout.strip().endswith(f"-> {out}")

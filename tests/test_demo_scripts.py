"""scripts/make_demo_pack.py and scripts/demo_check.py (Part F: demonstration on another machine)."""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import demo_check  # noqa: E402
import make_demo_pack  # noqa: E402


def _fake_runs(tmp_path: Path) -> Path:
    runs = tmp_path / "runs"
    for rel in (
        "dqn_B8_W10_gamma099/0/config.yaml",
        "dqn_B8_W10_gamma099/0/model.pt",
        "dqn_B8_W10_gamma099/0/checkpoints/step_00000000.pt",   # not in the manifest: must stay out
        "dqn_B8_W10_gamma099/0/checkpoints/step_00012800.pt",
        "dqn_A_W10/0/config.yaml",
        "dqn_A_W10/0/model.pt",
        "tabular_q_W10/0/q_table.npz",
        "tabular_q_W10/0/config.yaml",
        "behaviour/dqn_B8_W10_gamma099_R3_final/1/hack_1_10029.json",
        "behaviour/dqn_B8_W10_gamma099_R3_final/summary.json",
        "report/figures/gamma_W10_curve.png",
        "figures/partD_gui_agent.png",
        "figures/m5_W8_curve.png",                               # not partD_*: must stay out
    ):
        path = runs / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(rel.encode())
    return runs


def test_select_files_follows_the_manifest(tmp_path: Path) -> None:
    runs = _fake_runs(tmp_path)
    files, missing_required, missing_optional = make_demo_pack.select_files(runs)
    rel = {p.relative_to(runs).as_posix() for p in files}
    assert "dqn_B8_W10_gamma099/0/model.pt" in rel
    assert "dqn_B8_W10_gamma099/0/checkpoints/step_00012800.pt" in rel
    assert "dqn_B8_W10_gamma099/0/checkpoints/step_00000000.pt" not in rel
    assert "tabular_q_W10/0/q_table.npz" in rel and "tabular_q_W10/0/config.yaml" in rel
    assert "behaviour/dqn_B8_W10_gamma099_R3_final/1/hack_1_10029.json" in rel
    assert "figures/partD_gui_agent.png" in rel and "figures/m5_W8_curve.png" not in rel
    assert missing_required == []
    assert "export/runs_logs.zip" in missing_optional
    assert files == sorted(set(files))


def test_select_files_reports_missing_required(tmp_path: Path) -> None:
    runs = _fake_runs(tmp_path)
    (runs / "dqn_A_W10/0/model.pt").unlink()
    _, missing_required, _ = make_demo_pack.select_files(runs)
    assert missing_required == ["dqn_A_W10/0/model.pt"]


def test_write_pack_stores_runs_prefixed_paths(tmp_path: Path) -> None:
    runs = _fake_runs(tmp_path)
    files, _, _ = make_demo_pack.select_files(runs)
    out = tmp_path / "out" / "demo_pack.zip"
    total = make_demo_pack.write_pack(files, runs, out)
    assert total == sum(p.stat().st_size for p in files)
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        assert "runs/dqn_B8_W10_gamma099/0/model.pt" in names
        assert all(n.startswith("runs/") for n in names)
        assert zf.read("runs/dqn_A_W10/0/config.yaml") == b"dqn_A_W10/0/config.yaml"
    # unpacking at a clone root recreates runs/<path>
    clone = tmp_path / "clone"
    with zipfile.ZipFile(out) as zf:
        zf.extractall(clone)
    assert (clone / "runs/tabular_q_W10/0/q_table.npz").is_file()


def test_make_demo_pack_main(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runs = _fake_runs(tmp_path)
    out = tmp_path / "pack.zip"
    assert make_demo_pack.main(["--runs-dir", str(runs), "--out", str(out)]) == 0
    assert out.is_file()
    assert "optional, skipped" in capsys.readouterr().out
    (runs / "dqn_B8_W10_gamma099/0/model.pt").unlink()
    assert make_demo_pack.main(["--runs-dir", str(runs), "--out", str(tmp_path / "p2.zip")]) == 1
    assert make_demo_pack.main(["--runs-dir", str(tmp_path / "nowhere"), "--out", str(tmp_path / "p3.zip")]) == 1


def test_demo_check_items_are_headless_and_ordered(tmp_path: Path) -> None:
    items = demo_check.build_items(Path("runs"), tmp_path, train=True, train_seed=7)
    names = [i.name for i in items]
    assert names[0] == "files_report"
    assert names[-2:] == ["train_smoke_8x8", "gui_play_smoke_run"]
    for item in items:
        if item.argv and "snake_rl.gui.app" in item.argv:
            assert "--headless" in item.argv and "--screenshot" in item.argv and "--steps" in item.argv
    train = next(i for i in items if i.name == "train_smoke_8x8")
    assert "--runs-dir" in train.argv and str(tmp_path / "runs") in train.argv and "7" in train.argv
    play = items[-1]
    assert str(tmp_path / "runs" / "smoke_dqn_W8" / "7") in play.argv
    assert "train_smoke_8x8" not in [i.name for i in demo_check.build_items(Path("runs"), tmp_path, train=False)]


def test_demo_check_run_item_and_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    present = tmp_path / "a.png"
    present.write_bytes(b"x")
    ok = demo_check.run_item(demo_check.Item("files", files=[present]))
    bad = demo_check.run_item(demo_check.Item("files", files=[present, tmp_path / "missing.png"]))
    assert ok.ok and not bad.ok and "missing.png" in bad.detail

    def runner(argv: list[str]) -> tuple[int, str]:
        return (0, "fine") if argv[0] == "good" else (2, "line1\nTraceback\nboom")

    assert demo_check.run_item(demo_check.Item("g", ["good"]), runner).ok
    failed = demo_check.run_item(demo_check.Item("b", ["bad"]), runner)
    assert not failed.ok and "exit 2" in failed.detail and "boom" in failed.detail
    code = demo_check.report([ok, failed])
    out = capsys.readouterr().out
    assert code == 1 and "PASS  files" in out and "FAIL  b" in out and "1 / 2 passed" in out
    assert demo_check.report([ok]) == 0

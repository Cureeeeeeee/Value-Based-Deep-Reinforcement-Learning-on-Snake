"""``scripts/report_extra_figures.py``: the composite behaviour figures of the
final report are assembled from saved behaviour analyses (heat-map arrays)
and run directories (death causes), never drawn from numbers typed in.

Synthetic inputs in ``tmp_path``: one ``heatmap_mean.npy`` or ``heatmap.npy``
per behaviour directory the script expects, and three cause experiments
written with the builders of ``tests.test_m8_death_causes``.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.test_m8_death_causes import NOWIN, THREE, make_exp

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "report_extra_figures.py"


def load_script():
    spec = importlib.util.spec_from_file_location("report_extra_figures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def heat(W: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h = rng.random((W, W))
    return h / h.sum()


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Behaviour directories for every panel of the script, run directories for
    every cause experiment, and an empty output directory."""
    module = load_script()
    behaviour = tmp_path / "behaviour"
    for _, _, panels in module.PANELS:
        W = 10 if "W10" in panels[0][1] else 8
        for k, (_, name) in enumerate(panels):
            d = behaviour / name
            d.mkdir(parents=True, exist_ok=True)
            # experiments save the seed mean, single players the plain heat-map
            file = "heatmap_mean.npy" if name.startswith("dqn_") else "heatmap.npy"
            np.save(d / file, heat(W, k))
    runs = tmp_path / "runs"
    for _, _, panels in module.CAUSE_FIGURES:
        for k, (_, name) in enumerate(panels):
            make_exp(runs, name, THREE if k % 2 == 0 else NOWIN)
    (runs / "baselines").mkdir(parents=True, exist_ok=True)
    for _, panels in module.SCORE_DISTRIBUTIONS:
        for _, groups in panels:
            for k, (_, source) in enumerate(groups):
                if source.endswith(".json"):
                    episodes = [{"seed": 10000 + i, "score": (i * 7 + k) % 23, "steps": 10 + i, "cause": "wall"} for i in range(100)]
                    (runs / "baselines" / source).write_text(json.dumps({"episodes": episodes}), encoding="utf-8")
                elif not (runs / source).exists():
                    make_exp(runs, source, THREE)
    out = tmp_path / "figures"
    return behaviour, runs, out


def test_episode_scores_pools_the_seeds_of_an_experiment_and_reads_a_baseline_file(inputs):
    _, runs, _ = inputs
    module = load_script()
    exp = next(s for _, panels in module.SCORE_DISTRIBUTIONS for _, g in panels for _, s in g if not s.endswith(".json"))
    pooled = module.episode_scores(runs, exp)
    per_seed = [json.loads(f.read_text(encoding="utf-8"))["episodes"] for f in sorted((runs / exp).glob("*/final_eval.json"))]
    assert len(pooled) == sum(len(e) for e in per_seed) and len(per_seed) == len(THREE)
    assert pooled[: len(per_seed[0])] == [int(e["score"]) for e in per_seed[0]]
    baseline = next(s for _, panels in module.SCORE_DISTRIBUTIONS for _, g in panels for _, s in g if s.endswith(".json"))
    assert len(module.episode_scores(runs, baseline)) == 100
    with pytest.raises(FileNotFoundError):
        module.episode_scores(runs, "no_such_experiment")


def test_score_distribution_figure_has_one_box_per_player_and_panel():
    module = load_script()
    fig = module.score_distribution_figure([("left", [("a", [1, 2, 3]), ("b", [4, 5, 6, 7])]), ("right", [("c", [0, 0, 1])])])
    axes = fig.get_axes()
    assert len(axes) == 2
    assert [t.get_text() for t in axes[0].get_xticklabels()] == ["a", "b"]
    assert [t.get_text() for t in axes[1].get_xticklabels()] == ["c"]
    # a box plot draws one median line per group
    assert sum(1 for line in axes[0].get_lines() if line.get_color() == "#d62728") == 2


def test_load_heat_prefers_the_seed_mean_and_reports_a_missing_directory(tmp_path: Path):
    module = load_script()
    d = tmp_path / "x"
    d.mkdir()
    np.save(d / "heatmap.npy", np.full((4, 4), 1 / 16))
    assert module.load_heat(d).sum() == pytest.approx(1.0)
    np.save(d / "heatmap_mean.npy", np.eye(4) / 4)
    assert np.array_equal(module.load_heat(d), np.eye(4) / 4)
    with pytest.raises(FileNotFoundError):
        module.load_heat(tmp_path / "missing")


def test_main_writes_every_figure_from_the_inputs(inputs):
    behaviour, runs, out = inputs
    module = load_script()
    assert module.main(["--behaviour-dir", str(behaviour), "--runs-dir", str(runs), "--out-dir", str(out)]) == 0
    expected = ([name for name, _, _ in module.PANELS] + [name for name, _, _ in module.CAUSE_FIGURES]
                + [name for name, _ in module.SCORE_DISTRIBUTIONS])
    assert sorted(p.name for p in out.iterdir()) == sorted(expected)
    for name in expected:
        data = (out / name).read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n", name
        assert len(data) > 5_000, name


def test_the_script_runs_from_any_directory(inputs, tmp_path: Path):
    behaviour, runs, out = inputs
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--behaviour-dir", str(behaviour), "--runs-dir", str(runs), "--out-dir", str(out)],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert done.returncode == 0, done.stderr
    module = load_script()
    assert done.stdout.count("written") == len(module.PANELS) + len(module.CAUSE_FIGURES) + len(module.SCORE_DISTRIBUTIONS)


def test_a_missing_panel_fails_loudly(inputs):
    behaviour, runs, out = inputs
    module = load_script()
    victim = behaviour / module.PANELS[0][2][0][1]
    for f in victim.iterdir():
        f.unlink()
    with pytest.raises(FileNotFoundError):
        module.main(["--behaviour-dir", str(behaviour), "--runs-dir", str(runs), "--out-dir", str(out)])

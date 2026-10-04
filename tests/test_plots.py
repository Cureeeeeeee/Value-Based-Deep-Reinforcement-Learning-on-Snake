"""Learning-curve figure (``snake_rl/analysis/plots.py``).

Synthetic run directories from ``tests.test_aggregate`` in ``tmp_path``. The
five-seed fixture has the curve (env_steps 0, 100, 200): mean 0, 3, 6 and
Student-t half-width 0, 1.9629, 3.6195. The baselines are ``bfs_W8`` (scores
20 30 40: mean 30) and ``random_W8`` (0 1 0: mean 1/3).

The figure is inspected through ``learning_curve_figure`` (the object that
``plot_learning_curves`` saves): artists are found by their gid.
"""

from __future__ import annotations

import math
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_hex, to_rgba
from matplotlib.image import imread

from snake_rl.analysis import plots
from snake_rl.analysis.aggregate import learning_curve
from snake_rl.analysis.plots import (
    BAND_ALPHA,
    DEFAULT_YLABEL,
    SERIES_COLORS,
    learning_curve_figure,
    main,
    plot_learning_curves,
)
from snake_rl.utils import REPO_ROOT
from tests.test_aggregate import FIVE_SEEDS, T4, make_five_seed_exp, write_baselines, write_run

PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK, INK_SECONDARY, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
WIDTH_PX, HEIGHT_PX = 1050, 630  # 7 x 4.2 inches at 150 dpi


# ------------------------------------------------------------------- builders
@pytest.fixture
def five(tmp_path: Path) -> Path:
    return make_five_seed_exp(tmp_path / "runs")


@pytest.fixture
def two(tmp_path: Path) -> Path:
    exp = tmp_path / "runs" / "two_seeds"
    write_run(exp, 0, [(1, 5, "wall"), (3, 5, "self")], [(0, 0.0), (100, 2.0), (200, 3.0)])
    write_run(exp, 1, [(5, 9, "self"), (7, 9, "win")], [(0, 1.0), (100, 6.0), (200, 4.0)])
    return exp


@pytest.fixture
def one(tmp_path: Path) -> Path:
    exp = tmp_path / "runs" / "one_seed"
    write_run(exp, 0, FIVE_SEEDS[1], [(0, 0.5), (100, 2.5), (200, 5.0)])
    return exp


def by_gid(fig, gid: str):
    found = [a for a in fig.axes[0].get_children() if a.get_gid() == gid]
    assert len(found) <= 1, gid
    return found[0] if found else None


def gids(fig, prefix: str) -> list[str]:
    return sorted(a.get_gid() for a in fig.axes[0].get_children() if str(a.get_gid()).startswith(prefix + ":"))


def band_bounds(band) -> dict[float, tuple[float, float]]:
    """x -> (lowest y, highest y) over the vertices of the band polygon."""
    (path,) = band.get_paths()
    bounds: dict[float, tuple[float, float]] = {}
    for x, y in path.vertices:
        lo, hi = bounds.get(float(x), (math.inf, -math.inf))
        bounds[float(x)] = (min(lo, float(y)), max(hi, float(y)))
    return bounds


def annotations(fig) -> dict[str, object]:
    return {t.get_text(): t for t in fig.axes[0].texts}


def legend_texts(fig) -> list[str]:
    return [t.get_text() for t in fig.axes[0].get_legend().get_texts()]


def png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR"
    return struct.unpack(">II", data[16:24])


# ------------------------------------------------------------------- the file
def test_png_is_written_where_asked_and_has_the_expected_size(five, tmp_path):
    bfs, rnd = write_baselines(tmp_path)
    out = tmp_path / "figures" / "nested" / "curve.png"
    assert not out.parent.exists()
    returned = plot_learning_curves([("five seeds", five)], [("BFS", bfs), ("random", rnd)], out=out)
    assert returned == out and isinstance(returned, Path) and out.is_file()
    assert png_size(out) == (WIDTH_PX, HEIGHT_PX)

    image = imread(out)
    assert image.shape == (HEIGHT_PX, WIDTH_PX, 4)
    rgb = np.round(image[..., :3] * 255).astype(int)
    # a real drawing: mostly the chart surface, with pixels of the first series colour (blue) and of dark text
    surface = np.array([0xFC, 0xFC, 0xFB])
    assert 0.80 < (rgb == surface).all(axis=-1).mean() < 0.99
    assert (np.abs(rgb - np.array([0x2A, 0x78, 0xD6])).max(axis=-1) <= 2).sum() > 100
    assert (rgb.max(axis=-1) < 0x40).sum() > 100
    assert len(np.unique(rgb.reshape(-1, 3), axis=0)) > 50


def test_out_may_be_a_string_and_an_existing_figure_is_overwritten(five, one, tmp_path):
    out = tmp_path / "curve.png"
    assert plot_learning_curves([("a", str(five))], out=str(out)) == out
    first = out.read_bytes()
    assert plot_learning_curves([("a", five)], out=out).read_bytes() == first  # same logs, same file
    plot_learning_curves([("a", one)], out=out)
    assert out.read_bytes() != first and png_size(out) == (WIDTH_PX, HEIGHT_PX)


def test_out_must_be_a_png(five, tmp_path):
    with pytest.raises(ValueError, match=r"\.png"):
        plot_learning_curves([("a", five)], out=tmp_path / "curve.pdf")
    assert list(tmp_path.glob("curve.*")) == []


# ---------------------------------------------------------------- the artists
def test_one_axes_with_one_line_per_experiment_equal_to_the_learning_curve(five, two):
    fig = learning_curve_figure([("five", five), ("two", two)])
    assert len(fig.axes) == 1
    assert isinstance(fig.canvas, FigureCanvasAgg)
    assert gids(fig, "series") == ["series:0", "series:1"]
    for i, exp in enumerate((five, two)):
        curve = learning_curve(exp)
        line = by_gid(fig, f"series:{i}")
        assert np.array_equal(line.get_xdata(), curve["env_steps"].to_numpy())
        assert np.array_equal(line.get_ydata(), curve["mean"].to_numpy())
        assert line.get_linestyle() == "-" and line.get_linewidth() == 1.5 and line.get_marker() == "None"
    # by hand: means over seeds 0 3 6, and (0 + 1) / 2, (2 + 6) / 2, (3 + 4) / 2
    assert by_gid(fig, "series:0").get_xdata().tolist() == [0, 100, 200]
    assert by_gid(fig, "series:0").get_ydata().tolist() == pytest.approx([0.0, 3.0, 6.0], rel=1e-12)
    assert by_gid(fig, "series:1").get_ydata().tolist() == pytest.approx([0.5, 4.0, 3.5], rel=1e-12)
    # the lines of the axes are exactly the two curves: no markers, no extra series
    assert len(fig.axes[0].lines) == 2


def test_one_band_per_experiment_with_bounds_mean_plus_minus_ci95(five, two):
    fig = learning_curve_figure([("five", five), ("two", two)])
    assert gids(fig, "band") == ["band:0", "band:1"]
    assert len(fig.axes[0].collections) == 2
    for i, exp in enumerate((five, two)):
        curve = learning_curve(exp)
        band = by_gid(fig, f"band:{i}")
        bounds = band_bounds(band)
        assert sorted(bounds) == curve["env_steps"].tolist()
        for x, mean, ci in zip(curve["env_steps"], curve["mean"], curve["ci95"]):
            assert bounds[float(x)] == pytest.approx((mean - ci, mean + ci), rel=1e-12, abs=1e-12)
        # the series colour at low alpha, no outline
        assert tuple(band.get_facecolor()[0]) == pytest.approx(to_rgba(PALETTE[i], BAND_ALPHA))
        assert 0.1 <= BAND_ALPHA <= 0.25
        assert band.get_linewidth()[0] == 0.0
    # by hand, five seeds at env_steps 200: 6 +- 2.776 * sqrt(8.5 / 5)
    half = T4 * math.sqrt(8.5 / 5)
    assert band_bounds(by_gid(fig, "band:0"))[200.0] == pytest.approx((6.0 - half, 6.0 + half), rel=1e-12)
    assert band_bounds(by_gid(fig, "band:0"))[100.0] == pytest.approx((3.0 - 1.9629, 3.0 + 1.9629), abs=5e-5)
    # two seeds at env_steps 100 (2 and 6): 4 +- 12.706 * |2 - 6| / 2
    assert band_bounds(by_gid(fig, "band:1"))[100.0] == pytest.approx((4.0 - 25.412, 4.0 + 25.412), rel=1e-12)


def test_single_seed_draws_a_line_and_no_band(one, five):
    fig = learning_curve_figure([("one seed", one)])
    assert by_gid(fig, "series:0").get_ydata().tolist() == [0.5, 2.5, 5.0]
    assert gids(fig, "band") == [] and len(fig.axes[0].collections) == 0
    assert legend_texts(fig) == ["one seed"]  # no interval, so no key for it
    # next to an experiment with seeds, only that one has a band
    fig = learning_curve_figure([("one seed", one), ("five", five)])
    assert gids(fig, "band") == ["band:1"] and gids(fig, "series") == ["series:0", "series:1"]


def test_baselines_are_horizontal_dashed_lines_at_the_mean_score(five, tmp_path):
    bfs, rnd = write_baselines(tmp_path)
    fig = learning_curve_figure([("five", five)], [("BFS", bfs), ("random", rnd)])
    ax = fig.axes[0]
    assert gids(fig, "baseline") == ["baseline:0", "baseline:1"]
    for j, level in enumerate((30.0, 1 / 3)):
        line = by_gid(fig, f"baseline:{j}")
        assert line.get_ydata() == pytest.approx([level, level], rel=1e-12)  # horizontal, at the mean score
        assert list(line.get_xdata()) == [0, 1] and line.get_transform() == ax.get_yaxis_transform()  # full width
        assert line.get_linestyle() == "--" and line.is_dashed()
        assert to_hex(line.get_color()) == INK_SECONDARY  # a reference line, not a series colour
        assert line.get_linewidth() < by_gid(fig, "series:0").get_linewidth()
    # the axis makes room for the reference level above the curve, and starts at zero
    assert ax.get_ylim()[0] == 0.0 and ax.get_ylim()[1] > 30.0
    assert len(fig.axes) == 1 and len(ax.lines) == 3


def test_series_colours_follow_the_fixed_slot_order(five, two, one):
    assert SERIES_COLORS == PALETTE
    fig = learning_curve_figure([("a", five), ("b", two), ("c", one)])
    assert [to_hex(by_gid(fig, f"series:{i}").get_color()) for i in range(3)] == ["#2a78d6", "#eb6834", "#1baf7a"]
    # the colour belongs to the position in the list, not to the experiment
    fig = learning_curve_figure([("c", one), ("a", five)])
    assert [to_hex(by_gid(fig, f"series:{i}").get_color()) for i in range(2)] == ["#2a78d6", "#eb6834"]
    assert by_gid(fig, "series:0").get_ydata().tolist() == [0.5, 2.5, 5.0]


def test_eight_series_use_every_slot_once_and_a_ninth_is_refused(five):
    eight = [(f"s{i}", five) for i in range(8)]
    fig = learning_curve_figure(eight)
    assert [to_hex(by_gid(fig, f"series:{i}").get_color()) for i in range(8)] == list(PALETTE)
    with pytest.raises(ValueError, match="never reused"):
        learning_curve_figure(eight + [("s8", five)])


# ----------------------------------------------------- labels, legend, chrome
def test_axis_labels_title_and_surface(five):
    fig = learning_curve_figure([("five", five)])
    ax = fig.axes[0]
    assert ax.get_xlabel() == "environment steps"
    assert ax.get_ylabel() == DEFAULT_YLABEL == "mean score (100 greedy episodes)"
    assert fig.get_suptitle() == "" and ax.get_title() == ""  # no title unless given
    assert to_hex(fig.get_facecolor()) == SURFACE and to_hex(ax.get_facecolor()) == SURFACE
    assert fig.get_size_inches().tolist() == [7.0, 4.2] and fig.dpi == 150
    assert not ax.spines["top"].get_visible() and not ax.spines["right"].get_visible()
    assert ax.get_xlim()[0] <= 0 and ax.get_xlim()[1] >= 200

    fig = learning_curve_figure([("five", five)], title="Tabular Q on 8x8", ylabel="mean score (20 episodes)")
    assert fig.get_suptitle() == "Tabular Q on 8x8"
    assert fig.axes[0].get_ylabel() == "mean score (20 episodes)"


def test_legend_explains_series_band_and_baseline(five, two, tmp_path):
    bfs, rnd = write_baselines(tmp_path)
    fig = learning_curve_figure([("five", five), ("two", two)], [("BFS", bfs), ("random", rnd)])
    legend = fig.axes[0].get_legend()
    assert legend is not None
    # five and two seeds: no single count to print
    assert legend_texts(fig) == ["five", "two", "95% CI over seeds", "non-learning baseline"]
    handles = legend.legend_handles
    assert [to_hex(h.get_color()) for h in handles[:2]] == ["#2a78d6", "#eb6834"]
    assert handles[3].get_linestyle() == "--" and to_hex(handles[3].get_color()) == INK_SECONDARY
    assert all(to_hex(t.get_color()) == INK for t in legend.get_texts())  # text never wears the data colour

    assert legend_texts(learning_curve_figure([("five", five)])) == ["five", "95% CI over 5 seeds"]
    assert legend_texts(learning_curve_figure([("five", five)], [("BFS", bfs)])) == [
        "five", "95% CI over 5 seeds", "non-learning baseline",
    ]


def test_up_to_four_curves_are_labelled_at_their_right_end(five, two, one, tmp_path):
    bfs, rnd = write_baselines(tmp_path)
    fig = learning_curve_figure([("five", five), ("two", two), ("one", one)], [("BFS", bfs), ("random", rnd)])
    notes = annotations(fig)
    assert sorted(notes) == ["BFS", "five", "one", "random", "two"]
    assert all(to_hex(t.get_color()) == INK for t in notes.values())  # ink, not the series colour
    # each label is anchored at the end of its own line
    assert notes["five"].xy == (200.0, pytest.approx(6.0)) and notes["two"].xy == (200.0, pytest.approx(3.5))
    assert notes["one"].xy == (200.0, 5.0)
    assert notes["BFS"].xy == (1.0, pytest.approx(30.0)) and notes["random"].xy == (1.0, pytest.approx(1 / 3))

    # five curves: the legend identifies them; baselines keep their label (they share one colour)
    many = [(f"s{i}", five) for i in range(5)]
    fig = learning_curve_figure(many, [("BFS", bfs)])
    assert sorted(annotations(fig)) == ["BFS"]
    assert legend_texts(fig)[:5] == ["s0", "s1", "s2", "s3", "s4"]
    assert sorted(annotations(learning_curve_figure(many[:4]))) == ["s0", "s1", "s2", "s3"]


def test_labels_that_would_overlap_are_moved_apart_with_a_leader(five, tmp_path):
    bfs, _ = write_baselines(tmp_path)
    fig = learning_curve_figure([("first", five), ("second", five)], [("BFS", bfs)])  # two identical curves
    notes = annotations(fig)
    fig.draw_without_rendering()
    boxes = {k: t.get_window_extent() for k, t in notes.items()}
    assert not boxes["first"].overlaps(boxes["second"])
    assert notes["first"].xy == notes["second"].xy == (200.0, pytest.approx(6.0))
    assert notes["first"].arrow_patch is not None and notes["second"].arrow_patch is not None  # leaders
    assert notes["BFS"].arrow_patch is None  # not moved: no leader
    # nothing is clipped: every label lies inside the figure
    for box in boxes.values():
        assert 0 <= box.x0 and box.x1 <= WIDTH_PX and 0 <= box.y0 and box.y1 <= HEIGHT_PX


def test_a_curve_that_stops_early_is_labelled_at_its_own_end(five, tmp_path):
    long = tmp_path / "runs" / "long"
    write_run(long, 0, FIVE_SEEDS[1], [(0, 0.0), (100, 2.0), (200, 3.0), (300, 3.0), (400, 4.0)])
    write_run(long, 1, FIVE_SEEDS[2], [(0, 0.0), (100, 4.0), (200, 5.0), (300, 5.0), (400, 6.0)])
    fig = learning_curve_figure([("short", five), ("long", long)])
    notes = annotations(fig)
    fig.draw_without_rendering()
    axes_box = fig.axes[0].get_window_extent()
    assert notes["short"].xy == (200.0, pytest.approx(6.0)) and notes["short"].arrow_patch is None
    assert notes["short"].get_window_extent().x1 < axes_box.x1  # inside the axes, next to the line end
    assert notes["long"].get_window_extent().x0 > axes_box.x1  # in the right margin
    assert fig.axes[0].get_xlim()[1] >= 400


@pytest.mark.parametrize(
    "anchors, gap, expected",
    [
        ([0.0, 10.0], 1.0, [0.0, 10.0]),  # far apart: untouched
        ([1.0, 1.0], 0.5, [0.75, 1.25]),  # a tie is split around the anchor, the earlier label below
        ([5.0, 5.2, 9.0], 1.0, [4.6, 5.6, 9.0]),  # 5 and 5.2 pooled around 5.1; 9 untouched
        ([9.0, 5.2, 5.0], 1.0, [9.0, 5.6, 4.6]),  # the result is in the order of the input
        ([0.0, 0.1], 1.0, [0.0, 1.0]),  # pushed up from the lower limit 0
        ([19.9, 20.0], 1.0, [19.0, 20.0]),  # pushed down from the upper limit 20
        ([], 1.0, []),
    ],
)
def test_spread_by_hand(anchors, gap, expected):
    assert plots._spread(anchors, gap, 0.0, 20.0) == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    "value, text",
    [(0, "0"), (500, "500"), (25600, "25.6k"), (200000, "200k"), (1000000, "1M"), (1024000, "1.024M"),
     (1200000, "1.2M")],
)
def test_step_tick_labels(value, text):
    assert plots._format_steps(value) == text


# ----------------------------------------------------------------- bad inputs
def test_bad_inputs_are_refused(five, tmp_path):
    bfs, _ = write_baselines(tmp_path)
    with pytest.raises(ValueError, match="at least one experiment"):
        learning_curve_figure([], [("BFS", bfs)])
    with pytest.raises(ValueError, match="unique"):
        learning_curve_figure([("same", five), ("same", five)])
    with pytest.raises(ValueError, match="unique"):
        learning_curve_figure([("same", five)], [("same", bfs)])
    with pytest.raises(ValueError, match="not a directory"):
        plot_learning_curves([("a", tmp_path / "no_such_experiment")], out=tmp_path / "x.png")
    assert not (tmp_path / "x.png").exists()


# ------------------------------------------------------------------------ CLI
def test_cli_parses_dir_label_and_file_label(five, two, tmp_path, capsys, monkeypatch):
    bfs, rnd = write_baselines(tmp_path)
    out = tmp_path / "figures" / "cli.png"
    argv = ["--exp", f"{five}=tabular Q, 5 seeds", str(two), "--baseline", f"{bfs}=BFS greedy", str(rnd),
            "--out", str(out), "--title", "Part C"]
    calls = []

    def capture(experiments, baselines, **kwargs):
        calls.append((experiments, baselines, kwargs))
        return Path(kwargs["out"])

    with monkeypatch.context() as patch:
        patch.setattr(plots, "plot_learning_curves", capture)
        assert main(argv) == 0
    assert calls == [
        (
            [("tabular Q, 5 seeds", str(five)), ("two_seeds", str(two))],  # default label: the directory name
            [("BFS greedy", str(bfs)), ("random_W8", str(rnd))],  # default label: the file stem
            {"out": str(out), "title": "Part C", "ylabel": DEFAULT_YLABEL},
        )
    ]
    capsys.readouterr()

    assert main(argv) == 0  # the real thing
    assert capsys.readouterr().out.strip() == str(out)
    assert png_size(out) == (WIDTH_PX, HEIGHT_PX)


def test_cli_label_split_survives_equal_signs_in_paths_and_labels(tmp_path):
    sweep = make_five_seed_exp(tmp_path, name="tabular_q_W8__alpha=0.1__gamma=0.9")
    plain = make_five_seed_exp(tmp_path, name="plain")
    split = plots._split_label
    assert split(str(sweep), Path.is_dir) == (str(sweep), None)  # exists as a whole: no label
    assert split(f"{sweep}=alpha 0.1", Path.is_dir) == (str(sweep), "alpha 0.1")
    assert split(f"{sweep}=gamma=0.9", Path.is_dir) == (str(sweep), "gamma=0.9")
    assert split(f"{plain}=gamma=0.9", Path.is_dir) == (str(plain), "gamma=0.9")
    assert split(str(plain), Path.is_dir) == (str(plain), None)
    assert split("no/such/dir=label=x", Path.is_dir) == ("no/such/dir", "label=x")
    assert split(f"{plain}=", Path.is_dir) == (str(plain), "")


def test_cli_usage_errors(five, tmp_path, capsys):
    out = tmp_path / "x.png"
    for argv in (["--out", str(out)], ["--exp", str(five)], ["--exp", f"{five}=", "--out", str(out)]):
        with pytest.raises(SystemExit) as exc:
            main(argv)
        assert exc.value.code == 2
    assert "empty label" in capsys.readouterr().err
    assert not out.exists()


def test_module_needs_no_display_and_leaves_the_backend_alone(five, tmp_path):
    """A fresh process whose configured backend cannot be loaded: importing the
    module and drawing must work, because pyplot and the configured backend
    are never touched (the figure lives on its own Agg canvas)."""
    out = tmp_path / "headless.png"
    code = (
        "import sys\n"
        "import snake_rl.analysis.plots as plots\n"
        "assert 'matplotlib.pyplot' not in sys.modules\n"
        "plots.plot_learning_curves([('five', sys.argv[1])], out=sys.argv[2])\n"
        "assert 'matplotlib.pyplot' not in sys.modules\n"
        "import matplotlib\n"
        "assert matplotlib.rcParams['backend'] == 'module://no_such_backend', matplotlib.rcParams['backend']\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "DISPLAY"}
    env["MPLBACKEND"] = "module://no_such_backend"
    proc = subprocess.run(
        [sys.executable, "-c", code, str(five), str(out)], cwd=REPO_ROOT, env=env, capture_output=True, text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert png_size(out) == (WIDTH_PX, HEIGHT_PX)


def test_module_entry_point(five, tmp_path):
    bfs, _ = write_baselines(tmp_path)
    out = tmp_path / "entry" / "curve.png"
    proc = subprocess.run(
        [sys.executable, "-m", "snake_rl.analysis.plots", "--exp", f"{five}=five", "--baseline", f"{bfs}=BFS",
         "--out", str(out)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(out) and png_size(out) == (WIDTH_PX, HEIGHT_PX)

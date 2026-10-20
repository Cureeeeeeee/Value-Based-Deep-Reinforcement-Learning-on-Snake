"""The overestimation figure (M7, spec D9): ``value_curve_figure``,
``plot_value_curves`` and the ``--values`` / ``--start`` flags of the plots
CLI of ``snake_rl/analysis/plots.py``.

Per experiment ``i`` in ``SERIES_COLORS[i]``: a solid line of the max-Q
estimate (gid ``value:<i>``) with its Student-t band (``value_band:<i>``)
and a dashed line of the realised return (``return:<i>``) with its band
(``return_band:<i>``); no bands for one seed. The y axis starts at
``min(0, lowest band or line)`` (a death costs -1, so a realised return can
be negative) and ends at ``1.06 * highest``. Synthetic run directories from
``tests.test_m7_value_curve`` (three seeds with hand-picked numbers); the
figure is inspected through its artists' gids as in ``tests/test_plots.py``.
"""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_hex, to_rgba
from matplotlib.image import imread

from snake_rl.analysis import plots
from snake_rl.analysis.aggregate import value_curve
from snake_rl.analysis.plots import (
    BAND_ALPHA,
    DEFAULT_VALUE_YLABEL,
    DEFAULT_YLABEL,
    GID_RETURN,
    GID_RETURN_BAND,
    GID_VALUE,
    GID_VALUE_BAND,
    LINE_WIDTH,
    MAX_DIRECT_LABELS,
    SERIES_COLORS,
    main,
    plot_value_curves,
    value_curve_figure,
)
from snake_rl.utils import REPO_ROOT
from tests.test_aggregate import FIVE_SEEDS, write_baselines, write_run
from tests.test_m7_value_curve import T2, VISITED, expected_frame, make_three_seed_exp, points_of, write_dqn_run
from tests.test_plots import HEIGHT_PX, INK, INK_SECONDARY, PALETTE, SURFACE, WIDTH_PX, annotations, band_bounds, by_gid, gids, legend_texts, png_size

ESTIMATE_LABEL, RETURN_LABEL = "max-Q estimate", "realised return"


# ------------------------------------------------------------------- builders
@pytest.fixture
def three(tmp_path: Path) -> Path:
    return make_three_seed_exp(tmp_path / "runs")


@pytest.fixture
def shifted(tmp_path: Path) -> Path:
    """Three seeds with other numbers (values + 2, returns - 0.5)."""
    exp = tmp_path / "runs" / "shifted"
    for seed in VISITED:
        write_dqn_run(exp, seed, points_of(seed, visited=[(vv + 2.0, rv - 0.5) for vv, rv in VISITED[seed]]))
    return exp


@pytest.fixture
def two(tmp_path: Path) -> Path:
    exp = tmp_path / "runs" / "two_seeds"
    for seed in (0, 1):
        write_dqn_run(exp, seed, points_of(seed))
    return exp


@pytest.fixture
def one(tmp_path: Path) -> Path:
    exp = tmp_path / "runs" / "one_seed"
    write_dqn_run(exp, 0, points_of(0))
    return exp


@pytest.fixture
def negative_one(tmp_path: Path) -> Path:
    """One seed whose realised returns are negative (deaths): the axis must reach below zero."""
    exp = tmp_path / "runs" / "negative"
    write_dqn_run(exp, 0, points_of(0, visited=[(0.5, -0.25), (1.0, -1.5), (2.0, -0.75)]))
    return exp


def frame_extremes(frames) -> tuple[float, float]:
    """Lowest and highest of every line and band of the given value-curve frames (spec D9, the y axis)."""
    lows, highs = [], []
    for curve in frames:
        for kind in ("value", "return"):
            mean, ci = curve[f"{kind}_mean"].to_numpy(), curve[f"{kind}_ci95"].to_numpy()
            lows.append(mean.min())
            highs.append(mean.max())
            if np.isfinite(ci).all():
                lows.append((mean - ci).min())
                highs.append((mean + ci).max())
    return float(min(lows)), float(max(highs))


# ------------------------------------------------------------------- constants and signatures
def test_constants_and_signatures():
    assert DEFAULT_VALUE_YLABEL == "discounted return (greedy evaluation)"
    assert (GID_VALUE, GID_VALUE_BAND, GID_RETURN, GID_RETURN_BAND) == ("value", "value_band", "return", "return_band")
    params = inspect.signature(value_curve_figure).parameters
    assert list(params) == ["experiments", "which", "title", "ylabel"]
    assert params["which"].default == "visited" and params["title"].default is None
    assert params["ylabel"].default == DEFAULT_VALUE_YLABEL
    params = inspect.signature(plot_value_curves).parameters
    assert list(params) == ["experiments", "out", "which", "title", "ylabel"]
    assert params["out"].default is inspect.Parameter.empty and params["which"].default == "visited"
    assert params["ylabel"].default == DEFAULT_VALUE_YLABEL


# ------------------------------------------------------------------- the artists
def test_solid_value_line_and_dashed_return_line_per_experiment_equal_to_the_frame(three, shifted):
    fig = value_curve_figure([("three", three), ("shifted", shifted)])
    assert len(fig.axes) == 1 and isinstance(fig.canvas, FigureCanvasAgg)
    ax = fig.axes[0]
    assert gids(fig, GID_VALUE) == ["value:0", "value:1"] and gids(fig, GID_RETURN) == ["return:0", "return:1"]
    for i, exp in enumerate((three, shifted)):
        curve = value_curve(exp)
        value, ret = by_gid(fig, f"value:{i}"), by_gid(fig, f"return:{i}")
        assert np.array_equal(value.get_xdata(), curve["env_steps"].to_numpy())
        assert np.array_equal(value.get_ydata(), curve["value_mean"].to_numpy())
        assert np.array_equal(ret.get_xdata(), curve["env_steps"].to_numpy())
        assert np.array_equal(ret.get_ydata(), curve["return_mean"].to_numpy())
        assert value.get_linestyle() == "-" and not value.is_dashed()
        assert ret.get_linestyle() == "--" and ret.is_dashed()
        assert value.get_linewidth() == ret.get_linewidth() == LINE_WIDTH == 1.5
        assert to_hex(value.get_color()) == to_hex(ret.get_color()) == PALETTE[i] == SERIES_COLORS[i]
        assert value.get_marker() == "None" and ret.get_marker() == "None"
    # by hand: three seeds, values 0 2 4 and returns 0 1 3
    assert by_gid(fig, "value:0").get_xdata().tolist() == [0, 100, 200]
    assert by_gid(fig, "value:0").get_ydata().tolist() == pytest.approx([0.0, 2.0, 4.0], rel=1e-12)
    assert by_gid(fig, "return:0").get_ydata().tolist() == pytest.approx([0.0, 1.0, 3.0], rel=1e-12)
    assert by_gid(fig, "value:1").get_ydata().tolist() == pytest.approx([2.0, 4.0, 6.0], rel=1e-12)
    assert by_gid(fig, "return:1").get_ydata().tolist() == pytest.approx([-0.5, 0.5, 2.5], rel=1e-12)
    assert len(ax.lines) == 4  # two lines per experiment, nothing else


def test_one_band_per_line_with_bounds_mean_plus_minus_ci95(three, two):
    fig = value_curve_figure([("three", three), ("two", two)])
    assert gids(fig, GID_VALUE_BAND) == ["value_band:0", "value_band:1"]
    assert gids(fig, GID_RETURN_BAND) == ["return_band:0", "return_band:1"]
    assert len(fig.axes[0].collections) == 4
    for i, exp in enumerate((three, two)):
        curve = value_curve(exp)
        for kind in ("value", "return"):
            band = by_gid(fig, f"{kind}_band:{i}")
            bounds = band_bounds(band)
            assert sorted(bounds) == curve["env_steps"].tolist()
            for x, mean, ci in zip(curve["env_steps"], curve[f"{kind}_mean"], curve[f"{kind}_ci95"]):
                assert bounds[float(x)] == pytest.approx((mean - ci, mean + ci), rel=1e-12, abs=1e-12), (kind, x)
            assert tuple(band.get_facecolor()[0]) == pytest.approx(to_rgba(PALETTE[i], BAND_ALPHA))
            assert band.get_linewidth()[0] == 0.0
    # by hand at env_steps 200: values 4 +- 4.303 / sqrt(3), returns 3 +- 4.303 * sqrt(7 / 3)
    assert band_bounds(by_gid(fig, "value_band:0"))[200.0] == pytest.approx((4.0 - T2 / 3**0.5, 4.0 + T2 / 3**0.5), rel=1e-12)
    assert band_bounds(by_gid(fig, "return_band:0"))[200.0] == pytest.approx(
        (3.0 - T2 * (7 / 3) ** 0.5, 3.0 + T2 * (7 / 3) ** 0.5), rel=1e-12)


def test_which_start_draws_the_start_columns(three):
    fig = value_curve_figure([("three", three)], which="start")
    start = expected_frame("start")
    assert by_gid(fig, "value:0").get_ydata().tolist() == pytest.approx(start["value_mean"].tolist(), rel=1e-12)
    assert by_gid(fig, "return:0").get_ydata().tolist() == pytest.approx(start["return_mean"].tolist(), rel=1e-12)
    assert by_gid(fig, "value:0").get_ydata().tolist() != pytest.approx(expected_frame()["value_mean"].tolist())


def test_single_seed_draws_two_lines_and_no_band(one, three):
    fig = value_curve_figure([("one seed", one)])
    assert gids(fig, GID_VALUE) == ["value:0"] and gids(fig, GID_RETURN) == ["return:0"]
    assert gids(fig, GID_VALUE_BAND) == [] and gids(fig, GID_RETURN_BAND) == [] and len(fig.axes[0].collections) == 0
    assert by_gid(fig, "value:0").get_ydata().tolist() == [0.0, 1.0, 3.0]
    assert by_gid(fig, "return:0").get_ydata().tolist() == [0.0, 0.5, 1.0]
    assert legend_texts(fig) == ["one seed", ESTIMATE_LABEL, RETURN_LABEL]  # no interval, so no key for it
    # next to an experiment with seeds, only that one has bands
    fig = value_curve_figure([("one seed", one), ("three", three)])
    assert gids(fig, GID_VALUE_BAND) == ["value_band:1"] and gids(fig, GID_RETURN_BAND) == ["return_band:1"]
    assert gids(fig, GID_VALUE) == ["value:0", "value:1"]


def test_series_colours_follow_the_fixed_slot_order_and_a_ninth_is_refused(three, two, one):
    fig = value_curve_figure([("a", three), ("b", two), ("c", one)])
    for kind in ("value", "return"):
        assert [to_hex(by_gid(fig, f"{kind}:{i}").get_color()) for i in range(3)] == ["#2a78d6", "#eb6834", "#1baf7a"]
    fig = value_curve_figure([("c", one), ("a", three)])
    assert to_hex(by_gid(fig, "value:0").get_color()) == "#2a78d6" and by_gid(fig, "value:0").get_ydata().tolist() == [0.0, 1.0, 3.0]
    eight = [(f"s{i}", three) for i in range(8)]
    fig = value_curve_figure(eight)
    assert [to_hex(by_gid(fig, f"value:{i}").get_color()) for i in range(8)] == list(PALETTE)
    with pytest.raises(ValueError, match="never reused"):
        value_curve_figure(eight + [("s8", three)])


# ----------------------------------------------------- axes, legend, labels
def test_axes_labels_limits_formatter_and_title(three, one, negative_one):
    fig = value_curve_figure([("three", three)])
    ax = fig.axes[0]
    assert ax.get_xlabel() == "environment steps" and ax.get_ylabel() == DEFAULT_VALUE_YLABEL
    assert fig.get_suptitle() == "" and ax.get_title() == ""
    assert to_hex(fig.get_facecolor()) == SURFACE and to_hex(ax.get_facecolor()) == SURFACE
    assert fig.get_size_inches().tolist() == [7.0, 4.2] and fig.dpi == 150
    assert not ax.spines["top"].get_visible() and not ax.spines["right"].get_visible()
    assert ax.get_xlim()[0] <= 0 and ax.get_xlim()[1] >= 200
    assert ax.xaxis.get_major_formatter()(200_000, 0) == "200k" and ax.xaxis.get_major_formatter()(0, 0) == "0"
    # y: from min(0, lowest band or line) to 1.06 * highest; here the return band at step 200 reaches
    # 3 - 4.303 * sqrt(7 / 3) < 0, the highest point is its upper edge.
    lowest, highest = frame_extremes([value_curve(three)])
    assert lowest == pytest.approx(3.0 - T2 * (7 / 3) ** 0.5) and lowest < 0 < highest
    assert highest == pytest.approx(3.0 + T2 * (7 / 3) ** 0.5)
    assert ax.get_ylim() == pytest.approx((lowest, 1.06 * highest), rel=1e-9)
    # One seed, nothing negative: the axis starts at zero.
    fig = value_curve_figure([("one", one)])
    lowest, highest = frame_extremes([value_curve(one)])
    assert lowest == 0.0 and highest == 3.0
    assert fig.axes[0].get_ylim() == pytest.approx((0.0, 1.06 * 3.0), rel=1e-9)
    # One seed with negative returns: the lowest line sets the lower limit.
    fig = value_curve_figure([("neg", negative_one)])
    assert fig.axes[0].get_ylim() == pytest.approx((-1.5, 1.06 * 2.0), rel=1e-9)
    # Several experiments: the extremes over all of them.
    fig = value_curve_figure([("one", one), ("three", three), ("neg", negative_one)])
    lowest, highest = frame_extremes([value_curve(d) for d in (one, three, negative_one)])
    assert fig.axes[0].get_ylim() == pytest.approx((min(0.0, lowest), 1.06 * highest), rel=1e-9)

    fig = value_curve_figure([("three", three)], title="Double DQN on 8x8", ylabel="return at gamma 0.9")
    assert fig.get_suptitle() == "Double DQN on 8x8" and fig.axes[0].get_ylabel() == "return at gamma 0.9"


def test_legend_lists_experiments_then_the_two_line_styles_then_the_band(three, two, one):
    fig = value_curve_figure([("three", three), ("two", two)])
    legend = fig.axes[0].get_legend()
    assert legend is not None
    assert legend_texts(fig) == ["three", "two", ESTIMATE_LABEL, RETURN_LABEL, "95% CI over seeds"]  # 3 and 2 seeds
    handles = legend.legend_handles
    assert [to_hex(h.get_color()) for h in handles[:2]] == ["#2a78d6", "#eb6834"]
    assert handles[0].get_linestyle() == "-" and handles[1].get_linestyle() == "-"
    assert handles[2].get_linestyle() == "-" and to_hex(handles[2].get_color()) == INK_SECONDARY
    assert handles[3].get_linestyle() == "--" and to_hex(handles[3].get_color()) == INK_SECONDARY
    assert all(to_hex(t.get_color()) == INK for t in legend.get_texts())
    assert legend_texts(value_curve_figure([("three", three)])) == ["three", ESTIMATE_LABEL, RETURN_LABEL, "95% CI over 3 seeds"]
    assert legend_texts(value_curve_figure([("one", one), ("three", three)])) == [
        "one", "three", ESTIMATE_LABEL, RETURN_LABEL, "95% CI over 3 seeds"]  # bands of the three-seed run only
    assert legend_texts(value_curve_figure([("one", one)])) == ["one", ESTIMATE_LABEL, RETURN_LABEL]


def test_up_to_four_experiments_are_labelled_at_the_end_of_their_solid_line(three, two, one, shifted):
    fig = value_curve_figure([("three", three), ("two", two), ("one", one), ("shifted", shifted)])
    notes = annotations(fig)
    assert sorted(notes) == ["one", "shifted", "three", "two"]
    assert all(to_hex(t.get_color()) == INK for t in notes.values())
    # anchored at the right end of the VALUE line (not the dashed return line)
    assert notes["three"].xy == (200.0, pytest.approx(4.0)) and notes["two"].xy == (200.0, pytest.approx(4.0))
    assert notes["one"].xy == (200.0, 3.0) and notes["shifted"].xy == (200.0, pytest.approx(6.0))
    assert MAX_DIRECT_LABELS == 4
    many = [(f"s{i}", three) for i in range(5)]
    fig = value_curve_figure(many)
    assert annotations(fig) == {}
    assert legend_texts(fig)[:5] == ["s0", "s1", "s2", "s3", "s4"]


# ----------------------------------------------------------------- bad inputs
def test_bad_inputs_are_refused(three, tmp_path):
    with pytest.raises(ValueError, match="at least one experiment"):
        value_curve_figure([])
    with pytest.raises(ValueError, match="unique"):
        value_curve_figure([("same", three), ("same", three)])
    with pytest.raises(ValueError, match="not a directory"):
        plot_value_curves([("a", tmp_path / "no_such_experiment")], out=tmp_path / "x.png")
    with pytest.raises(ValueError):
        value_curve_figure([("three", three)], which="foo")
    tabular = tmp_path / "runs" / "tabular"
    write_run(tabular, 0, FIVE_SEEDS[1], [(0, 0.0), (100, 1.0)])
    with pytest.raises(ValueError, match="no value estimates"):
        value_curve_figure([("t", tabular)])
    assert not (tmp_path / "x.png").exists()


# ------------------------------------------------------------------- the file
def test_png_is_written_where_asked_overwritten_and_only_as_png(three, one, tmp_path):
    out = tmp_path / "figures" / "nested" / "values.png"
    assert not out.parent.exists()
    returned = plot_value_curves([("three seeds", three)], out)
    assert returned == out and isinstance(returned, Path) and out.is_file()
    assert png_size(out) == (WIDTH_PX, HEIGHT_PX)
    image = imread(out)
    assert image.shape == (HEIGHT_PX, WIDTH_PX, 4)
    rgb = np.round(image[..., :3] * 255).astype(int)
    surface = np.array([0xFC, 0xFC, 0xFB])
    # Mostly the chart surface (the four translucent bands cover more of it than the learning-curve figure's one).
    assert 0.55 < (rgb == surface).all(axis=-1).mean() < 0.99
    assert (np.abs(rgb - np.array([0x2A, 0x78, 0xD6])).max(axis=-1) <= 2).sum() > 100  # the first series colour
    assert (rgb.max(axis=-1) < 0x40).sum() > 100  # dark text
    first = out.read_bytes()
    assert plot_value_curves([("three seeds", str(three))], out=str(out)) == out  # str paths, overwritten
    assert out.read_bytes() == first
    plot_value_curves([("one", one)], out=out, which="start", title="t")
    assert out.read_bytes() != first and png_size(out) == (WIDTH_PX, HEIGHT_PX)
    with pytest.raises(ValueError, match=r"\.png"):
        plot_value_curves([("three", three)], out=tmp_path / "values.pdf")
    assert not (tmp_path / "values.pdf").exists()


# ------------------------------------------------------------------------ CLI
def bound_call(func, args, kwargs) -> dict:
    bound = inspect.signature(func).bind(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


def test_cli_values_flag_selects_the_value_figure(three, one, tmp_path, monkeypatch, capsys):
    out = tmp_path / "figures" / "cli.png"
    calls = []
    real = plots.plot_value_curves

    def capture(*args, **kwargs):
        call = bound_call(real, args, kwargs)
        calls.append(call)
        return Path(call["out"])

    with monkeypatch.context() as patch:
        patch.setattr(plots, "plot_value_curves", capture)
        patch.setattr(plots, "plot_learning_curves", lambda *a, **k: pytest.fail("--values must not draw the learning curves"))
        assert main(["--values", "--exp", f"{three}=three seeds", str(one), "--out", str(out), "--title", "Part E"]) == 0
        assert main(["--values", "--start", "--exp", str(three), "--out", str(out), "--ylabel", "return at start"]) == 0
    first, second = calls
    assert [(label, str(path)) for label, path in first["experiments"]] == [("three seeds", str(three)), ("one_seed", str(one))]
    assert str(first["out"]) == str(out) and first["which"] == "visited" and first["title"] == "Part E"
    assert first["ylabel"] == DEFAULT_VALUE_YLABEL  # an unset --ylabel means the value label with --values
    assert [(label, str(path)) for label, path in second["experiments"]] == [("three_seeds", str(three))]
    assert second["which"] == "start" and second["title"] is None and second["ylabel"] == "return at start"
    capsys.readouterr()
    assert main(["--values", "--exp", f"{three}=three", "--out", str(out)]) == 0  # the real thing
    assert capsys.readouterr().out.strip() == str(out) and png_size(out) == (WIDTH_PX, HEIGHT_PX)


def test_cli_without_values_is_unchanged(three, tmp_path, monkeypatch):
    bfs, _ = write_baselines(tmp_path)
    out = tmp_path / "learning.png"
    calls = []

    def capture(experiments, baselines, **kwargs):
        calls.append((experiments, baselines, kwargs))
        return Path(kwargs["out"])

    with monkeypatch.context() as patch:
        patch.setattr(plots, "plot_learning_curves", capture)
        patch.setattr(plots, "plot_value_curves", lambda *a, **k: pytest.fail("the value figure needs --values"))
        assert main(["--exp", f"{three}=three", "--baseline", f"{bfs}=BFS", "--out", str(out)]) == 0
    assert calls == [([("three", str(three))], [("BFS", str(bfs))], {"out": str(out), "title": None, "ylabel": DEFAULT_YLABEL})]


def test_cli_usage_errors(three, tmp_path, capsys):
    bfs, _ = write_baselines(tmp_path)
    out = tmp_path / "x.png"
    for argv in (
        ["--start", "--exp", str(three), "--out", str(out)],  # --start needs --values
        ["--values", "--exp", str(three), "--baseline", str(bfs), "--out", str(out)],  # no baselines on the value figure
        ["--values", "--out", str(out)],  # nothing to plot
        ["--values", "--exp", str(three)],  # --out is required
    ):
        with pytest.raises(SystemExit) as exc:
            main(argv)
        assert exc.value.code == 2, argv
        assert capsys.readouterr().err != ""
    assert not out.exists()


def test_module_entry_point_and_headless_drawing(three, tmp_path):
    out = tmp_path / "entry" / "values.png"
    proc = subprocess.run(
        [sys.executable, "-m", "snake_rl.analysis.plots", "--values", "--start", "--exp", f"{three}=three", "--out", str(out),
         "--title", "values"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(out) and png_size(out) == (WIDTH_PX, HEIGHT_PX)
    # No display, no pyplot: the figure lives on its own Agg canvas.
    headless = tmp_path / "headless.png"
    code = (
        "import sys\n"
        "import snake_rl.analysis.plots as plots\n"
        "assert 'matplotlib.pyplot' not in sys.modules\n"
        "plots.plot_value_curves([('three', sys.argv[1])], sys.argv[2])\n"
        "assert 'matplotlib.pyplot' not in sys.modules\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "DISPLAY"}
    env["MPLBACKEND"] = "module://no_such_backend"
    proc = subprocess.run([sys.executable, "-c", code, str(three), str(headless)], cwd=REPO_ROOT, env=env,
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert png_size(headless) == (WIDTH_PX, HEIGHT_PX)

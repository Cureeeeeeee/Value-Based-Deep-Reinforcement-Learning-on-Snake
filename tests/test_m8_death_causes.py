"""Death-cause evolution over training (M8, spec sections 1 and 2):
``aggregate.death_cause_curve``, ``plots.death_cause_figure`` /
``plots.plot_death_causes`` and the ``--causes`` flag of the plots CLI.

Synthetic run directories in ``tmp_path`` written with the builders of
``tests.test_aggregate`` (``config.yaml`` with a ``run`` block,
``final_eval.json``, the exact ``metrics.csv`` header of the training loop);
only the four ``eval_frac_<cause>`` columns are rewritten here with
hand-picked fractions, every row summing to 1. Three seeds over env_steps
0, 100, 200; at step 200 the wall fractions are 0.2, 0.3, 0.1 (mean 0.2,
sd 0.1, Student t with 2 degrees of freedom 4.303). The figure is inspected
through its artists' gids as in ``tests/test_plots.py``.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_hex, to_rgba

from snake_rl.analysis import plots
from snake_rl.analysis.aggregate import (
    CAUSES,
    DEATH_CAUSE_COLUMNS,
    AggregateWarning,
    death_cause_curve,
    learning_curve,
    t_critical_95,
)
from snake_rl.analysis.plots import (
    BAND_ALPHA,
    CAUSE_COLORS,
    DEFAULT_CAUSE_YLABEL,
    DEFAULT_VALUE_YLABEL,
    DEFAULT_YLABEL,
    death_cause_figure,
    main,
    plot_death_causes,
)
from tests.test_aggregate import FIVE_SEEDS, METRICS_HEADER, write_baselines, write_run
from tests.test_plots import band_bounds, png_size

STEPS = (0, 100, 200)
T2 = 4.303  # two-sided 95 % Student t, 2 degrees of freedom
FRAC_COLUMNS = tuple(f"eval_frac_{c}" for c in CAUSES)

#: seed -> per env_steps (wall, self, starve, win); every row sums to 1, win appears at step 200.
THREE = {
    0: [(1.0, 0.0, 0.0, 0.0), (0.5, 0.3, 0.2, 0.0), (0.2, 0.5, 0.2, 0.1)],
    1: [(0.9, 0.1, 0.0, 0.0), (0.6, 0.2, 0.2, 0.0), (0.3, 0.4, 0.2, 0.1)],
    2: [(0.8, 0.0, 0.2, 0.0), (0.4, 0.4, 0.2, 0.0), (0.1, 0.6, 0.1, 0.2)],
}
#: Two seeds, never a win.
NOWIN = {
    0: [(1.0, 0.0, 0.0, 0.0), (0.7, 0.2, 0.1, 0.0), (0.5, 0.4, 0.1, 0.0)],
    1: [(0.9, 0.1, 0.0, 0.0), (0.5, 0.3, 0.2, 0.0), (0.3, 0.5, 0.2, 0.0)],
}
#: One seed, never a win.
ONE = {0: [(1.0, 0.0, 0.0, 0.0), (0.6, 0.3, 0.1, 0.0), (0.4, 0.4, 0.2, 0.0)]}


# ------------------------------------------------------------------- builders
def cause_metrics_csv(rows, drop: str | None = None) -> str:
    """A ``metrics.csv`` whose rows are ``(env_steps, eval_mean_score, (wall, self, starve, win))``;
    every other column is a decoy. ``drop`` removes one column from the file altogether."""
    columns = METRICS_HEADER.split(",")
    lines = [columns]
    for k, (env_steps, mean, fracs) in enumerate(rows):
        first = k == 0
        cells = [
            env_steps, env_steps // 64, 1.0 if first else 0.05, repr(float(mean)), 0.5, mean + 100.0, 99, 50.0,
            *[repr(float(f)) for f in fracs], 0 if first else 12, "" if first else 1.5, "" if first else 0.01,
            10 * k, 0.1 * k,
        ]
        lines.append([str(c) for c in cells])
    if drop is not None:
        index = columns.index(drop)
        lines = [line[:index] + line[index + 1 :] for line in lines]
    return "\n".join(",".join(line) for line in lines) + "\n"


def write_cause_run(exp_dir: Path, seed: int, fractions, steps=STEPS, *, drop: str | None = None,
                    dir_name: str | None = None) -> Path:
    """One finished run whose ``metrics.csv`` carries the given cause fractions per evaluation."""
    rows = [(s, float(k + seed), f) for k, (s, f) in enumerate(zip(steps, fractions))]
    run_dir = write_run(exp_dir, seed, FIVE_SEEDS[1], [(s, m) for s, m, _ in rows], dir_name=dir_name)
    (run_dir / "metrics.csv").write_text(cause_metrics_csv(rows, drop), encoding="utf-8")
    return run_dir


def make_exp(root: Path, name: str, table) -> Path:
    exp = root / name
    for seed in reversed(sorted(table)):  # creation order is not seed order
        write_cause_run(exp, seed, table[seed])
    return exp


def expected_curve(table) -> pd.DataFrame:
    """Spec section 1 recomputed from the fraction table with numpy alone."""
    seeds = sorted(table)
    n = len(seeds)
    values = np.array([table[s] for s in seeds], dtype=np.float64)  # (n_seeds, n_steps, 4)
    data: dict[str, object] = {"env_steps": np.array(STEPS, dtype=np.int64), "n_seeds": n}
    for j, cause in enumerate(CAUSES):
        v = values[:, :, j]
        data[f"{cause}_mean"] = v.mean(axis=0)
        if n > 1:
            data[f"{cause}_ci95"] = t_critical_95(n - 1) * v.std(axis=0, ddof=1) / math.sqrt(n)
        else:
            data[f"{cause}_ci95"] = np.full(len(STEPS), np.nan)
    return pd.DataFrame(data, columns=list(DEATH_CAUSE_COLUMNS))


@pytest.fixture
def three(tmp_path: Path) -> Path:
    return make_exp(tmp_path / "runs", "three_seeds", THREE)


@pytest.fixture
def nowin(tmp_path: Path) -> Path:
    return make_exp(tmp_path / "runs", "no_win", NOWIN)


@pytest.fixture
def one(tmp_path: Path) -> Path:
    return make_exp(tmp_path / "runs", "one_seed", ONE)


def by_gid(ax, gid: str):
    found = [a for a in ax.get_children() if a.get_gid() == gid]
    assert len(found) <= 1, gid
    return found[0] if found else None


def gids(ax, prefix: str) -> list[str]:
    return sorted(a.get_gid() for a in ax.get_children() if str(a.get_gid()).startswith(prefix + ":"))


def figure_legend_texts(fig) -> list[str]:
    assert len(fig.legends) == 1, "one figure-level legend"
    return [t.get_text() for t in fig.legends[0].get_texts()]


def axes_title(ax) -> str:
    """The title of an axes wherever it is aligned (matplotlib keeps left, centre and right titles apart)."""
    titles = [ax.get_title(loc=loc) for loc in ("left", "center", "right")]
    found = [t for t in titles if t]
    assert len(found) <= 1, titles
    return found[0] if found else ""


def handle_colour(handle) -> str:
    """Colour of a legend key, whether it is a line or a patch."""
    colour = handle.get_color() if hasattr(handle, "get_color") else handle.get_facecolor()
    return to_hex(colour)


def bound_call(func, args, kwargs) -> dict:
    bound = inspect.signature(func).bind(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


# ----------------------------------------------------------- constants and signatures
def test_constants_and_signatures():
    assert DEATH_CAUSE_COLUMNS == ("env_steps", "n_seeds", "wall_mean", "wall_ci95", "self_mean", "self_ci95",
                                   "starve_mean", "starve_ci95", "win_mean", "win_ci95")
    assert CAUSE_COLORS == {"wall": "#2a78d6", "self": "#eb6834", "starve": "#1baf7a", "win": "#eda100"}
    assert DEFAULT_CAUSE_YLABEL == "fraction of evaluation episodes"
    params = inspect.signature(death_cause_figure).parameters
    assert list(params) == ["experiments", "title", "ylabel"]
    assert params["title"].default is None and params["ylabel"].default == DEFAULT_CAUSE_YLABEL
    params = inspect.signature(plot_death_causes).parameters
    assert list(params) == ["experiments", "out", "title", "ylabel"]
    assert params["out"].default is inspect.Parameter.empty
    assert params["title"].default is None and params["ylabel"].default == DEFAULT_CAUSE_YLABEL


# ------------------------------------------------------------- death_cause_curve
def test_columns_and_env_steps_are_those_of_the_learning_curve(three):
    curve = death_cause_curve(three)
    assert tuple(curve.columns) == DEATH_CAUSE_COLUMNS
    assert curve["env_steps"].tolist() == learning_curve(three)["env_steps"].tolist() == list(STEPS)
    assert curve["n_seeds"].tolist() == [3, 3, 3] and np.issubdtype(curve["n_seeds"].dtype, np.integer)
    assert death_cause_curve(str(three)).equals(curve)


def test_means_and_ci95_equal_an_independent_recomputation(three, nowin):
    for exp, table in ((three, THREE), (nowin, NOWIN)):
        pd.testing.assert_frame_equal(death_cause_curve(exp), expected_curve(table), rtol=1e-12)
    curve = death_cause_curve(three)
    # by hand at step 200: wall 0.2 0.3 0.1 -> mean 0.2, sd 0.1; win 0.1 0.1 0.2 -> mean 0.4 / 3
    assert curve["wall_mean"].iloc[2] == pytest.approx(0.2, rel=1e-12)
    assert curve["wall_ci95"].iloc[2] == pytest.approx(T2 * 0.1 / math.sqrt(3), rel=1e-12)
    assert curve["win_mean"].iloc[2] == pytest.approx(0.4 / 3, rel=1e-12)
    assert curve["win_mean"].iloc[0] == 0.0 and curve["win_ci95"].iloc[0] == 0.0
    assert curve["wall_mean"].tolist() == pytest.approx([0.9, 0.5, 0.2], rel=1e-12)
    assert curve["self_mean"].tolist() == pytest.approx([1 / 30, 0.3, 0.5], rel=1e-12)
    assert curve["starve_mean"].tolist() == pytest.approx([1 / 15, 0.2, 1 / 6], rel=1e-12)


def test_the_four_means_of_a_row_sum_to_one(three, nowin, one):
    for exp in (three, nowin, one):
        curve = death_cause_curve(exp)
        total = sum(curve[f"{c}_mean"].to_numpy() for c in CAUSES)
        assert np.abs(total - 1.0).max() < 1e-9


def test_one_seed_has_the_ci95_the_learning_curve_gives(one):
    curve = death_cause_curve(one)
    assert curve["n_seeds"].tolist() == [1, 1, 1]
    assert learning_curve(one)["ci95"].isna().all()
    for j, cause in enumerate(CAUSES):
        assert curve[f"{cause}_ci95"].isna().all(), cause
        assert curve[f"{cause}_mean"].tolist() == pytest.approx([row[j] for row in ONE[0]], rel=1e-12)


def test_intersection_rule_over_seeds_with_the_warning(tmp_path):
    exp = tmp_path / "runs" / "ragged"
    write_cause_run(exp, 0, THREE[0] + [(0.0, 0.0, 0.0, 1.0)], steps=STEPS + (300,))  # evaluated longer
    write_cause_run(exp, 1, [THREE[1][0], (0.0, 0.0, 1.0, 0.0)] + THREE[1][1:], steps=(0, 50, 100, 200))  # more often
    write_cause_run(exp, 2, THREE[2])
    with pytest.warns(AggregateWarning, match="2 env_steps value\\(s\\) are not present in all 3 finished runs"):
        curve = death_cause_curve(exp)
    pd.testing.assert_frame_equal(curve, expected_curve(THREE), rtol=1e-12)


def test_missing_fraction_column_names_the_run_and_the_column(tmp_path):
    exp = tmp_path / "runs" / "broken"
    write_cause_run(exp, 0, THREE[0])
    bad = write_cause_run(exp, 1, THREE[1], drop="eval_frac_win", dir_name="seed_b")
    assert learning_curve(exp)["env_steps"].tolist() == list(STEPS)  # the score curve does not need the fractions
    with pytest.raises(ValueError) as exc:
        death_cause_curve(exp)
    assert "eval_frac_win" in str(exc.value) and bad.name in str(exc.value)
    for column in ("eval_frac_wall", "eval_frac_self", "eval_frac_starve"):
        (bad / "metrics.csv").write_text(cause_metrics_csv(
            [(s, 1.0, f) for s, f in zip(STEPS, THREE[1])], drop=column), encoding="utf-8")
        with pytest.raises(ValueError, match=column):
            death_cause_curve(exp)


def test_same_errors_as_load_runs(tmp_path):
    with pytest.raises(ValueError):
        death_cause_curve(tmp_path / "does_not_exist")
    exp = tmp_path / "runs" / "running"
    write_run(exp, 0, FIVE_SEEDS[1], [(0, 0.0)], finished=False)
    with pytest.warns(AggregateWarning), pytest.raises(ValueError, match="no finished run"):
        death_cause_curve(exp)


# ----------------------------------------------------------- death_cause_figure
def test_one_axes_per_experiment_in_one_row_with_titles_limits_and_formatter(three, nowin, one):
    fig = death_cause_figure([("three", three), ("no win", nowin), ("one", one)])
    assert isinstance(fig.canvas, FigureCanvasAgg)
    assert len(fig.axes) == 3
    assert [axes_title(ax) for ax in fig.axes] == ["three", "no win", "one"]
    for ax in fig.axes:
        assert ax.get_ylim() == (0.0, 1.0)
        assert ax.xaxis.get_major_formatter()(200_000, 0) == "200k" and ax.xaxis.get_major_formatter()(0, 0) == "0"
        assert ax.get_xlim()[0] <= 0 and ax.get_xlim()[1] >= 200
    fig.draw_without_rendering()
    boxes = [ax.get_position() for ax in fig.axes]
    assert boxes[0].x0 < boxes[1].x0 < boxes[2].x0  # left to right, in the given order
    assert all(abs(b.y0 - boxes[0].y0) < 1e-6 and abs(b.y1 - boxes[0].y1) < 1e-6 for b in boxes)  # one row
    assert fig.get_suptitle() == ""
    assert any(ax.get_ylabel() == DEFAULT_CAUSE_YLABEL for ax in fig.axes)
    fig = death_cause_figure([("three", three)], title="Deaths over training", ylabel="share of episodes")
    assert fig.get_suptitle() == "Deaths over training" and len(fig.axes) == 1
    assert any(ax.get_ylabel() == "share of episodes" for ax in fig.axes)


def test_one_line_per_cause_in_order_with_the_curve_values_and_colours(three, nowin):
    fig = death_cause_figure([("three", three), ("no win", nowin)])
    for i, (exp, drawn) in enumerate(((three, CAUSES), (nowin, CAUSES[:3]))):
        ax = fig.axes[i]
        curve = death_cause_curve(exp)
        assert gids(ax, "cause") == sorted(f"cause:{i}:{name}" for name in drawn)
        assert len(ax.lines) == len(drawn)
        for name in drawn:
            line = by_gid(ax, f"cause:{i}:{name}")
            assert np.array_equal(line.get_xdata(), curve["env_steps"].to_numpy())
            assert np.array_equal(line.get_ydata(), curve[f"{name}_mean"].to_numpy())
            assert to_hex(line.get_color()) == CAUSE_COLORS[name]
        # the lines of the axes are drawn wall, self, starve(, win)
        assert [line.get_gid() for line in ax.lines] == [f"cause:{i}:{name}" for name in drawn]
    # by hand: experiment 0 at step 200
    assert by_gid(fig.axes[0], "cause:0:wall").get_ydata().tolist() == pytest.approx([0.9, 0.5, 0.2], rel=1e-12)
    assert by_gid(fig.axes[0], "cause:0:win").get_ydata().tolist() == pytest.approx([0.0, 0.0, 0.4 / 3], rel=1e-12)
    assert by_gid(fig.axes[1], "cause:1:wall").get_ydata().tolist() == pytest.approx([0.95, 0.6, 0.4], rel=1e-12)


def test_bands_mean_plus_minus_ci95_only_with_more_than_one_seed(three, nowin, one):
    fig = death_cause_figure([("three", three), ("one", one), ("no win", nowin)])
    for i, (exp, drawn) in enumerate(((three, CAUSES), (one, ()), (nowin, CAUSES[:3]))):
        ax = fig.axes[i]
        assert gids(ax, "cause_band") == sorted(f"cause_band:{i}:{name}" for name in drawn)
        assert len(ax.collections) == len(drawn)
        curve = death_cause_curve(exp)
        for name in drawn:
            band = by_gid(ax, f"cause_band:{i}:{name}")
            bounds = band_bounds(band)
            assert sorted(bounds) == curve["env_steps"].tolist()
            for x, mean, ci in zip(curve["env_steps"], curve[f"{name}_mean"], curve[f"{name}_ci95"]):
                assert bounds[float(x)] == pytest.approx((mean - ci, mean + ci), rel=1e-12, abs=1e-12), (name, x)
            assert tuple(band.get_facecolor()[0]) == pytest.approx(to_rgba(CAUSE_COLORS[name], BAND_ALPHA))
    # by hand: three seeds, wall at step 200: 0.2 +- 4.303 * 0.1 / sqrt(3)
    half = T2 * 0.1 / math.sqrt(3)
    assert band_bounds(by_gid(fig.axes[0], "cause_band:0:wall"))[200.0] == pytest.approx((0.2 - half, 0.2 + half), rel=1e-12)
    # one seed: a line per cause and nothing else
    assert gids(fig.axes[1], "cause") == ["cause:1:self", "cause:1:starve", "cause:1:wall"]


def test_win_line_is_absent_when_win_is_zero_everywhere(nowin, three, one):
    fig = death_cause_figure([("no win", nowin), ("three", three)])
    assert by_gid(fig.axes[0], "cause:0:win") is None and by_gid(fig.axes[0], "cause_band:0:win") is None
    assert by_gid(fig.axes[1], "cause:1:win") is not None and by_gid(fig.axes[1], "cause_band:1:win") is not None
    assert all(by_gid(fig.axes[0], f"cause:0:{name}") is not None for name in ("wall", "self", "starve"))
    fig = death_cause_figure([("one", one)])
    assert gids(fig.axes[0], "cause") == ["cause:0:self", "cause:0:starve", "cause:0:wall"]


def test_figure_level_legend_lists_the_causes_drawn_and_the_ci_patch(three, nowin, one):
    assert figure_legend_texts(death_cause_figure([("three", three)])) == ["wall", "self", "starve", "win", "95% CI over 3 seeds"]
    assert figure_legend_texts(death_cause_figure([("no win", nowin)])) == ["wall", "self", "starve", "95% CI over 2 seeds"]
    assert figure_legend_texts(death_cause_figure([("one", one)])) == ["wall", "self", "starve"]  # no band, no CI key
    fig = death_cause_figure([("no win", nowin), ("three", three)])
    assert figure_legend_texts(fig) == ["wall", "self", "starve", "win", "95% CI over seeds"]  # 2 and 3 seeds
    handles = fig.legends[0].legend_handles
    assert [handle_colour(h) for h in handles[:4]] == [CAUSE_COLORS[c] for c in CAUSES]
    assert fig.legends[0].get_texts()[4].get_text() == "95% CI over seeds"
    assert all(ax.get_legend() is None for ax in fig.axes)


def test_bad_inputs_are_refused(three, tmp_path):
    with pytest.raises(ValueError):
        death_cause_figure([])
    with pytest.raises(ValueError):
        death_cause_figure([("same", three), ("same", three)])
    with pytest.raises(ValueError):
        plot_death_causes([], out=tmp_path / "x.png")
    with pytest.raises(ValueError):
        plot_death_causes([("same", three), ("same", three)], out=tmp_path / "x.png")
    with pytest.raises(ValueError):
        plot_death_causes([("a", tmp_path / "no_such_experiment")], out=tmp_path / "x.png")
    assert not (tmp_path / "x.png").exists()


# ------------------------------------------------------------------- the file
def test_plot_death_causes_writes_the_png_and_sets_the_title(three, nowin, tmp_path):
    out = tmp_path / "figures" / "nested" / "causes.png"
    assert not out.parent.exists()
    returned = plot_death_causes([("three", three), ("no win", nowin)], out)
    assert returned == out and isinstance(returned, Path) and out.is_file()
    width, height = png_size(out)
    assert width > 0 and height > 0
    first = out.read_bytes()
    assert plot_death_causes([("three", str(three)), ("no win", str(nowin))], out=str(out)) == out  # str paths, overwritten
    assert out.read_bytes() == first
    plot_death_causes([("three", three)], out=out, title="Part E", ylabel="share")
    assert out.read_bytes() != first and png_size(out)[0] > 0
    fig = death_cause_figure([("three", three)], title="Part E")
    assert fig._suptitle is not None and fig._suptitle.get_text() == "Part E" and fig.get_suptitle() == "Part E"


# ------------------------------------------------------------------------ CLI
def test_cli_causes_flag_calls_plot_death_causes(three, nowin, tmp_path, monkeypatch, capsys):
    out = tmp_path / "figures" / "cli.png"
    calls = []
    real = plots.plot_death_causes

    def capture(*args, **kwargs):
        call = bound_call(real, args, kwargs)
        calls.append(call)
        return Path(call["out"])

    with monkeypatch.context() as patch:
        patch.setattr(plots, "plot_death_causes", capture)
        patch.setattr(plots, "plot_learning_curves", lambda *a, **k: pytest.fail("--causes must not draw the learning curves"))
        patch.setattr(plots, "plot_value_curves", lambda *a, **k: pytest.fail("--causes must not draw the value figure"))
        assert main(["--causes", "--exp", f"{three}=three seeds", str(nowin), "--out", str(out), "--title", "Part E"]) == 0
        assert main(["--causes", "--exp", str(three), "--out", str(out), "--ylabel", "share of episodes"]) == 0
    first, second = calls
    assert [(label, str(path)) for label, path in first["experiments"]] == [("three seeds", str(three)), ("no_win", str(nowin))]
    assert str(first["out"]) == str(out) and first["title"] == "Part E"
    assert first["ylabel"] == DEFAULT_CAUSE_YLABEL  # an unset --ylabel means the cause label with --causes
    assert [(label, str(path)) for label, path in second["experiments"]] == [("three_seeds", str(three))]
    assert second["title"] is None and second["ylabel"] == "share of episodes"
    capsys.readouterr()
    assert main(["--causes", "--exp", f"{three}=three", str(nowin), "--out", str(out)]) == 0  # the real thing
    assert capsys.readouterr().out.strip() == str(out) and png_size(out)[0] > 0


def test_cli_causes_usage_errors(three, tmp_path, capsys):
    bfs, _ = write_baselines(tmp_path)
    out = tmp_path / "x.png"
    for argv in (
        ["--causes", "--values", "--exp", str(three), "--out", str(out)],  # one figure at a time
        ["--causes", "--start", "--exp", str(three), "--out", str(out)],  # --start belongs to --values
        ["--causes", "--exp", str(three), "--baseline", str(bfs), "--out", str(out)],  # no baselines on the cause figure
        ["--causes", "--out", str(out)],  # nothing to plot
        ["--causes", "--exp", str(three)],  # --out is required
    ):
        with pytest.raises(SystemExit) as exc:
            main(argv)
        assert exc.value.code == 2, argv
        assert capsys.readouterr().err != ""
    assert not out.exists()


def test_cli_without_causes_is_unchanged(three, tmp_path, monkeypatch):
    bfs, _ = write_baselines(tmp_path)
    out = tmp_path / "learning.png"
    calls = []

    def capture_learning(experiments, baselines, **kwargs):
        calls.append(("learning", experiments, baselines, kwargs))
        return Path(kwargs["out"])

    def capture_values(experiments, **kwargs):
        calls.append(("values", experiments, kwargs))
        return Path(kwargs["out"])

    with monkeypatch.context() as patch:
        patch.setattr(plots, "plot_learning_curves", capture_learning)
        patch.setattr(plots, "plot_value_curves", capture_values)
        patch.setattr(plots, "plot_death_causes", lambda *a, **k: pytest.fail("the cause figure needs --causes"))
        assert main(["--exp", f"{three}=three", "--baseline", f"{bfs}=BFS", "--out", str(out)]) == 0
        assert main(["--values", "--exp", f"{three}=three", "--out", str(out)]) == 0
    assert calls[0] == ("learning", [("three", str(three))], [("BFS", str(bfs))],
                        {"out": str(out), "title": None, "ylabel": DEFAULT_YLABEL})
    assert calls[1] == ("values", [("three", str(three))],
                        {"out": str(out), "which": "visited", "title": None, "ylabel": DEFAULT_VALUE_YLABEL})

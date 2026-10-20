"""Learning-curve figure regenerated from run directories (Part C: "one
evaluation curve"; project rule: every figure comes from the logs).

What the figure shows, in the words of an oral defence:

* x: environment steps. y: the greedy evaluation score (``eval_mean_score``
  of ``metrics.csv``: the mean over the evaluation episodes, played on the
  same boards as every baseline). The y axis starts at zero.
* One LINE per experiment: the mean over training seeds at every evaluation
  step (``aggregate.learning_curve``; only steps present in all finished runs).
* One BAND per experiment: ``mean +- t(n - 1) * sd / sqrt(n)`` over the ``n``
  seeds at that step, the Student-t 95 % interval of the mean with the seed as
  the unit of replication. It is an interval for the mean over seeds, not the
  range of single episodes. One seed has no spread: a line and no band.
* One horizontal DASHED line per baseline: the mean score of one evaluation
  of a non-learning player (``aggregate.summarise_baseline``). A reference
  level, not a curve: baselines do not train.

Encoding: learner series take the categorical colours of ``SERIES_COLORS`` in
the order given (fixed slots, never cycled: more than eight experiments is a
``ValueError`` - split the figure). Baselines are all drawn in the secondary
ink and told apart by the label at their right end. The legend, a row above
the axes, explains the encodings (series, band, dashed reference); with at
most four experiments each curve is also labelled at its right end. Labels
that would overlap are moved apart and tied to their line by a thin leader.

No display is needed and no global matplotlib state is touched: the figure is
built with the object-oriented API on an Agg canvas (``matplotlib.pyplot`` is
never imported, ``matplotlib.use`` is never called), so importing this module
in a notebook leaves the inline backend of the notebook alone.

Value figure (M7, :func:`plot_value_curves`): the overestimation measurement
of van Hasselt et al. (2016). Per experiment a SOLID line, the mean over
seeds of the network's value estimate ``max_a Q`` on the states the greedy
policy visits during evaluation, and a DASHED line in the same colour, the
discounted return the policy then actually realises from those states
(``aggregate.value_curve``); both with their seed-level t bands. The vertical
distance between the two is the overestimation; a Double DQN row is read
against a DQN row. No baselines (a non-learning player has no estimate).

CLI::

    python -m snake_rl.analysis.plots --exp DIR[=LABEL] ... [--baseline FILE[=LABEL] ...] --out FILE.png
                                      [--title T] [--ylabel TEXT]
    python -m snake_rl.analysis.plots --values --exp DIR[=LABEL] ... --out FILE.png [--title T] [--ylabel TEXT]
                                      [--start]

e.g. ``--exp "runs/tabular_q_W10=tabular Q" "runs/dqn_A_W10=DQN" --baseline
"runs/baselines/bfs_W10_relative.json=BFS greedy" --out
runs/figures/part_c_W10.png``. Without ``=LABEL`` the label is the directory
name (experiments) or the file stem (baselines). ``--values`` draws the value
figure (``--start``: the first state of each episode instead of all visited
states; no ``--baseline``).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import offset_copy

from snake_rl.analysis.aggregate import learning_curve, summarise_baseline, value_curve
from snake_rl.utils import REPO_ROOT

#: Categorical colours of the learner series, in fixed slot order
#: (blue, orange, aqua, yellow, magenta, green, violet, red).
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
SURFACE = "#fcfcfb"  # figure and axes background
INK = "#0b0b0b"  # title, legend text, direct labels
INK_SECONDARY = "#52514e"  # axis labels, tick labels, baseline lines
INK_MUTED = "#898781"  # leaders
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

FIGSIZE = (7.0, 4.2)  # inches
DPI = 150  # -> 1050 x 630 pixels
LINE_WIDTH = 1.5  # points; 2 px at 96 px per inch
BASELINE_WIDTH = 1.0
HAIRLINE = 0.75
BAND_ALPHA = 0.18
FONT_SIZE = 9.0
#: Curves are labelled at their right end up to this many experiments.
MAX_DIRECT_LABELS = 4

DEFAULT_YLABEL = "mean score (100 greedy episodes)"
DEFAULT_VALUE_YLABEL = "discounted return (greedy evaluation)"
#: Figures are regenerable from the logs: this file may be overwritten.
DEFAULT_OUT = REPO_ROOT / "runs" / "figures" / "learning_curves.png"

#: gid prefixes of the artists, for tests and for restyling in a notebook:
#: ``series:<i>`` line and ``band:<i>`` of experiment i, ``baseline:<j>``.
GID_SERIES, GID_BAND, GID_BASELINE = "series", "band", "baseline"
#: The value figure: ``value:<i>`` / ``value_band:<i>`` (the estimate, solid)
#: and ``return:<i>`` / ``return_band:<i>`` (the realised return, dashed).
GID_VALUE, GID_VALUE_BAND, GID_RETURN, GID_RETURN_BAND = "value", "value_band", "return", "return_band"


def _format_steps(value: float, _pos: int | None = None) -> str:
    """Tick label of a step count: 0 -> '0', 200000 -> '200k', 1200000 -> '1.2M'."""
    for unit, suffix in ((1e6, "M"), (1e3, "k")):
        if abs(value) >= unit:
            return f"{value / unit:g}{suffix}"
    return f"{value:g}"


def _spread(anchors: Sequence[float], gap: float, lo: float, hi: float) -> list[float]:
    """Label positions at least ``gap`` apart, as close to ``anchors`` as possible.

    Least-squares solution that keeps the order of the anchors: with
    ``r_i = q_i - i * gap`` (i = rank) the constraint ``q_{i+1} - q_i >= gap``
    becomes "r is non-decreasing", so r is the isotonic regression of
    ``a_i - i * gap`` (pool adjacent violators). The stack is then pushed
    inside ``[lo, hi]`` as far as it fits.
    """
    n = len(anchors)
    order = sorted(range(n), key=lambda i: (anchors[i], i))
    blocks: list[list[float]] = []  # [sum of targets, count] of each pooled block
    for rank, i in enumerate(order):
        blocks.append([anchors[i] - rank * gap, 1.0])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            total, count = blocks.pop()
            blocks[-1][0] += total
            blocks[-1][1] += count
    pooled = [total / count for total, count in blocks for _ in range(int(count))]
    q = [pooled[rank] + rank * gap for rank in range(n)]
    for rank in range(n):
        q[rank] = max(q[rank], lo + rank * gap)
    for rank in range(n):
        q[rank] = min(q[rank], hi - (n - 1 - rank) * gap)
    placed = [0.0] * n
    for rank, i in enumerate(order):
        placed[i] = q[rank]
    return placed


def _legend_columns(labels: Sequence[str]) -> int:
    """Columns of the legend row above the axes (a character budget, not a measurement)."""
    budget, per_entry = 74, 7  # characters across the axes at FONT_SIZE; handle + spacing of one entry
    if sum(len(s) + per_entry for s in labels) <= budget:
        return len(labels)
    return max(1, min(len(labels), budget // (max(len(s) for s in labels) + per_entry)))


def learning_curve_figure(
    experiments: Sequence[tuple[str, str | Path]],
    baselines: Sequence[tuple[str, str | Path]] = (),
    title: str | None = None,
    ylabel: str = DEFAULT_YLABEL,
) -> Figure:
    """The figure of :func:`plot_learning_curves`, not saved.

    One axes. Artists carry a gid: ``series:<i>`` (the mean line of
    experiment i), ``band:<i>`` (its interval; absent for one seed) and
    ``baseline:<j>`` (the dashed reference line of baseline j).
    """
    experiments = [(str(label), Path(d)) for label, d in experiments]
    baselines = [(str(label), Path(f)) for label, f in baselines]
    if not experiments:
        raise ValueError("at least one experiment is needed: a baseline alone has no curve")
    if len(experiments) > len(SERIES_COLORS):
        raise ValueError(
            f"{len(experiments)} experiments, but there are {len(SERIES_COLORS)} series colours and they are "
            f"never reused: split the comparison into several figures"
        )
    labels = [label for label, _ in experiments] + [label for label, _ in baselines]
    if len(set(labels)) != len(labels):
        raise ValueError(f"labels must be unique within one figure, got {labels}")

    curves = [(label, d, learning_curve(d)) for label, d in experiments]
    for _, d, curve in curves:
        if curve.empty:
            raise ValueError(f"{d}: no evaluation step is common to all finished runs")
    levels = [(label, float(summarise_baseline(f)["mean_score"])) for label, f in baselines]

    fig = Figure(figsize=FIGSIZE, dpi=DPI, facecolor=SURFACE, layout="constrained")
    FigureCanvasAgg(fig)  # Agg: renders to a pixel buffer, needs no display
    ax = fig.add_subplot()
    ax.set_facecolor(SURFACE)

    # Right ends to label: (label, anchor xy, coordinate system of the anchor).
    ends: list[tuple[str, tuple[float, float], object]] = []
    band_seeds: set[int] = set()
    x_lo, x_hi, y_top = np.inf, -np.inf, 0.0
    for i, (label, _, curve) in enumerate(curves):
        x = curve["env_steps"].to_numpy(dtype=np.float64)
        mean = curve["mean"].to_numpy(dtype=np.float64)
        half = curve["ci95"].to_numpy(dtype=np.float64)
        upper = mean
        if np.isfinite(half).all():  # nan for a single seed: one replicate has no spread
            upper = mean + half
            band = ax.fill_between(
                x, mean - half, upper, facecolor=SERIES_COLORS[i], edgecolor="none", linewidth=0.0,
                alpha=BAND_ALPHA, zorder=2,
            )
            band.set_gid(f"{GID_BAND}:{i}")
            band_seeds.add(int(curve["n_seeds"].iloc[0]))
        (line,) = ax.plot(
            x, mean, color=SERIES_COLORS[i], linewidth=LINE_WIDTH, solid_capstyle="round",
            solid_joinstyle="round", zorder=4, label=label,
        )
        line.set_gid(f"{GID_SERIES}:{i}")
        if len(curves) <= MAX_DIRECT_LABELS:
            ends.append((label, (float(x[-1]), float(mean[-1])), ax.transData))
        x_lo, x_hi, y_top = min(x_lo, float(x[0])), max(x_hi, float(x[-1])), max(y_top, float(upper.max()))
    for j, (label, level) in enumerate(levels):
        reference = ax.axhline(level, color=INK_SECONDARY, linewidth=BASELINE_WIDTH, linestyle="--", zorder=3)
        reference.set_gid(f"{GID_BASELINE}:{j}")
        # Baselines share one colour: the label at the line is what identifies them, so it is always drawn.
        ends.append((label, (1.0, level), ax.get_yaxis_transform()))
        y_top = max(y_top, level)

    pad = 0.012 * (x_hi - x_lo) if x_hi > x_lo else 1.0
    ax.set_xlim(x_lo - pad, x_hi + pad)
    ax.set_ylim(0.0, 1.06 * y_top if y_top > 0.0 else 1.0)  # a score is a count of food eaten: the axis starts at 0
    _style_axes(fig, ax, ylabel, title)

    # Legend: a row above the axes, so that it can never cover a curve.
    handles: list[Line2D | Patch] = [
        Line2D([], [], color=SERIES_COLORS[i], linewidth=LINE_WIDTH, label=label)
        for i, (label, _, _) in enumerate(curves)
    ]
    if band_seeds:
        handles.append(_band_handle(band_seeds))
    if levels:
        handles.append(
            Line2D([], [], color=INK_SECONDARY, linewidth=BASELINE_WIDTH, linestyle="--",
                   label="non-learning baseline")
        )
    _legend_row(ax, handles)

    _label_line_ends(fig, ax, ends, x_reach=x_hi - 0.04 * (x_hi - x_lo))
    return fig


def _style_axes(fig: Figure, ax, ylabel: str, title: str | None) -> None:
    """Axis labels, grid, spines, ticks and the title, shared by the figures."""
    ax.xaxis.set_major_formatter(FuncFormatter(_format_steps))
    ax.set_xlabel("environment steps", fontsize=FONT_SIZE + 1, color=INK_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE + 1, color=INK_SECONDARY)
    ax.grid(axis="y", color=GRID, linewidth=HAIRLINE, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(HAIRLINE)
    ax.tick_params(axis="x", labelsize=FONT_SIZE, labelcolor=INK_SECONDARY, color=AXIS, length=3.0, width=HAIRLINE)
    ax.tick_params(axis="y", labelsize=FONT_SIZE, labelcolor=INK_SECONDARY, length=0.0)
    if title is not None:
        fig.suptitle(title, x=0.015, ha="left", fontsize=FONT_SIZE + 2, color=INK)


def _band_handle(band_seeds: set[int]) -> Patch:
    over = f"{next(iter(band_seeds))} seeds" if len(band_seeds) == 1 else "seeds"
    return Patch(facecolor=INK_SECONDARY, alpha=BAND_ALPHA, edgecolor="none", label=f"95% CI over {over}")


def _legend_row(ax, handles: Sequence[Line2D | Patch]) -> None:
    """A legend row above the axes, so that it can never cover a curve."""
    ax.legend(
        handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.01), borderaxespad=0.0, frameon=False,
        ncols=_legend_columns([h.get_label() for h in handles]), fontsize=FONT_SIZE, labelcolor=INK,
        handlelength=2.2, columnspacing=1.4,  # 2.2 em: four whole dashes in the key of a dashed line
    )


def value_curve_figure(
    experiments: Sequence[tuple[str, str | Path]],
    which: str = "visited",
    title: str | None = None,
    ylabel: str = DEFAULT_VALUE_YLABEL,
) -> Figure:
    """The figure of :func:`plot_value_curves`, not saved (module docstring).

    One axes. Per experiment i, in ``SERIES_COLORS[i]``: the solid line
    ``value:<i>`` (mean over seeds of the value estimate) with its band
    ``value_band:<i>``, and the dashed line ``return:<i>`` (mean realised
    return) with its band ``return_band:<i>``; no bands for one seed.
    """
    experiments = [(str(label), Path(d)) for label, d in experiments]
    if not experiments:
        raise ValueError("at least one experiment is needed")
    if len(experiments) > len(SERIES_COLORS):
        raise ValueError(
            f"{len(experiments)} experiments, but there are {len(SERIES_COLORS)} series colours and they are "
            f"never reused: split the comparison into several figures"
        )
    labels = [label for label, _ in experiments]
    if len(set(labels)) != len(labels):
        raise ValueError(f"labels must be unique within one figure, got {labels}")
    curves = [(label, d, value_curve(d, which)) for label, d in experiments]
    for _, d, curve in curves:
        if curve.empty:
            raise ValueError(f"{d}: no evaluation step is common to all finished runs")

    fig = Figure(figsize=FIGSIZE, dpi=DPI, facecolor=SURFACE, layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    ax.set_facecolor(SURFACE)

    ends: list[tuple[str, tuple[float, float], object]] = []
    band_seeds: set[int] = set()
    x_lo, x_hi, y_lo, y_hi = np.inf, -np.inf, 0.0, 0.0
    series = ((GID_VALUE, GID_VALUE_BAND, "value_mean", "value_ci95", "-"),
              (GID_RETURN, GID_RETURN_BAND, "return_mean", "return_ci95", "--"))
    for i, (label, _, curve) in enumerate(curves):
        x = curve["env_steps"].to_numpy(dtype=np.float64)
        for gid_line, gid_band, mean_column, ci_column, style in series:
            mean = curve[mean_column].to_numpy(dtype=np.float64)
            half = curve[ci_column].to_numpy(dtype=np.float64)
            lower, upper = mean, mean
            if np.isfinite(half).all():  # nan for a single seed: one replicate has no spread
                lower, upper = mean - half, mean + half
                band = ax.fill_between(
                    x, lower, upper, facecolor=SERIES_COLORS[i], edgecolor="none", linewidth=0.0,
                    alpha=BAND_ALPHA, zorder=2,
                )
                band.set_gid(f"{gid_band}:{i}")
                band_seeds.add(int(curve["n_seeds"].iloc[0]))
            (line,) = ax.plot(
                x, mean, color=SERIES_COLORS[i], linewidth=LINE_WIDTH, linestyle=style, solid_capstyle="round",
                solid_joinstyle="round", zorder=4, label=label if style == "-" else None,
            )
            line.set_gid(f"{gid_line}:{i}")
            y_lo, y_hi = min(y_lo, float(lower.min())), max(y_hi, float(upper.max()))
        if len(curves) <= MAX_DIRECT_LABELS:
            ends.append((label, (float(x[-1]), float(curve["value_mean"].iloc[-1])), ax.transData))
        x_lo, x_hi = min(x_lo, float(x[0])), max(x_hi, float(x[-1]))

    pad = 0.012 * (x_hi - x_lo) if x_hi > x_lo else 1.0
    ax.set_xlim(x_lo - pad, x_hi + pad)
    ax.set_ylim(min(0.0, y_lo), 1.06 * y_hi if y_hi > 0.0 else 1.0)  # a death costs -1: the return can be negative
    _style_axes(fig, ax, ylabel, title)

    handles: list[Line2D | Patch] = [
        Line2D([], [], color=SERIES_COLORS[i], linewidth=LINE_WIDTH, label=label)
        for i, (label, _, _) in enumerate(curves)
    ]
    handles.append(Line2D([], [], color=INK_SECONDARY, linewidth=LINE_WIDTH, label="max-Q estimate"))
    handles.append(Line2D([], [], color=INK_SECONDARY, linewidth=LINE_WIDTH, linestyle="--", label="realised return"))
    if band_seeds:
        handles.append(_band_handle(band_seeds))
    _legend_row(ax, handles)

    _label_line_ends(fig, ax, ends, x_reach=x_hi - 0.04 * (x_hi - x_lo))
    return fig


def _label_line_ends(fig: Figure, ax, ends: Sequence[tuple[str, tuple[float, float], object]], x_reach: float) -> None:
    """Write each label of ``ends`` (label, anchor, anchor coordinates) next to the right end of its line.

    A line that reaches ``x_reach`` is labelled in the right margin at the
    height of its end; margin labels that would overlap are moved apart
    (:func:`_spread`) and a moved label is tied to its line end by a thin
    leader. A curve that stops earlier is labelled right after its last
    point, inside the axes: a long leader at the height of the curve would
    read as a continuation of the curve.
    """
    if not ends:
        return
    fig.draw_without_rendering()  # runs the layout: the height of the axes is needed in points
    y_lo, y_hi = ax.get_ylim()
    height_pt = ax.get_window_extent().height * 72.0 / fig.dpi
    gap = 1.35 * FONT_SIZE / height_pt * (y_hi - y_lo)  # one line of text, in data units
    to_data = ax.transData.inverted()
    anchors = [to_data.transform(coords.transform(xy)) for _, xy, coords in ends]
    style = {"ha": "left", "va": "center", "fontsize": FONT_SIZE, "color": INK, "annotation_clip": False}

    margin = [k for k, a in enumerate(anchors) if float(a[0]) >= x_reach]
    placed = _spread([float(anchors[k][1]) for k in margin], gap, y_lo, y_hi)
    text_coords = offset_copy(ax.get_yaxis_transform(), fig=fig, x=6.0, y=0.0, units="points")
    for k, y_text in zip(margin, placed):
        label, xy, coords = ends[k]
        leader = None
        if abs(y_text - float(anchors[k][1])) > 0.2 * gap:
            leader = {"arrowstyle": "-", "color": INK_MUTED, "linewidth": HAIRLINE, "shrinkA": 1.0,
                      "shrinkB": 2.0, "relpos": (0.0, 0.5)}
        ax.annotate(label, xy=xy, xycoords=coords, xytext=(1.0, y_text), textcoords=text_coords,
                    arrowprops=leader, **style)
    for k in sorted(set(range(len(ends))) - set(margin)):
        label, xy, coords = ends[k]
        ax.annotate(label, xy=xy, xycoords=coords, xytext=(6.0, 0.0), textcoords="offset points", zorder=5,
                    bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5}, **style)


def plot_learning_curves(
    experiments: Sequence[tuple[str, str | Path]],
    baselines: Sequence[tuple[str, str | Path]] = (),
    out: str | Path = DEFAULT_OUT,
    title: str | None = None,
    ylabel: str = DEFAULT_YLABEL,
) -> Path:
    """Draw the learning curves of ``experiments`` and save them as a PNG.

    ``experiments`` are ``(label, experiment directory)`` pairs, ``baselines``
    ``(label, EvalResult JSON)`` pairs. Per experiment: the across-seed mean
    of ``eval_mean_score`` against ``env_steps`` as a line, with a translucent
    band ``mean +- Student-t 95 % half-width`` (no band for one seed). Per
    baseline: a horizontal dashed line at that player's mean score. The PNG is
    7 x 4.2 inches at 150 dpi (1050 x 630 pixels). Parent directories are
    created and an existing file is overwritten (figures are regenerable).
    Returns the path written.
    """
    out = Path(out)
    if out.suffix.lower() != ".png":
        raise ValueError(f"out must be a .png file, got {out}")
    fig = learning_curve_figure(experiments, baselines, title=title, ylabel=ylabel)
    return _save_png(fig, out)


def plot_value_curves(
    experiments: Sequence[tuple[str, str | Path]],
    out: str | Path,
    which: str = "visited",
    title: str | None = None,
    ylabel: str = DEFAULT_VALUE_YLABEL,
) -> Path:
    """Draw the value estimates against the realised returns of
    ``experiments`` (``(label, experiment directory)`` pairs) and save them as
    a PNG (module docstring; :func:`value_curve_figure`). ``which`` is
    ``"visited"`` or ``"start"`` (``aggregate.value_curve``). Same size,
    overwrite and ``.png`` rules as :func:`plot_learning_curves`."""
    out = Path(out)
    if out.suffix.lower() != ".png":
        raise ValueError(f"out must be a .png file, got {out}")
    fig = value_curve_figure(experiments, which=which, title=title, ylabel=ylabel)
    return _save_png(fig, out)


def _save_png(fig: Figure, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="png", dpi=DPI)
    return out


# ----------------------------------------------------------------------- CLI
def _split_label(arg: str, exists: Callable[[Path], bool]) -> tuple[str, str | None]:
    """``PATH[=LABEL]`` -> ``(path, label or None)``.

    Directory names of sweeps contain ``=`` themselves (``alpha=0.1``), so
    the path is the longest part before an ``=`` that exists; an argument
    that exists as a whole has no label. If nothing exists the first ``=``
    separates (loading then fails and names the path).
    """
    if "=" not in arg or exists(Path(arg)):
        return arg, None
    cuts = [i for i, ch in enumerate(arg) if ch == "="]
    for cut in reversed(cuts):
        if exists(Path(arg[:cut])):
            return arg[:cut], arg[cut + 1 :]
    return arg[: cuts[0]], arg[cuts[0] + 1 :]


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Learning-curve figure (mean over seeds, 95% CI band) from run directories.")
    p.add_argument("--exp", nargs="+", action="extend", default=[], metavar="DIR[=LABEL]",
                   help="experiment directories <runs_dir>/<exp>; default label: the directory name")
    p.add_argument("--baseline", nargs="+", action="extend", default=[], metavar="FILE[=LABEL]",
                   help="EvalResult JSON files of non-learning players; default label: the file stem")
    p.add_argument("--out", required=True, metavar="FILE.png", help="the figure to write (overwritten if present)")
    p.add_argument("--title", default=None, metavar="T", help="figure title (default: none)")
    p.add_argument("--ylabel", default=None, metavar="TEXT",
                   help=f"y axis label (default: {DEFAULT_YLABEL!r}, with --values {DEFAULT_VALUE_YLABEL!r})")
    p.add_argument("--values", action="store_true",
                   help="the value figure: max-Q estimate (solid) against realised return (dashed) per experiment")
    p.add_argument("--start", action="store_true",
                   help="with --values: the first state of each episode instead of all visited states")
    args = p.parse_args(argv)
    if not args.exp:
        p.error("nothing to plot: give at least one --exp DIR[=LABEL]")
    if args.start and not args.values:
        p.error("--start only applies to the value figure (--values)")
    if args.values and args.baseline:
        p.error("the value figure has no baselines: drop --baseline or --values")

    experiments, baselines = [], []
    for arg in args.exp:
        path, label = _split_label(arg, Path.is_dir)
        experiments.append((label if label is not None else Path(path).resolve().name, path))
    for arg in args.baseline:
        path, label = _split_label(arg, Path.is_file)
        baselines.append((label if label is not None else Path(path).stem, path))
    if any(not label for label, _ in experiments + baselines):
        p.error("empty label: write DIR=LABEL with a non-empty LABEL, or DIR alone")

    if args.values:
        ylabel = args.ylabel if args.ylabel is not None else DEFAULT_VALUE_YLABEL
        which = "start" if args.start else "visited"
        print(plot_value_curves(experiments, out=args.out, which=which, title=args.title, ylabel=ylabel))
        return 0
    ylabel = args.ylabel if args.ylabel is not None else DEFAULT_YLABEL
    print(plot_learning_curves(experiments, baselines, out=args.out, title=args.title, ylabel=ylabel))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

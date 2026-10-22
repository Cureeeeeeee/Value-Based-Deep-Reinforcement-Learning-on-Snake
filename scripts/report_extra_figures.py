"""Composite behaviour figures for the final report, from saved behaviour analyses.

`python -m snake_rl.analysis.behaviour` writes one heat-map per player (or the
seed mean of an experiment) into `runs/behaviour/<name>/`. The report shows
several players side by side on one colour scale, which is what
`snake_rl.analysis.behaviour.heatmap_figure` does; this script only chooses the
panels and reads the saved arrays, so the figure is regenerated from the
analysis outputs and never drawn by hand.

Usage (repo root):
    python scripts/report_extra_figures.py [--behaviour-dir runs/behaviour] [--out-dir runs/report/figures]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snake_rl.analysis.behaviour import heatmap_figure  # noqa: E402
from snake_rl.analysis.plots import death_cause_figure  # noqa: E402

# (figure file, title, [(panel label, experiment directory under runs/)]): the
# death-cause figure of the report with three panels instead of the six of
# `runs/report/figures/reward_B8_W8_gamma099_causes.png`, which is too wide for
# a page; the panels are drawn by the same function as the generated figure.
CAUSE_FIGURES: list[tuple[str, str, list[tuple[str, str]]]] = [
    (
        "causes_B8_W8_gamma099_R0R2R3.png",
        "8x8, observation B8, gamma 0.99: how the evaluation episodes end (5 seeds, 95% t CI)",
        [("R0", "dqn_B8_W8_gamma099"), ("R2", "dqn_B8_W8_gamma099_R2"), ("R3", "dqn_B8_W8_gamma099_R3")],
    ),
]

# (figure file, title, [(panel label, behaviour directory)])
PANELS: list[tuple[str, str, list[tuple[str, str]]]] = [
    (
        "behaviour_heatmaps_W10.png",
        "10x10: where the head is (100 evaluation episodes; DQN panels are the mean over 5 seeds)",
        [
            ("BFS greedy", "bfs_W10"),
            ("Hamiltonian", "hamiltonian_W10"),
            ("B8, gamma 0.9", "dqn_B8_W10_final"),
            ("B8, gamma 0.99, early", "dqn_B8_W10_gamma099_early"),
            ("B8, gamma 0.99, final", "dqn_B8_W10_gamma099_final"),
        ],
    ),
    (
        "behaviour_heatmaps_W8.png",
        "8x8: where the head is (100 evaluation episodes; DQN panels are the mean over 5 seeds)",
        [
            ("BFS greedy", "bfs_W8"),
            ("Hamiltonian", "hamiltonian_W8"),
            ("A, gamma 0.9", "dqn_A_W8_final"),
            ("B8, gamma 0.9", "dqn_B8_W8_final"),
            ("B8, gamma 0.99", "dqn_B8_W8_gamma099_final"),
            ("R3, closer 0.1", "dqn_B8_W8_gamma099_R3_c01_final"),
        ],
    ),
]


# (figure file, [(panel title, [(label, source)])]): per-episode score
# distributions of the final policies (the box plots promised in the plan). A
# source is a baseline JSON name under runs/baselines (100 episodes) or an
# experiment directory under runs/ (the 100 final-evaluation episodes of every
# seed, pooled).
SCORE_DISTRIBUTIONS: list[tuple[str, list[tuple[str, list[tuple[str, str]]]]]] = [
    (
        "score_distributions.png",
        [
            ("8x8: score per episode, final policies", [
                ("random", "random_W8_relative.json"), ("BFS", "bfs_W8_relative.json"),
                ("Hamiltonian", "hamiltonian_W8_relative.json"), ("tabular Q", "tabular_q_W8"),
                ("A (MLP)", "dqn_A_W8"), ("B8 (CNN)", "dqn_B8_W8"), ("B8, gamma 0.99", "dqn_B8_W8_gamma099"),
                ("R3, bonus 0.1", "dqn_B8_W8_gamma099_R3_c01"),
            ]),
            ("10x10: score per episode, final policies", [
                ("random", "random_W10_relative.json"), ("BFS", "bfs_W10_relative.json"),
                ("Hamiltonian", "hamiltonian_W10_relative.json"), ("tabular Q", "tabular_q_W10"),
                ("A (MLP)", "dqn_A_W10"), ("B8 (CNN)", "dqn_B8_W10"), ("B8, G=1, t31", "dqn_B8_W10_G1_t31"),
                ("B8, gamma 0.99", "dqn_B8_W10_gamma099"),
            ]),
        ],
    ),
]


def episode_scores(runs_dir: Path, source: str) -> list[int]:
    """Per-episode scores of a baseline JSON (``*.json`` under runs/baselines)
    or of every seed's ``final_eval.json`` of an experiment, pooled."""
    if source.endswith(".json"):
        files = [runs_dir / "baselines" / source]
    else:
        files = sorted((runs_dir / source).glob("*/final_eval.json"), key=lambda p: int(p.parent.name))
    if not files or not all(f.exists() for f in files):
        raise FileNotFoundError(f"no evaluation records for {source} under {runs_dir}")
    scores: list[int] = []
    for f in files:
        scores += [int(e["score"]) for e in json.loads(f.read_text(encoding="utf-8"))["episodes"]]
    return scores


def score_distribution_figure(panels: list[tuple[str, list[tuple[str, list[int]]]]]) -> Figure:
    """One box plot per player and panel (median, quartiles, 1.5 IQR whiskers,
    outliers as dots), sharing nothing but the style."""
    fig = Figure(figsize=(5.0 * len(panels), 3.6), dpi=150, layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, len(panels), squeeze=False)[0]
    for ax, (title, groups) in zip(axes, panels):
        data = [scores for _, scores in groups]
        ax.boxplot(data, whis=1.5, widths=0.6, showfliers=True,
                   flierprops={"marker": ".", "markersize": 3, "alpha": 0.5},
                   medianprops={"color": "#d62728"}, boxprops={"color": "#1f4e79"}, whiskerprops={"color": "#1f4e79"})
        ax.set_xticks(range(1, len(groups) + 1), [label for label, _ in groups], rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylabel("score (food eaten)")
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    return fig


def load_heat(directory: Path) -> np.ndarray:
    """The saved heat-map of a behaviour directory: the seed mean of an
    experiment (`heatmap_mean.npy`) or the single player's `heatmap.npy`."""
    for name in ("heatmap_mean.npy", "heatmap.npy"):
        path = directory / name
        if path.exists():
            return np.load(path)
    raise FileNotFoundError(f"no heat-map in {directory}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--behaviour-dir", default="runs/behaviour")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out-dir", default="runs/report/figures")
    args = parser.parse_args(argv)
    behaviour_dir = Path(args.behaviour_dir)
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for file_name, title, panels in PANELS:
        heats = [(label, load_heat(behaviour_dir / name)) for label, name in panels]
        fig = heatmap_figure(heats, title=title)
        out = out_dir / file_name
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print("written", out)
    for file_name, title, panels in CAUSE_FIGURES:
        fig = death_cause_figure([(label, runs_dir / name) for label, name in panels], title=title)
        out = out_dir / file_name
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print("written", out)
    for file_name, panels in SCORE_DISTRIBUTIONS:
        fig = score_distribution_figure(
            [(title, [(label, episode_scores(runs_dir, source)) for label, source in groups]) for title, groups in panels]
        )
        out = out_dir / file_name
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print("written", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Every table and figure of the report, regenerated from ``runs/`` in one
command (M10; project rule: no number typed by hand).

A :class:`ReportItem` names a set of matrix experiments
(``snake_rl.analysis.matrix``) and the baselines drawn next to them, and says
which outputs it produces: a results table (``aggregate.results_table`` with
the last-k column), a learning-curve figure (``plots.plot_learning_curves``),
a value figure (``plots.plot_value_curves``, the overestimation measurement)
and / or a death-cause figure (``plots.plot_death_causes``). :func:`build`
writes ``tables/<name>.md`` and ``figures/<name>_<kind>.png`` under the
output directory, plus ``index.md`` (what was built from which experiments
and seeds) and ``manifest.json`` (the same, machine-readable, with the git
hash recorded by every run) - the notebook reads the manifest.

An experiment without a finished run is left out of its items with a warning
and listed as missing; an item with nothing left is skipped, not an error, so
the report can be built while the matrix is still being trained.

CLI::

    python -m snake_rl.analysis.report [--runs-dir runs] [--out-dir runs/report] [--only NAME ...]
                                       [--last-k 5] [--list]
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from snake_rl.analysis import matrix
from snake_rl.analysis.aggregate import (
    CONFIG_FILE,
    FINAL_FILE,
    AggregateWarning,
    _summarise_runs,
    results_table,
    scan_experiment,
    summarise_baseline,
)
from snake_rl.analysis.plots import plot_death_causes, plot_learning_curves, plot_value_curves
from snake_rl.utils import REPO_ROOT, git_hash

#: Output kinds in the order they are written and listed.
KINDS = ("table", "curve", "values", "causes")
INDEX_FILE = "index.md"
MANIFEST_FILE = "manifest.json"
#: ``plots.learning_curve_figure`` draws at most this many experiments.
MAX_CURVES = 8


@dataclass(frozen=True)
class ReportItem:
    """One table / figure group of the report."""

    name: str
    W: int
    experiments: tuple[tuple[str, str], ...]
    baselines: tuple[str, ...] = ()
    table: bool = True
    curve: bool = True
    values: bool = False
    causes: bool = False
    title: str | None = None

    def kinds(self) -> tuple[str, ...]:
        return tuple(kind for kind in KINDS if getattr(self, kind))


def _item(name: str, W: int, experiments: Sequence[tuple[str, str]], baselines: Sequence[str] = (), **kw: Any) -> ReportItem:
    return ReportItem(name, W, tuple((n, lab) for n, lab in experiments), tuple(baselines), **kw)


def _labelled(names: Sequence[str]) -> list[tuple[str, str]]:
    """``(name, matrix label)`` pairs."""
    return [(n, matrix.experiment(n).label) for n in names]


_ALL = ("random", "bfs", "hamiltonian")
_CI = "(5 seeds, 95% t CI)"
_REWARD_LABELS = ("R0", "R1", "R2", "R3", "R3 symmetric", "R3, H=1000")


def _reward_item(name: str, W: int, base: str, variants: Sequence[str], title: str) -> ReportItem:
    names = [base, *(f"{base}_{v}" for v in variants)]
    return _item(name, W, list(zip(names, _REWARD_LABELS)), ["bfs"], causes=True, title=title)


ITEMS: tuple[ReportItem, ...] = (
    _item("partC_W8", 8, _labelled(["tabular_q_W8", "tabular_q_W8_gamma0", "dqn_A_W8"]), _ALL,
          title=f"8x8: tabular Q-learning and DQN on observation A {_CI}"),
    _item("partC_W10", 10, _labelled(["tabular_q_W10", "tabular_q_W10_gamma0", "dqn_A_W10"]), _ALL,
          title=f"10x10: tabular Q-learning and DQN on observation A {_CI}"),
    _item("obs_W8", 8, _labelled(["dqn_A_W8", "dqn_B3_W8", "dqn_B7_W8", "dqn_B8_W8"]), ["bfs"],
          title=f"8x8: state representation {_CI}"),
    _item("obs_W10", 10, _labelled(["dqn_A_W10", "dqn_B3_W10", "dqn_B7_W10", "dqn_B8_W10"]), ["bfs"],
          title=f"10x10: state representation {_CI}"),
    _item("gamma_W8", 8, _labelled(["dqn_A_W8", "dqn_A_W8_gamma099", "dqn_B8_W8", "dqn_B8_W8_gamma099"]), ["bfs"],
          values=True, title=f"8x8: discount factor x representation {_CI}"),
    _item("gamma_W10", 10, _labelled(["dqn_A_W10", "dqn_B8_W10", "dqn_B8_W10_gamma099"]), ["bfs"],
          values=True, title=f"10x10: discount factor x representation {_CI}"),
    _item("components_A_W8", 8, _labelled(["dqn_A_W8", "dqn_A_W8_noreplay", "dqn_A_W8_noreplay_t31",
                                           "dqn_A_W8_notarget", "dqn_A_W8_noreplay_notarget"]),
          title=f"8x8, observation A: replay and target network {_CI}"),
    _item("components_B8_W8", 8, _labelled(["dqn_B8_W8", "dqn_B8_W8_noreplay", "dqn_B8_W8_noreplay_t31",
                                            "dqn_B8_W8_notarget", "dqn_B8_W8_noreplay_notarget"]),
          title=f"8x8, observation B8: replay and target network {_CI}"),
    _item("components_A_W10", 10, _labelled(["dqn_A_W10", "dqn_A_W10_noreplay_t31", "dqn_A_W10_notarget"]),
          title=f"10x10, observation A: replay and target network {_CI}"),
    _item("components_B8_W10", 10, _labelled(["dqn_B8_W10", "dqn_B8_W10_noreplay", "dqn_B8_W10_noreplay_t31",
                                              "dqn_B8_W10_notarget"]),
          title=f"10x10, observation B8: replay and target network {_CI}"),
    _item("budget_W8", 8, _labelled(["dqn_A_W8", "dqn_A_W8_G1", "dqn_A_W8_G4", "dqn_A_W8_G8_b1024",
                                     "dqn_B8_W8", "dqn_B8_W8_G1", "dqn_B8_W8_G1_t31"]),
          title=f"8x8: gradient steps per vectorised step {_CI}"),
    _item("budget_W10", 10, _labelled(["dqn_B8_W10", "dqn_B8_W10_G1_t31"]),
          title=f"10x10: gradient steps per vectorised step {_CI}"),
    _item("capacity_W8", 8, _labelled(["dqn_A_W8", "dqn_A_W8_h16", "dqn_A_W8_linear"]),
          title=f"8x8, observation A: network capacity {_CI}"),
    _item("variants_A_W8", 8, _labelled(["dqn_A_W8", "dqn_A_W8_double", "dqn_A_W8_dueling", "dqn_A_W8_per"]),
          title=f"8x8, observation A: Double, Dueling, PER {_CI}"),
    _item("variants_B8_W8", 8, _labelled(["dqn_B8_W8", "dqn_B8_W8_double", "dqn_B8_W8_dueling", "dqn_B8_W8_per"]),
          title=f"8x8, observation B8: Double, Dueling, PER {_CI}"),
    _item("values_double_A_W8", 8, [("dqn_A_W8_gamma099", "DQN"), ("dqn_A_W8_gamma099_double", "Double DQN")],
          table=False, curve=False, values=True,
          title=f"8x8, observation A, gamma 0.99: value estimate against realised return {_CI}"),
    _item("values_double_B8_W8", 8, [("dqn_B8_W8_gamma099", "DQN"), ("dqn_B8_W8_gamma099_double", "Double DQN")],
          table=False, curve=False, values=True,
          title=f"8x8, observation B8, gamma 0.99: value estimate against realised return {_CI}"),
    _reward_item("reward_A_W8_gamma09", 8, "dqn_A_W8", ["R1", "R2", "R3"],
                 f"8x8, observation A, gamma 0.9: reward design {_CI}"),
    _reward_item("reward_A_W8_gamma099", 8, "dqn_A_W8_gamma099", ["R1", "R2", "R3", "R3sym", "R3_H1000"],
                 f"8x8, observation A, gamma 0.99: reward design {_CI}"),
    _reward_item("reward_B8_W8_gamma09", 8, "dqn_B8_W8", ["R1", "R2", "R3"],
                 f"8x8, observation B8, gamma 0.9: reward design {_CI}"),
    _reward_item("reward_B8_W8_gamma099", 8, "dqn_B8_W8_gamma099", ["R1", "R2", "R3", "R3sym", "R3_H1000"],
                 f"8x8, observation B8, gamma 0.99: reward design {_CI}"),
    _reward_item("reward_B8_W10_gamma099", 10, "dqn_B8_W10_gamma099", ["R1", "R2", "R3"],
                 f"10x10, observation B8, gamma 0.99: reward design {_CI}"),
    _item("probe_W8", 8, [("dqn_B8_W8_gamma099", "R0"), ("dqn_B8_W8_gamma099_R3", "R3 (closer 0.01)"),
                          ("dqn_B8_W8_gamma099_R3_c01", "R3 (closer 0.1)")], ["bfs"], causes=True,
          title=f"8x8, observation B8, gamma 0.99: the reward-hacking probe {_CI}"),
    _item("summary_W8", 8, _labelled([e.name for e in matrix.experiments(W=8)]), _ALL, curve=False,
          title="8x8: every configuration"),
    _item("summary_W10", 10, _labelled([e.name for e in matrix.experiments(W=10)]), _ALL, curve=False,
          title="10x10: every configuration"),
)

_ITEMS_BY_NAME = {item.name: item for item in ITEMS}


def item(name: str) -> ReportItem:
    """The item called ``name``; ``ValueError`` if there is none."""
    try:
        return _ITEMS_BY_NAME[name]
    except KeyError:
        raise ValueError(f"unknown report item {name!r}; items are {', '.join(_ITEMS_BY_NAME)}") from None


# ------------------------------------------------------------------- building
@dataclass(frozen=True)
class ItemResult:
    """What :func:`build` did for one item."""

    item: ReportItem
    outputs: dict[str, Path]
    used: tuple[str, ...]
    missing: tuple[str, ...]
    seeds: dict[str, int]
    seed_values: dict[str, list[int]]
    git: dict[str, list[str]]

    @property
    def skipped(self) -> bool:
        return not self.used


def _has_finished_run(exp_dir: Path) -> bool:
    return exp_dir.is_dir() and any(
        (sub / CONFIG_FILE).is_file() and (sub / FINAL_FILE).is_file() for sub in exp_dir.iterdir() if sub.is_dir()
    )


def _build_item(item: ReportItem, runs_dir: Path, out_dir: Path, last_k: int | None) -> ItemResult:
    used: list[tuple[str, str, Path, Any]] = []  # (name, label, directory, scan)
    missing: list[str] = []
    for name, label in item.experiments:
        exp_dir = runs_dir / name
        if _has_finished_run(exp_dir):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", AggregateWarning)  # unfinished siblings: counted below, not shouted
                scan = scan_experiment(exp_dir)
            used.append((name, label, exp_dir, scan))
        else:
            missing.append(name)
    if missing:
        warnings.warn(
            f"{item.name}: no finished run for {', '.join(missing)}; left out of the item", AggregateWarning, stacklevel=3
        )
    seeds = {name: len(scan.runs) for name, _, _, scan in used}
    seed_values = {name: [r.seed for r in scan.runs] for name, _, _, scan in used}
    git = {
        name: [str((r.config.get("run") or {}).get("git", "unknown")) for r in scan.runs] for name, _, _, scan in used
    }
    if not used:
        return ItemResult(item, {}, (), tuple(missing), seeds, seed_values, git)

    baselines: list[tuple[str, Path]] = []
    for base in item.baselines:
        b = next(b for b in matrix.BASELINES if b.name == base and b.W == item.W)
        path = runs_dir / "baselines" / Path(b.path).name
        if path.is_file():
            baselines.append((b.label, path))
        else:
            warnings.warn(f"{item.name}: baseline file {path} not found; left out", AggregateWarning, stacklevel=3)

    experiments = [(label, exp_dir) for _, label, exp_dir, _ in used]
    outputs: dict[str, Path] = {}
    if item.table:
        rows = [summarise_baseline(path, label) for label, path in baselines]
        rows += [_summarise_runs(scan.runs, label, last_k, scan.aborted) for _, label, _, scan in used]
        path = out_dir / "tables" / f"{item.name}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(results_table(rows) + "\n", encoding="utf-8", newline="\n")
        outputs["table"] = path
    figures = out_dir / "figures"
    if item.curve:
        outputs["curve"] = plot_learning_curves(experiments, baselines, figures / f"{item.name}_curve.png", title=item.title)
    if item.values:
        outputs["values"] = plot_value_curves(experiments, figures / f"{item.name}_values.png", title=item.title)
    if item.causes:
        outputs["causes"] = plot_death_causes(experiments, figures / f"{item.name}_causes.png", title=item.title)
    return ItemResult(item, outputs, tuple(name for name, _, _, _ in used), tuple(missing), seeds, seed_values, git)


def _relative(path: Path, out_dir: Path) -> str:
    return path.resolve().relative_to(out_dir.resolve()).as_posix()


def _write_index(out_dir: Path, results: Sequence[ItemResult], runs_dir: Path, code_git: str, last_k: int | None) -> Path:
    lines = [
        "# Report",
        "",
        f"Regenerated from `{runs_dir.as_posix()}` by `python -m snake_rl.analysis.report` (code {code_git}; "
        f"tables carry the last-{last_k} column)" if last_k is not None else
        f"Regenerated from `{runs_dir.as_posix()}` by `python -m snake_rl.analysis.report` (code {code_git}; "
        f"no last-k column)",
        "",
    ]
    for res in results:
        lines.append(f"## {res.item.name}")
        lines.append("")
        if res.item.title:
            lines.append(res.item.title)
            lines.append("")
        if res.skipped:
            lines.append("skipped: no finished run")
        else:
            lines.append("outputs: " + ", ".join(f"[{kind}]({_relative(path, out_dir)})" for kind, path in res.outputs.items()))
            lines.append("")
            for name, label in res.item.experiments:
                if name in res.seeds:
                    lines.append(f"- {label}: {name} ({res.seeds[name]} seeds)")
        if res.missing:
            lines.append(f"- missing: {', '.join(res.missing)}")
        lines.append("")
    skipped = [res.item.name for res in results if res.skipped]
    lines += ["## Skipped", "", ", ".join(skipped) if skipped else "none", ""]
    path = out_dir / INDEX_FILE
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path


def _write_manifest(out_dir: Path, results: Sequence[ItemResult], runs_dir: Path, code_git: str, last_k: int | None) -> Path:
    items = []
    for res in results:
        items.append(
            {
                "name": res.item.name,
                "W": res.item.W,
                "title": res.item.title,
                "outputs": {kind: _relative(path, out_dir) for kind, path in res.outputs.items()},
                "experiments": [
                    {"name": name, "label": label, "seeds": res.seed_values.get(name, []), "git": res.git.get(name, [])}
                    for name, label in res.item.experiments
                    if name in res.seeds
                ],
                "missing": list(res.missing),
            }
        )
    manifest = {"runs_dir": runs_dir.as_posix(), "git": code_git, "last_k": last_k, "items": items}
    path = out_dir / MANIFEST_FILE
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8", newline="\n")
    return path


def build(
    runs_dir: str | Path = "runs",
    out_dir: str | Path = "runs/report",
    only: Sequence[str] | None = None,
    last_k: int | None = 5,
    repo_root: str | Path = REPO_ROOT,
) -> list[ItemResult]:
    """Build the items (all, or those named in ``only``) from ``runs_dir``
    into ``out_dir`` and write ``index.md`` and ``manifest.json`` there (both
    list the items built in this call). Returns one :class:`ItemResult` per
    item in ``ITEMS`` order. ``last_k=None``: tables without the last-k column.
    """
    items = list(ITEMS) if only is None else [item(name) for name in only]
    if only is not None:
        order = {it.name: i for i, it in enumerate(ITEMS)}
        items.sort(key=lambda it: order[it.name])
    runs_dir, out_dir = Path(runs_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = [_build_item(it, runs_dir, out_dir, last_k) for it in items]
    code_git = git_hash()
    _write_index(out_dir, results, runs_dir, code_git, last_k)
    _write_manifest(out_dir, results, runs_dir, code_git, last_k)
    return results


# ----------------------------------------------------------------------- CLI
def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Every table and figure of the report from the run directories.")
    p.add_argument("--runs-dir", default="runs", metavar="DIR")
    p.add_argument("--out-dir", default="runs/report", metavar="DIR")
    p.add_argument("--only", nargs="+", default=None, metavar="NAME", help="build only these items")
    p.add_argument("--last-k", type=int, default=5, metavar="K", help="last-k column of the tables (0: none)")
    p.add_argument("--list", action="store_true", help="print the items and their experiments; build nothing")
    args = p.parse_args(argv)
    if args.last_k < 0:
        p.error("--last-k must be >= 0")
    if args.only:
        unknown = [name for name in args.only if name not in _ITEMS_BY_NAME]
        if unknown:
            p.error(f"unknown item(s) {', '.join(unknown)}; --list prints the names")
    if args.list:
        for it in ITEMS:
            print(f"{it.name}: {it.W}x{it.W}, {', '.join(it.kinds())}; " + ", ".join(n for n, _ in it.experiments))
        return 0
    results = build(args.runs_dir, args.out_dir, args.only, args.last_k if args.last_k > 0 else None)
    for res in results:
        print(f"{res.item.name}: " + (", ".join(res.outputs) if res.outputs else "skipped"))
    print(Path(args.out_dir) / INDEX_FILE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

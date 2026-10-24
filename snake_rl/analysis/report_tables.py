"""The tables of the final report (Part E), computed from the run directories.

``docs/PartE-final-report.md`` quotes nine tables in its body and three in the
appendices. Tables 1 (training settings) and 2 (the questions and the answers)
are text; Appendix B is ``runs/report/tables/summary_W{8,10}.md`` as
``snake_rl.analysis.report`` writes it. Every other table is a statistic over
the five seeds of each configuration that the generated tables do not carry
(paired differences on the same seeds, the comparison with BFS on the same
evaluation boards, the overestimation gap, the behaviour statistics, the
tabular sweep), and this module computes them from the run directories:

===========  ============================  ==============================================
table        function                      source files
===========  ============================  ==============================================
Table 3      :func:`table_headline`        ``final_eval.json``, ``metrics.csv``, ``runs/baselines/*.json``
Table 4      :func:`table_representation`  ``final_eval.json``, ``metrics.csv``
Table 5      :func:`table_discount`        plus the value columns (``metrics.csv`` or ``values.csv``)
Table 6      :func:`table_components`      plus ``config.yaml`` (replay, G, target period)
Table 7      :func:`table_refinements`     plus the value columns
Table 8      :func:`table_reward`          ``final_eval.json``, ``metrics.csv``, baselines
Table 9      :func:`table_behaviour`       ``runs/behaviour/<name>/summary.json`` and ``episodes.csv``
Table C1     :func:`table_sweep`           ``runs/sweep_tabular_W8/<alpha, gamma>/<seed>/``
===========  ============================  ==============================================

Each function returns the table as Markdown in the report's wording, row
order and number formatting, so that the results notebook shows the report's
tables and :func:`compare_with_report` can check them against the report
source cell by cell. Statistics: mean over the seeds with a 95 % Student-t
half-width (:func:`snake_rl.analysis.aggregate.mean_ci95`; the seed is the
unit of replication), paired differences from the per-seed differences,
reference players with a normal interval over their 100 episodes.

Command line (repo root)::

    python -m snake_rl.analysis.report_tables [--runs-dir runs] [--out-dir runs/report/tables]
                                              [--report docs/PartE-final-report.md]

writes ``report_table_<n>.md`` per table, skips the tables whose experiments
are missing, and with ``--report`` prints the cells that differ from the
report (exit status 1 if any).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from snake_rl.analysis.aggregate import load_runs, mean_ci95
from snake_rl.evaluate import EvalResult

MINUS = "−"  # the report prints negative numbers with U+2212
DASH = "–"   # en dash: an empty cell of the report
NBSP = " "
CAUSES = ("wall", "self", "starve")
#: Evaluations averaged for the "last-k" score columns and the overestimation gap.
LAST_K_VALUES = 5
#: An evaluation episode "loops" when its post-food tail repeats for this many states (Table 9).
LOOP_STEPS = 20
BASELINE_NAMES = {"Random": "random", "BFS greedy": "bfs", "Hamiltonian cycle": "hamiltonian"}
SWEEP_DIR = "sweep_tabular_W8"
SWEEP_ALPHAS = ("0.02", "0.1", "0.3")
SWEEP_GAMMAS = ("0.8", "0.9", "0.95", "0.99")


class MissingExperiment(FileNotFoundError):
    """An experiment (or behaviour analysis) directory the table needs is not there."""


# ------------------------------------------------------------------- loading
def _flat(node: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(node, dict):
        return {prefix[:-1]: node}
    out: dict[str, Any] = {}
    for key, value in node.items():
        out.update(_flat(value, f"{prefix}{key}."))
    return out


@dataclass(frozen=True)
class SeedStats:
    """What one run contributes: its final evaluation, its evaluation curve, its value columns."""

    seed: int
    episode_seeds: tuple[int, ...]
    episode_scores: tuple[float, ...]
    episode_steps: tuple[float, ...]
    episode_causes: tuple[str, ...]
    curve: tuple[float, ...]
    starve_curve: tuple[float, ...]
    env_steps: tuple[int, ...]
    values: dict[str, tuple[float, ...]] | None
    settings: dict[str, Any]

    @property
    def n_episodes(self) -> int:
        return len(self.episode_scores)

    @property
    def final_mean(self) -> float:
        return float(np.mean(self.episode_scores))

    @property
    def final_mean_steps(self) -> float:
        return float(np.mean(self.episode_steps))

    @property
    def food_per_100(self) -> float:
        return 100.0 * float(np.sum(self.episode_scores)) / float(np.sum(self.episode_steps))

    def frac(self, cause: str) -> float:
        return self.episode_causes.count(cause) / self.n_episodes

    def last_k(self, k: int) -> float:
        return float(np.mean(self.curve[-k:]))

    def value_gap(self, which: str = "visited", k: int = LAST_K_VALUES) -> float:
        """Mean over the last ``k`` evaluations of the value estimate minus the realised return."""
        if self.values is None:
            raise MissingExperiment(f"seed {self.seed}: no value columns")
        return float(np.mean(self.values[f"value_{which}"][-k:]) - np.mean(self.values[f"return_{which}"][-k:]))

    def value_mean(self, which: str, k: int = LAST_K_VALUES) -> tuple[float, float]:
        if self.values is None:
            raise MissingExperiment(f"seed {self.seed}: no value columns")
        return float(np.mean(self.values[f"value_{which}"][-k:])), float(np.mean(self.values[f"return_{which}"][-k:]))


_KEYS: dict[str, Callable[[SeedStats], float]] = {
    "final": lambda s: s.final_mean,
    "last5": lambda s: s.last_k(5),
    "last10": lambda s: s.last_k(10),
    "last20": lambda s: s.last_k(20),
    "steps": lambda s: s.final_mean_steps,
    "food": lambda s: s.food_per_100,
    "wall": lambda s: s.frac("wall"),
    "self": lambda s: s.frac("self"),
    "starve": lambda s: s.frac("starve"),
    "gap_visited": lambda s: s.value_gap("visited"),
    "gap_start": lambda s: s.value_gap("start"),
}


@dataclass(frozen=True)
class Experiment:
    name: str
    seeds: tuple[SeedStats, ...]

    def per_seed(self, key: str) -> list[float]:
        return [_KEYS[key](s) for s in self.seeds]

    @property
    def episodes(self) -> list[tuple[float, float, str]]:
        return [(sc, st, c) for s in self.seeds for sc, st, c in zip(s.episode_scores, s.episode_steps, s.episode_causes)]


def _values_of(run_dir: Path, metrics: pd.DataFrame) -> dict[str, tuple[float, ...]] | None:
    columns = ("value_visited", "return_visited", "value_start", "return_start")
    if "eval_value_visited" in metrics.columns:
        frame = metrics
    elif (run_dir / "values.csv").is_file():
        frame = pd.read_csv(run_dir / "values.csv", float_precision="round_trip")
    else:
        return None
    out = {}
    for name in columns:
        col = frame[f"eval_{name}"].to_numpy(dtype=np.float64)
        out[name] = tuple(float(v) for v in col[~np.isnan(col)])
    return out


@lru_cache(maxsize=None)
def _load(runs_dir: str, name: str) -> Experiment:
    exp_dir = Path(runs_dir) / name
    if not exp_dir.is_dir():
        raise MissingExperiment(f"{exp_dir} is missing")
    seeds = []
    for run in load_runs(exp_dir):
        metrics = run.metrics
        seeds.append(SeedStats(
            seed=run.seed,
            episode_seeds=tuple(int(e.seed) for e in run.final.episodes),
            episode_scores=tuple(float(e.score) for e in run.final.episodes),
            episode_steps=tuple(float(e.steps) for e in run.final.episodes),
            episode_causes=tuple(str(e.cause) for e in run.final.episodes),
            curve=tuple(float(v) for v in metrics["eval_mean_score"]),
            starve_curve=tuple(float(v) for v in metrics["eval_frac_starve"]),
            env_steps=tuple(int(v) for v in metrics["env_steps"]),
            values=_values_of(run.run_dir, metrics),
            settings=_flat(run.config),
        ))
    return Experiment(name=name, seeds=tuple(seeds))


def load_experiment(runs_dir: str | Path, name: str) -> Experiment:
    """The finished runs of ``runs_dir/name`` (cached; sorted by seed)."""
    return _load(str(Path(runs_dir).resolve()), name)


def has_experiment(runs_dir: str | Path, name: str) -> bool:
    return (Path(runs_dir) / name).is_dir()


@lru_cache(maxsize=None)
def _baseline(runs_dir: str, name: str, W: int) -> EvalResult:
    path = Path(runs_dir) / "baselines" / f"{name}_W{W}_relative.json"
    if not path.is_file():
        raise MissingExperiment(f"{path} is missing")
    return EvalResult.load(path)


def load_baseline(runs_dir: str | Path, name: str, W: int) -> EvalResult:
    """``runs/baselines/<name>_W<W>_relative.json`` (random, bfs, hamiltonian)."""
    return _baseline(str(Path(runs_dir).resolve()), name, W)


# ---------------------------------------------------------------- statistics
@dataclass(frozen=True)
class Paired:
    """A paired difference over the seeds: mean, 95 % t interval, how many seeds are positive."""

    mean: float
    lo: float
    hi: float
    npos: int
    n: int

    @property
    def excludes_zero(self) -> bool:
        return self.lo > 0 or self.hi < 0


def _paired(diffs: Sequence[float]) -> Paired:
    m, h = mean_ci95(diffs)
    return Paired(mean=m, lo=m - h, hi=m + h, npos=sum(d > 0 for d in diffs), n=len(diffs))


def paired(a: Experiment, b: Experiment, key: str) -> Paired:
    """``a`` minus ``b`` on the same seeds for the statistic ``key`` (see ``per_seed``)."""
    if [s.seed for s in a.seeds] != [s.seed for s in b.seeds]:
        raise ValueError(f"{a.name} and {b.name} do not have the same seeds")
    return _paired([x - y for x, y in zip(a.per_seed(key), b.per_seed(key))])


def bfs_paired(exp: Experiment, runs_dir: str | Path, W: int) -> Paired:
    """Agent minus BFS on the same evaluation boards: per seed the mean over
    the boards of the difference, then mean and t interval over the seeds."""
    bfs = {int(e.seed): float(e.score) for e in load_baseline(runs_dir, "bfs", W).episodes}
    diffs = []
    for s in exp.seeds:
        if any(es not in bfs for es in s.episode_seeds):
            raise ValueError(f"{exp.name} seed {s.seed}: an evaluation board has no BFS episode")
        diffs.append(float(np.mean([sc - bfs[es] for es, sc in zip(s.episode_seeds, s.episode_scores)])))
    return _paired(diffs)


def value_ratio(exp: Experiment, which: str = "visited", k: int = LAST_K_VALUES) -> float:
    """Mean value estimate over mean realised return (over seeds, last ``k`` evaluations)."""
    pairs = [s.value_mean(which, k) for s in exp.seeds]
    return float(np.mean([v for v, _ in pairs]) / np.mean([r for _, r in pairs]))


# ---------------------------------------------------------------- formatting
def fmt_num(x: float, nd: int = 2) -> str:
    return f"{x:,.{nd}f}".replace("-", MINUS)


def fmt_signed(x: float, nd: int = 2) -> str:
    return f"{x:+,.{nd}f}".replace("-", MINUS)


def fmt_ci(values: Sequence[float], nd: int = 2) -> str:
    m, h = mean_ci95(values)
    return f"{fmt_num(m, nd)} ± {fmt_num(h, nd)}"


def fmt_interval(p: Paired, nd: int = 2) -> str:
    return f"{fmt_signed(p.mean, nd)} [{fmt_signed(p.lo, nd)}, {fmt_signed(p.hi, nd)}]"


def fmt_causes(episodes: Sequence[tuple[float, float, str]]) -> str:
    """``wall / self / starve`` as rounded percentages of the episodes."""
    n = len(episodes)
    return " / ".join(str(round(100 * sum(c == cause for _, _, c in episodes) / n)) for cause in CAUSES)


def markdown_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _ref_ci(result: EvalResult) -> str:
    return f"{fmt_num(result.mean_score)} ± {fmt_num(result.ci95)}"


# ------------------------------------------------------------------ Table 3
HEADLINE_ROWS: dict[int, list[tuple[str, str]]] = {
    10: [("Random", "random"), ("BFS greedy", "bfs"), ("Hamiltonian cycle", "hamiltonian"),
         ("Tabular Q-learning (γ = 0.9)", "tabular_q_W10"),
         ("Tabular Q-learning, γ = 0 (control)", "tabular_q_W10_gamma0"),
         ("DQN, observation A (MLP)", "dqn_A_W10"), ("DQN, observation B8 (CNN)", "dqn_B8_W10"),
         ("DQN, B8, G = 1, target period 31 (`G1_t31`)", "dqn_B8_W10_G1_t31"),
         ("DQN, B8, γ = 0.99", "dqn_B8_W10_gamma099")],
    8: [("Random", "random"), ("BFS greedy", "bfs"), ("Hamiltonian cycle", "hamiltonian"),
        ("Tabular Q-learning (γ = 0.9)", "tabular_q_W8"),
        ("Tabular Q-learning, γ = 0 (control)", "tabular_q_W8_gamma0"),
        ("DQN, observation A (MLP)", "dqn_A_W8"), ("DQN, observation B8 (CNN)", "dqn_B8_W8"),
        ("DQN, B8, γ = 0.99", "dqn_B8_W8_gamma099"),
        ("DQN, B8, γ = 0.99, R3 with bonus 0.1", "dqn_B8_W8_gamma099_R3_c01")],
}
HEADLINE_HEADER = ["Player", "Score, final", "Score, last-5", "Steps per episode", "Food per 100 steps",
                   "Wall / self / starve (%)", f"Agent {MINUS} BFS, same boards"]


def _headline_reference(runs_dir: str | Path, name: str, W: int) -> list[str]:
    result = load_baseline(runs_dir, name, W)
    episodes = [(float(e.score), float(e.steps), e.cause) for e in result.episodes]
    steps = float(np.mean([st for _, st, _ in episodes]))
    food = 100 * sum(sc for sc, _, _ in episodes) / sum(st for _, st, _ in episodes)
    if name == "hamiltonian":
        score = f"{result.mean_score:.0f} (fills the board)"
    else:
        score = _ref_ci(result)
    return [score, DASH, f"{steps:,.1f}", DASH if name == "random" else fmt_num(food), fmt_causes(episodes), DASH]


def _headline_learner(runs_dir: str | Path, name: str, W: int) -> list[str]:
    exp = load_experiment(runs_dir, name)
    control = name.endswith("_gamma0")  # the γ = 0 control: every seed converges to (almost) the same policy
    episodes = exp.episodes
    bfs = bfs_paired(exp, runs_dir, W)
    vs = fmt_interval(bfs) if (bfs.hi - bfs.lo) / 2 > 0.005 else fmt_signed(bfs.mean)
    dagger = " †" if control else ""
    return [fmt_ci(exp.per_seed("final")) + dagger, fmt_ci(exp.per_seed("last5")) + dagger,
            f"{np.mean([st for _, st, _ in episodes]):.1f}", fmt_num(float(np.mean(exp.per_seed("food")))),
            fmt_causes(episodes), vs + dagger]


def table_headline(runs_dir: str | Path = "runs") -> str:
    """Table 3: every headline configuration between the reference players, both boards."""
    rows: list[list[str]] = []
    for W in (10, 8):
        rows.append([f"**{W} × {W} board**"] + [""] * (len(HEADLINE_HEADER) - 1))
        for label, name in HEADLINE_ROWS[W]:
            cells = _headline_reference(runs_dir, name, W) if name in BASELINE_NAMES.values() else _headline_learner(runs_dir, name, W)
            rows.append([label] + cells)
    return markdown_table(HEADLINE_HEADER, rows)


# ------------------------------------------------------------------ Table 4
REPRESENTATION_ROWS = [("A: 11 features, MLP", "dqn_A"), ("B3: head, body, food", "dqn_B3"),
                       ("B7: B8 without the age plane", "dqn_B7"), ("B8: head, body, food, age, heading × 4", "dqn_B8")]


def table_representation(runs_dir: str | Path = "runs") -> str:
    """Table 4: the representation axis at γ = 0.9, paired against B8 on the same seeds."""
    header = ["Observation"]
    for W in (8, 10):
        header += [f"{W} × {W}, final", f"{W} × {W}, last-5", f"{W} × {W}, minus B8"]
    rows = []
    for label, prefix in REPRESENTATION_ROWS:
        row = [label]
        for W in (8, 10):
            exp = load_experiment(runs_dir, f"{prefix}_W{W}")
            base = load_experiment(runs_dir, f"dqn_B8_W{W}")
            row += [fmt_ci(exp.per_seed("final")), fmt_ci(exp.per_seed("last5"))]
            if prefix == "dqn_B8":
                row.append(DASH)
            else:
                row.append(f"{fmt_interval(paired(exp, base, 'final'))} / {fmt_interval(paired(exp, base, 'last5'))}")
        rows.append(row)
    return markdown_table(header, rows)


# ------------------------------------------------------------------ Table 5
def table_discount(runs_dir: str | Path = "runs") -> str:
    """Table 5: the discount axis, γ = 0.99 minus γ = 0.9 on the same seeds."""
    header = [""]
    for W in (8, 10):
        header += [f"{W} × {W}, γ = 0.9", f"{W} × {W}, γ = 0.99", f"{W} × {W}, paired"]
    b = {W: (load_experiment(runs_dir, f"dqn_B8_W{W}"), load_experiment(runs_dir, f"dqn_B8_W{W}_gamma099")) for W in (8, 10)}

    def each(cell: Callable[[Experiment, Experiment, int], list[str]]) -> list[str]:
        return [c for W in (8, 10) for c in cell(b[W][0], b[W][1], W)]

    rows = []
    a_row = ["A (MLP), final"]
    for W in (8, 10):
        low = load_experiment(runs_dir, f"dqn_A_W{W}")
        if has_experiment(runs_dir, f"dqn_A_W{W}_gamma099"):
            high = load_experiment(runs_dir, f"dqn_A_W{W}_gamma099")
            a_row += [fmt_ci(low.per_seed("final")), fmt_ci(high.per_seed("final")), fmt_interval(paired(high, low, "final"))]
        else:
            a_row += [fmt_ci(low.per_seed("final")), "not run", DASH]
    rows.append(a_row)
    rows.append(["B8 (CNN), final"] + each(lambda lo, hi, W: [
        fmt_ci(lo.per_seed("final")), fmt_ci(hi.per_seed("final")),
        f"{fmt_interval(p := paired(hi, lo, 'final'))} ({p.npos}/{p.n})"]))
    rows.append(["B8, last-5"] + each(lambda lo, hi, W: [
        fmt_ci(lo.per_seed("last5")), fmt_ci(hi.per_seed("last5")), fmt_interval(paired(hi, lo, "last5"))]))
    rows.append(["B8, steps per episode"] + each(lambda lo, hi, W: [
        f"{np.mean(lo.per_seed('steps')):.1f}", f"{np.mean(hi.per_seed('steps')):.1f}",
        fmt_interval(paired(hi, lo, "steps"), 0) if W == 10 else ""]))
    rows.append(["B8, food per 100 steps"] + each(lambda lo, hi, W: [
        fmt_ci(lo.per_seed("food")), fmt_ci(hi.per_seed("food")),
        fmt_interval(paired(hi, lo, "food")) if W == 10 else ""]))
    rows.append(["B8, starvation rate"] + each(lambda lo, hi, W: [
        fmt_num(float(np.mean(lo.per_seed("starve")))), fmt_num(float(np.mean(hi.per_seed("starve")))),
        fmt_interval(paired(hi, lo, "starve")) if W == 10 else ""]))
    rows.append(["B8, overestimation on visited states (ratio)"] + each(lambda lo, hi, W: [
        f"{fmt_signed(float(np.mean(lo.per_seed('gap_visited'))))} ({fmt_num(value_ratio(lo))})",
        f"{fmt_signed(float(np.mean(hi.per_seed('gap_visited'))))} ({fmt_num(value_ratio(hi))})", ""]))
    rows.append(["B8, minus BFS on the same boards"] + each(lambda lo, hi, W: [
        fmt_interval(bfs_paired(lo, runs_dir, W)), fmt_interval(bfs_paired(hi, runs_dir, W)), ""]))
    return markdown_table(header, rows)


# ------------------------------------------------------------------ Table 6
COMPONENT_ROWS = [
    ("B8, full", "dqn_B8_W8", "dqn_B8_W10"),
    ("no target network", "dqn_B8_W8_notarget", "dqn_B8_W10_notarget"),
    ("no replay", "dqn_B8_W8_noreplay", "dqn_B8_W10_noreplay"),
    ("no replay, no target network", "dqn_B8_W8_noreplay_notarget", "dqn_B8_W10_noreplay_notarget"),
    ("no replay, target period matched (`t31`)", "dqn_B8_W8_noreplay_t31", "dqn_B8_W10_noreplay_t31"),
    ("replay, G = 1 (`G1`)", "dqn_B8_W8_G1", "dqn_B8_W10_G1"),
    ("replay, G = 1, target period matched (`G1_t31`)", "dqn_B8_W8_G1_t31", "dqn_B8_W10_G1_t31"),
]


def _component_settings(exp: Experiment) -> tuple[str, int, str]:
    """(replay yes/no, G, target period with its equivalent in environment steps) from ``config.yaml``."""
    s = exp.seeds[0].settings
    for key in ("agent.replay", "agent.target_update_every", "train.n_envs"):
        if key not in s:
            raise MissingExperiment(f"{exp.name}: config.yaml has no {key} (not a DQN run)")
    replay = bool(s["agent.replay"])
    n_envs = int(s["train.n_envs"])
    if replay:
        g = s.get("agent.grad_steps_per_vec_step")
        g = max(1, n_envs // 4) if g is None else int(g)
    else:
        g = 1  # one update per vectorised step on that step's own transitions
    period = int(s["agent.target_update_every"])
    if period == 1:
        text = "1"
    else:
        env_steps = period * n_envs / g
        exact = g == 1 and env_steps % 1000 == 0
        text = f"{period:,} ({'=' if exact else '≈'} {round(env_steps / 1000)}k env steps)"
    return "yes" if replay else "no", g, text


def table_components(runs_dir: str | Path = "runs") -> str:
    """Table 6: replay, target network and the number of gradient steps on observation B8."""
    header = ["Configuration", "Replay", "G", "Target period", "8 × 8, final", "8 × 8, last-5", "10 × 10, final", "10 × 10, last-5"]
    rows = []
    for label, name8, name10 in COMPONENT_ROWS:
        exp8 = load_experiment(runs_dir, name8)
        replay, g, period = _component_settings(exp8)
        row = [label, replay, str(g), period, fmt_ci(exp8.per_seed("final")), fmt_ci(exp8.per_seed("last5"))]
        if has_experiment(runs_dir, name10):
            exp10 = load_experiment(runs_dir, name10)
            row += [fmt_ci(exp10.per_seed("final")), fmt_ci(exp10.per_seed("last5"))]
        else:
            row += ["not run", DASH]
        rows.append(row)
    return markdown_table(header, rows)


# ------------------------------------------------------------------ Table 7
REFINEMENT_ROWS = [
    ("A, γ = 0.9 → Double", "dqn_A_W8", "dqn_A_W8_double"),
    ("A, γ = 0.99 → Double", "dqn_A_W8_gamma099", "dqn_A_W8_gamma099_double"),
    ("A, γ = 0.9 → Dueling", "dqn_A_W8", "dqn_A_W8_dueling"),
    ("A, γ = 0.9 → PER", "dqn_A_W8", "dqn_A_W8_per"),
    ("B8, γ = 0.9 → Double", "dqn_B8_W8", "dqn_B8_W8_double"),
    ("B8, γ = 0.99 → Double", "dqn_B8_W8_gamma099", "dqn_B8_W8_gamma099_double"),
    ("B8, γ = 0.9 → Dueling", "dqn_B8_W8", "dqn_B8_W8_dueling"),
    ("B8, γ = 0.9 → PER", "dqn_B8_W8", "dqn_B8_W8_per"),
]


def table_refinements(runs_dir: str | Path = "runs") -> str:
    """Table 7: Double DQN, Dueling and PER on 8 × 8 with the overestimation gap of each."""
    header = ["Baseline → variant", "Baseline score", "Variant score", "Score difference: final / last-5 / last-20",
              "Overestimation gap, visited states", "Start-state gap"]
    rows = []
    for label, base_name, var_name in REFINEMENT_ROWS:
        base, var = load_experiment(runs_dir, base_name), load_experiment(runs_dir, var_name)
        score = " / ".join(fmt_interval(paired(var, base, key)) for key in ("final", "last5", "last20"))
        gap_b, gap_v = (float(np.mean(e.per_seed("gap_visited"))) for e in (base, var))
        gap = f"{fmt_num(gap_b)} → {fmt_num(gap_v)}"
        p = paired(var, base, "gap_visited")
        if p.excludes_zero:
            gap += f" ({fmt_interval(p)})"
        start_b, start_v = (float(np.mean(e.per_seed("gap_start"))) for e in (base, var))
        rows.append([label, fmt_ci(base.per_seed("final")), fmt_ci(var.per_seed("final")), score, gap,
                     f"{fmt_num(start_b)} → {fmt_num(start_v)}"])
    return markdown_table(header, rows)


# ------------------------------------------------------------------ Table 8
REWARD_COLUMNS: list[tuple[str, str, int, tuple[str, ...]]] = [
    ("A, γ = 0.9", "dqn_A_W8", 8, ("final", "last10")),
    ("A, γ = 0.99", "dqn_A_W8_gamma099", 8, ("final", "last10")),
    ("B8, γ = 0.9", "dqn_B8_W8", 8, ("final", "last10")),
    ("B8, γ = 0.99", "dqn_B8_W8_gamma099", 8, ("final", "last10")),
    ("B8, γ = 0.99, 10 × 10", "dqn_B8_W10_gamma099", 10, ("last5",)),
]
REWARD_ROWS = [("R0, sparse", ""), ("R1, step penalty", "_R1"), ("R2, potential-based shaping", "_R2"),
               ("R3, naive distance shaping", "_R3"), ("R3 symmetric (± 0.01)", "_R3sym"),
               ("R3, starvation limit 1,000", "_R3_H1000"), ("R3, bonus 0.1 (probe)", "_R3_c01")]


def table_reward(runs_dir: str | Path = "runs") -> str:
    """Table 8: the reward axis, final scores and paired differences against R0."""
    header = ["Reward"] + [title for title, _, _, _ in REWARD_COLUMNS]
    rows = []
    for label, suffix in REWARD_ROWS:
        row = [label]
        for _, base_name, _, _ in REWARD_COLUMNS:
            name = base_name + suffix
            row.append(fmt_ci(load_experiment(runs_dir, name).per_seed("final")) if has_experiment(runs_dir, name) else DASH)
        rows.append(row)
    bfs_row, seen = ["BFS greedy, same boards"], set()
    for _, _, W, _ in REWARD_COLUMNS:
        bfs_row.append("" if W in seen else _ref_ci(load_baseline(runs_dir, "bfs", W)))
        seen.add(W)
    rows.append(bfs_row)
    for label, suffix in REWARD_ROWS[1:4]:
        row = [f"{label.split(',')[0]} {MINUS} R0"]
        for _, base_name, _, windows in REWARD_COLUMNS:
            base, var = load_experiment(runs_dir, base_name), load_experiment(runs_dir, base_name + suffix)
            row.append(" / ".join(fmt_interval(paired(var, base, key)) for key in windows))
        rows.append(row)
    return markdown_table(header, rows)


# ------------------------------------------------------------------ Table 9
BEHAVIOUR_ROWS = [
    ("BFS greedy", "bfs_W10"), ("Hamiltonian cycle", "hamiltonian_W10"), ("B8, γ = 0.9", "dqn_B8_W10_final"),
    ("B8, γ = 0.99, 51k steps", "dqn_B8_W10_gamma099_early"), ("B8, γ = 0.99, 256k steps", "dqn_B8_W10_gamma099_mid"),
    ("B8, γ = 0.99, final", "dqn_B8_W10_gamma099_final"), ("B8, γ = 0.99, R1, final", "dqn_B8_W10_gamma099_R1_final"),
    ("B8, γ = 0.99, R2, final", "dqn_B8_W10_gamma099_R2_final"), ("B8, γ = 0.99, R3, final", "dqn_B8_W10_gamma099_R3_final"),
]
BEHAVIOUR_HEADER = ["Player", "Score", "Steps", "Self / wall / starve", "Border share", "Looping episodes",
                    "Q estimate: loops / others", "Tail after last food (steps)"]


def _behaviour_episodes(analysis_dir: Path) -> list[dict[str, str]]:
    """The ``episodes.csv`` rows of a behaviour analysis (every seed of an experiment, or the one file of a player)."""
    files = sorted(analysis_dir.glob("*/episodes.csv"), key=lambda p: int(p.parent.name)) or [analysis_dir / "episodes.csv"]
    rows: list[dict[str, str]] = []
    for path in files:
        if not path.is_file():
            raise MissingExperiment(f"{path} is missing")
        with path.open(encoding="utf-8", newline="") as f:
            rows += list(csv.DictReader(f))
    return rows


def _fmt_starve(x: float) -> str:
    if x == 0:
        return "0"
    return fmt_num(x, 3 if x < 0.1 else 2)


def _loop_cells(rows: Sequence[dict[str, str]], frac: float, half: float | None) -> tuple[str, str]:
    """(looping episodes, Q estimate loops / others) of Table 9."""
    loops = [r for r in rows if int(r["loop_steps"]) >= LOOP_STEPS]
    if frac == 0:
        loop_cell = "0"
    elif frac >= 0.1:
        loop_cell = f"{fmt_num(frac)} ± {fmt_num(half or 0.0)}"
    else:
        loop_cell = fmt_num(frac, 3)
    if loops:
        loop_cell += f" ({len(loops)})"
    if not loops:
        return loop_cell, DASH
    q = lambda rs: float(np.mean([float(r["q_max_mean"]) for r in rs]))  # noqa: E731
    others = [r for r in rows if int(r["loop_steps"]) < LOOP_STEPS]
    return loop_cell, f"{fmt_num(q(loops))} / {fmt_num(q(others))}"


def _behaviour_row(analysis_dir: Path) -> list[str]:
    summary = json.loads((analysis_dir / "summary.json").read_text(encoding="utf-8"))
    rows = _behaviour_episodes(analysis_dir)
    if "score_mean_mean" in summary:  # an experiment: mean over its seeds ± t interval
        ci = lambda key, nd=2: f"{fmt_num(summary[key + '_mean'], nd)} ± {fmt_num(summary[key + '_ci95'], nd)}"  # noqa: E731
        causes = f"{fmt_num(summary['frac_self_mean'])} / {fmt_num(summary['frac_wall_mean'])} / {_fmt_starve(summary['frac_starve_mean'])}"
        loops, q = _loop_cells(rows, summary["loop_fraction_mean"], summary["loop_fraction_ci95"])
        return [ci("score_mean"), ci("steps_mean", 1), causes, ci("border_fraction_mean"), loops, q, ci("tail_without_food_mean", 1)]
    if summary["frac_win"] == 1:
        score, causes = f"{summary['score_mean']:.0f} (all won)", DASH
    else:
        score = fmt_num(summary["score_mean"])
        causes = f"{fmt_num(summary['frac_self'])} / {fmt_num(summary['frac_wall'])} / {_fmt_starve(summary['frac_starve'])}"
    loops, q = _loop_cells(rows, summary["loop_fraction"], None)
    tail = summary["tail_without_food_mean"]
    return [score, f"{summary['steps_mean']:,.1f}", causes, fmt_num(summary["border_fraction_mean"]), loops, q,
            "0" if tail == 0 else fmt_num(tail, 1)]


def table_behaviour(runs_dir: str | Path = "runs") -> str:
    """Table 9: the behaviour analyses of the final policies on 10 × 10 (``runs/behaviour``)."""
    behaviour_dir = Path(runs_dir) / "behaviour"
    rows = []
    for label, name in BEHAVIOUR_ROWS:
        if not (behaviour_dir / name / "summary.json").is_file():
            raise MissingExperiment(f"{behaviour_dir / name} is missing")
        rows.append([label] + _behaviour_row(behaviour_dir / name))
    return markdown_table(BEHAVIOUR_HEADER, rows)


# ----------------------------------------------------------------- Table C1
def starving_snapshots(exp: Experiment) -> int:
    """Evaluations (after the one before training) in which at least one greedy episode starved."""
    return sum(sum(1 for step, frac in zip(s.env_steps, s.starve_curve) if step > 0 and frac > 0) for s in exp.seeds)


def table_sweep(runs_dir: str | Path = "runs") -> str:
    """Table C1: the tabular α × γ sweep on 8 × 8, with the number of starving evaluation snapshots."""
    sweep = Path(runs_dir) / SWEEP_DIR
    rows = []
    for alpha in SWEEP_ALPHAS:
        row = [alpha]
        for gamma in SWEEP_GAMMAS:
            exp = load_experiment(sweep, f"tabular_q_W8__alpha={alpha}__gamma={gamma}")
            row.append(f"{fmt_ci(exp.per_seed('final'))} ({starving_snapshots(exp)})")
        rows.append(row)
    return markdown_table(["α \\ γ"] + list(SWEEP_GAMMAS), rows)


# ---------------------------------------------------------------- Appendix B
APPENDIX_COLUMNS = [("label", "Configuration"), ("score mean +- 95% CI", "Score, final"),
                    ("last-5 evals mean +- 95% CI", "Score, last-5"), ("mean steps", "Steps"),
                    ("wall", "Wall"), ("self", "Self"), ("starve", "Starve")]


def appendix_table(summary_md: str | Path) -> str:
    """Appendix B: the columns of the report from ``runs/report/tables/summary_W<W>.md``."""
    lines = [l for l in Path(summary_md).read_text(encoding="utf-8").splitlines() if l.strip().startswith("|")]
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines]
    index = [rows[0].index(name) for name, _ in APPENDIX_COLUMNS]
    body = [[r[i].replace("+-", "±") for i in index] for r in rows[2:]]
    return markdown_table([title for _, title in APPENDIX_COLUMNS], body)


# ------------------------------------------------------------------ registry
TABLES: dict[str, Callable[[str | Path], str]] = {
    "Table 3": table_headline,
    "Table 4": table_representation,
    "Table 5": table_discount,
    "Table 6": table_components,
    "Table 7": table_refinements,
    "Table 8": table_reward,
    "Table 9": table_behaviour,
    "Table C1": table_sweep,
}


def build(runs_dir: str | Path = "runs") -> tuple[dict[str, str], dict[str, str]]:
    """Every table of :data:`TABLES` that the run directories allow: ``(tables, skipped)``,
    both keyed by the table name; ``skipped`` holds the reason."""
    tables: dict[str, str] = {}
    skipped: dict[str, str] = {}
    for name, fn in TABLES.items():
        try:
            tables[name] = fn(runs_dir)
        except MissingExperiment as err:
            skipped[name] = str(err)
    return tables, skipped


# ------------------------------------------------------------ the report file
def parse_table(markdown: str) -> list[list[str]]:
    """The cells of a Markdown pipe table (header row, then the body; the ``---`` row dropped)."""
    rows = []
    for line in markdown.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip().replace(NBSP, " ") for c in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-+:?", c) for c in cells if c):
            continue
        rows.append(cells)
    return rows


def report_tables(report_md: str | Path) -> dict[str, list[list[str]]]:
    """The pipe tables of the report source keyed by the caption before them (``"Table 3"``)."""
    lines = Path(report_md).read_text(encoding="utf-8").splitlines()
    tables: dict[str, list[list[str]]] = {}
    caption = None
    i = 0
    while i < len(lines):
        m = re.match(r"^(Table [A-Z]?\d+)\.", lines[i])
        if m:
            caption = m.group(1)
        if lines[i].startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            if caption:
                tables[caption] = parse_table("\n".join(block))
            caption = None
            continue
        i += 1
    return tables


def report_captions(report_md: str | Path) -> dict[str, str]:
    """Caption text of every ``Table N.`` and ``Figure N.`` line of the report source."""
    out = {}
    for line in Path(report_md).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^((?:Table|Figure) [A-Z]?\d+)\.\s*(.*)$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def report_figures(report_md: str | Path) -> dict[str, list[str]]:
    """The files of every figure of the report source, keyed like the captions (``"Figure 2"``)."""
    lines = Path(report_md).read_text(encoding="utf-8").splitlines()
    out = {}
    for k, line in enumerate(lines):
        if line.startswith("@figure"):
            files = [s.strip() for s in line.split(" ", 1)[1].split("|")]
            m = re.match(r"^(Figure \d+)\.", lines[k + 1])
            if m:
                out[m.group(1)] = files
    return out


def compare_with_report(name: str, markdown: str, report_md: str | Path) -> list[str]:
    """The cells in which the computed table differs from the report's; empty when identical."""
    want = report_tables(report_md).get(name)
    if want is None:
        return [f"{name} is not in {report_md}"]
    got = parse_table(markdown)
    diffs = []
    if len(got) != len(want):
        diffs.append(f"{name}: {len(got)} rows computed, {len(want)} in the report")
    for i, (g, w) in enumerate(zip(got, want)):
        if len(g) != len(w):
            diffs.append(f"{name} row {i}: {len(g)} cells computed, {len(w)} in the report")
        for j, (a, b) in enumerate(zip(g, w)):
            if a != b:
                diffs.append(f"{name} row {i} col {j}: computed {a!r}, report {b!r}")
    return diffs


# ------------------------------------------------------------------------ CLI
def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="The tables of the final report, computed from the run directories.")
    p.add_argument("--runs-dir", default="runs", metavar="DIR")
    p.add_argument("--out-dir", default="runs/report/tables", metavar="DIR", help="writes report_table_<n>.md per table")
    p.add_argument("--report", default=None, metavar="MD", help="compare every table with this report source")
    args = p.parse_args(argv)
    if not Path(args.runs_dir).is_dir():
        print(f"error: {args.runs_dir} is not a directory", file=sys.stderr)
        return 2
    tables, skipped = build(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, markdown in tables.items():
        path = out_dir / f"report_table_{name.split()[1]}.md"
        path.write_text(markdown + "\n", encoding="utf-8")
        print(f"{name}: written {path}")
    for name, reason in skipped.items():
        print(f"{name}: skipped ({reason})")
    bad = 0
    if args.report:
        for name, markdown in tables.items():
            diffs = compare_with_report(name, markdown, args.report)
            bad += len(diffs)
            print(f"{name}: {'identical to the report' if not diffs else str(len(diffs)) + ' differences'}")
            for d in diffs:
                print("  " + d)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

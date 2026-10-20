"""Cross-seed aggregation: results tables and learning curves regenerated from
run directories (PLAN.md section 4; project rule "5 seeds + confidence
intervals, no number typed by hand").

A run directory ``<runs_dir>/<exp>/<seed>/`` is written by ``snake_rl.train``:

* ``config.yaml``      the experiment configuration plus a ``run`` block;
                       ``run.seed`` is the training seed.
* ``metrics.csv``      the evaluation curve, one row per evaluation; used
                       here: ``env_steps`` and ``eval_mean_score``.
* ``final_eval.json``  ``EvalResult.to_dict()`` of the last evaluation.

A run is *finished* when ``config.yaml`` and ``final_eval.json`` both exist.
A run with ``aborted.json`` (a DQN update produced a non-finite loss) is
*aborted*; one with only ``config.yaml`` or only ``final_eval.json`` is
*unfinished* (still running, or interrupted). Neither enters a table or a
curve; each kind is reported by its own :class:`AggregateWarning`, and the
count of aborted runs is carried into the summary (``n_aborted``) and the
table (the ``seeds`` cell), so a configuration whose seeds diverged is not
represented by its survivors alone. Extra columns in ``metrics.csv`` and
extra keys in ``config.yaml`` are ignored (DQN runs log more than tabular
runs).

One table carries two different confidence intervals; the column "CI over"
says which one a row has:

``seeds``     Learning agents. The unit of replication is the training seed:
              every seed contributes ONE number, its mean score over the
              final evaluation episodes, and the interval is
              ``mean +- t(n - 1) * sd / sqrt(n)`` over those ``n`` per-seed
              means (Student t, ``ddof = 1``; with 5 seeds t = 2.776, not
              1.96). Pooling the ``n * episodes`` episodes instead would treat
              the episodes of one trained agent as independent draws and hide
              the seed-to-seed variance, which is the variance that matters
              when two training methods are compared.
``episodes``  Non-learning baselines. There is no training seed, so the
              interval is the episode-level normal approximation of the single
              evaluation (``EvalResult.ci95``). It measures how well 100
              boards pin down the mean of one fixed player, nothing more.

Median, max, mean steps and the end-cause fractions describe the distribution
of episodes, so they are computed over all episodes pooled across seeds (every
episode has the same weight, also when seeds have different episode counts).
All episode statistics are recomputed from the per-episode records; the
``summary`` block stored in the JSON files is never read.

Last-k summary (optional, ``last_k`` / ``--last-k K``). The final evaluation
is ONE checkpoint per seed, and the greedy policy of a DQN still moves from
one evaluation to the next at the end of training. With ``last_k`` every seed
additionally contributes the mean of ``eval_mean_score`` over the LAST ``k``
rows of its ``metrics.csv`` (file order), and the table gains a column with
the mean and the Student-t interval over those per-seed numbers. The seed is
still the unit of replication: averaging k evaluations of one seed reduces
the checkpoint-to-checkpoint noise inside that seed, it does not create k
replicates. Baselines have no evaluation curve; their cell is ``n/a``.

Value curve (M7, :func:`value_curve`). For DQN runs the metrics also carry,
per evaluation, the network's value estimate ``max_a Q`` on the states the
greedy policy visits and the discounted return it then realises from them
(``snake_rl.evaluate``); runs trained before M7 get them in ``values.csv``
from ``scripts/backfill_values.py``. The curve gives, across seeds, the
estimate, the return and their per-seed difference (the overestimation of
the Double DQN study), with the same seed-level Student-t intervals.

CLI::

    python -m snake_rl.analysis.aggregate [--baseline FILE ...] [--exp DIR ...] [--out FILE.md] [--curves-dir DIR]
                                          [--last-k K]

prints the results table (baselines first, in the order given, then
experiments), ``--out`` also writes it to a file and ``--curves-dir`` writes
``<exp name>_curve.csv`` per experiment, e.g. ``--baseline
runs/baselines/bfs_W10_relative.json --exp runs/tabular_q_W10 --out
runs/tables/part_c.md --curves-dir runs/tables``. Without ``--last-k`` the
table has no last-k column.
"""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml
from numpy.typing import ArrayLike

from snake_rl.env.common import CAUSE_NAMES
from snake_rl.evaluate import EvalResult

CONFIG_FILE = "config.yaml"
FINAL_FILE = "final_eval.json"
ABORTED_FILE = "aborted.json"
METRICS_FILE = "metrics.csv"

#: End causes reported in tables, in column order.
CAUSES = tuple(c for c in CAUSE_NAMES if c != "none")

#: Columns of :func:`learning_curve`.
CURVE_COLUMNS = ("env_steps", "n_seeds", "mean", "ci95", "min", "max")
#: ``value_curve`` (M7): value estimate, realised return and their gap across seeds.
VALUE_CURVE_COLUMNS = (
    "env_steps", "n_seeds", "value_mean", "value_ci95", "return_mean", "return_ci95", "gap_mean", "gap_ci95",
)
#: Value estimates of a run trained before M7, written by ``scripts/backfill_values.py``.
VALUES_FILE = "values.csv"

#: Two-sided 95 % Student t critical values t(0.975, df) for df = 1 .. 30.
_T_95 = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)


class AggregateWarning(UserWarning):
    """Something was left out of an aggregate (unfinished run, unmatched curve point)."""


# ------------------------------------------------------------------ statistics
def t_critical_95(df: int) -> float:
    """Two-sided 95 % critical value of Student's t with ``df`` degrees of freedom.

    Table lookup for ``df`` = 1 .. 30 (no scipy dependency); 1.96 (the normal
    value) for ``df`` > 30, where the exact value is below 2.04.
    """
    if isinstance(df, bool) or not isinstance(df, (int, np.integer)) or df < 1:
        raise ValueError(f"df must be an integer >= 1, got {df!r}")
    return _T_95[df - 1] if df <= len(_T_95) else 1.96


def _t_half_width(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """``t(n - 1) * sd(ddof=1) / sqrt(n)`` along ``axis``; nan where ``n == 1``."""
    n = x.shape[axis]
    if n < 2:
        return np.full(x.shape[:axis] + x.shape[axis + 1 :], np.nan)
    return t_critical_95(n - 1) * x.std(axis=axis, ddof=1) / np.sqrt(n)


def mean_ci95(values: ArrayLike) -> tuple[float, float]:
    """Mean and half-width of its 95 % Student-t confidence interval.

    ``values`` are the independent replicates (one number per seed). The
    half-width is nan for a single value: one replicate has no spread.
    """
    x = np.asarray(values, dtype=np.float64).ravel()
    if x.size == 0:
        raise ValueError("mean_ci95 needs at least one value")
    return float(x.mean()), float(_t_half_width(x))


# ------------------------------------------------------------------- loading
@dataclass
class RunResult:
    """One finished run directory."""

    run_dir: Path
    seed: int
    config: dict[str, Any]
    final: EvalResult
    metrics: pd.DataFrame


def _load_run(run_dir: Path) -> RunResult:
    config = yaml.safe_load((run_dir / CONFIG_FILE).read_text(encoding="utf-8"))
    run = config.get("run") if isinstance(config, dict) else None
    seed = run.get("seed") if isinstance(run, dict) else None
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"{run_dir / CONFIG_FILE}: run.seed must be an integer, got {seed!r}")
    final = EvalResult.load(run_dir / FINAL_FILE)
    if not final.episodes:
        raise ValueError(f"{run_dir / FINAL_FILE}: no episodes")
    metrics_path = run_dir / METRICS_FILE
    # round_trip: the training loop writes floats with repr; the default (fast)
    # parser of pandas can be off by one unit in the last place, and then the
    # last curve point would not equal the mean score of final_eval.json.
    metrics = pd.read_csv(metrics_path, float_precision="round_trip") if metrics_path.is_file() else pd.DataFrame()
    return RunResult(run_dir=run_dir, seed=seed, config=config, final=final, metrics=metrics)


@dataclass(frozen=True)
class ExperimentScan:
    """What one experiment directory holds: finished runs (sorted by seed),
    and the names of the aborted and the unfinished run directories."""

    runs: list[RunResult]
    aborted: list[str]
    unfinished: list[str]


def scan_experiment(exp_dir: str | Path, stacklevel: int = 2) -> ExperimentScan:
    """Classify the sub-directories of one experiment directory and load the
    finished runs.

    Finished: ``config.yaml`` and ``final_eval.json``. Aborted: ``aborted.json``
    (whatever else is there, except ``final_eval.json``). Unfinished: only one
    of ``config.yaml`` / ``final_eval.json``. Sub-directories with none of
    these files are not runs and are ignored. Aborted and unfinished runs are
    named in one :class:`AggregateWarning` each (issued with ``stacklevel``),
    then ``ValueError`` if there is no finished run or if two finished runs
    claim the same seed (a copied directory must not count as a new
    replicate), or if the finished runs are not one configuration.
    """
    exp_dir = Path(exp_dir)
    if not exp_dir.is_dir():
        raise ValueError(f"{exp_dir} is not a directory")
    runs: list[RunResult] = []
    aborted: list[str] = []
    unfinished: list[str] = []
    for sub in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
        has_config = (sub / CONFIG_FILE).is_file()
        has_final = (sub / FINAL_FILE).is_file()
        if has_config and has_final:
            runs.append(_load_run(sub))
        elif (sub / ABORTED_FILE).is_file():
            aborted.append(sub.name)
        elif has_config or has_final:
            unfinished.append(sub.name)
    if aborted:
        warnings.warn(
            f"{exp_dir}: skipped {len(aborted)} aborted run(s) ({ABORTED_FILE}): {', '.join(aborted)}",
            AggregateWarning,
            stacklevel=stacklevel,
        )
    if unfinished:
        warnings.warn(
            f"{exp_dir}: skipped {len(unfinished)} unfinished run(s): {', '.join(unfinished)}",
            AggregateWarning,
            stacklevel=stacklevel,
        )
    if not runs:
        raise ValueError(f"no finished run in {exp_dir}")
    runs.sort(key=lambda r: r.seed)  # numeric: seed 10 comes after seed 2
    seeds = [r.seed for r in runs]
    if len(set(seeds)) != len(seeds):
        dirs = ", ".join(f"{r.run_dir.name} (seed {r.seed})" for r in runs)
        raise ValueError(f"{exp_dir}: duplicate seeds among run directories: {dirs}")
    _require_one_configuration(exp_dir, runs)
    return ExperimentScan(runs=runs, aborted=aborted, unfinished=unfinished)


def load_runs(exp_dir: str | Path) -> list[RunResult]:
    """Finished runs of one experiment directory, sorted by seed.

    Aborted runs (``aborted.json``) and unfinished runs (only one of
    ``config.yaml`` / ``final_eval.json``) are skipped and named in one
    :class:`AggregateWarning` each (:func:`scan_experiment` has the rules).
    Raises ``ValueError`` if there is no finished run or if two runs claim
    the same seed.
    """
    return scan_experiment(exp_dir, stacklevel=3).runs


#: Top-level keys of ``config.yaml`` that may differ between the runs of one
#: experiment: the ``run`` block (seed, git, start time, argv) and the name.
_PER_RUN_KEYS = ("run", "exp")
_ABSENT = object()


def _flat_settings(node: Any, prefix: str = "") -> dict[str, Any]:
    """``{"agent": {"alpha": 0.1}}`` -> ``{"agent.alpha": 0.1}``."""
    if not isinstance(node, dict):
        return {prefix[:-1]: node}
    flat: dict[str, Any] = {}
    for key, value in node.items():
        flat.update(_flat_settings(value, f"{prefix}{key}."))
    return flat


def _require_one_configuration(exp_dir: Path, runs: Sequence[RunResult]) -> None:
    """``ValueError`` unless all finished runs were trained with one configuration.

    The seed is the only thing that may differ between the replicates of a
    row; a run started with another ``--set`` under the same experiment name
    is not a replicate. Compared: every key of ``config.yaml`` except the
    ``run`` block and ``exp``, as written (a key missing in one file differs).
    """
    settings = [_flat_settings({k: v for k, v in r.config.items() if k not in _PER_RUN_KEYS}) for r in runs]
    first, first_dir = settings[0], runs[0].run_dir.name
    differences = []
    for run, other in zip(runs[1:], settings[1:]):
        for key in sorted(first.keys() | other.keys()):
            a, b = first.get(key, _ABSENT), other.get(key, _ABSENT)
            if a is _ABSENT or b is _ABSENT or a != b:
                shown = [("absent" if v is _ABSENT else repr(v)) for v in (a, b)]
                differences.append(f"{key}: {shown[0]} in {first_dir}, {shown[1]} in {run.run_dir.name}")
    if differences:
        raise ValueError(
            f"{exp_dir}: the runs do not share one configuration ({'; '.join(differences)}); "
            f"a row must average seeds of ONE configuration - move the odd run(s) to their own experiment directory"
        )


# ----------------------------------------------------------------- summaries
def _check_last_k(last_k: int) -> None:
    if isinstance(last_k, bool) or not isinstance(last_k, (int, np.integer)) or last_k < 1:
        raise ValueError(f"last_k must be an integer >= 1, got {last_k!r}")


def _last_k_summary(runs: Sequence[RunResult], last_k: int) -> dict[str, Any]:
    """The keys ``last_k`` adds to a summary row (see :func:`summarise_experiment`)."""
    _check_last_k(last_k)
    per_seed = []
    for r in runs:
        path = r.run_dir / METRICS_FILE
        if "eval_mean_score" not in r.metrics.columns:
            raise ValueError(f"{path}: missing column(s) ['eval_mean_score']")
        scores = r.metrics["eval_mean_score"].to_numpy(dtype=np.float64)
        if scores.size < last_k:
            raise ValueError(f"{path}: {scores.size} evaluation row(s), fewer than last_k = {last_k}")
        tail = scores[-last_k:]  # file order: the training loop appends one row per evaluation
        if not np.isfinite(tail).all():
            raise ValueError(f"{path}: eval_mean_score is not a finite number in the last {last_k} row(s)")
        per_seed.append(float(tail.mean()))
    mean, ci = mean_ci95(per_seed)  # the seed is the unit of replication, as for the final evaluation
    return {"last_k": int(last_k), "lastk_mean_score": mean, "lastk_ci95": ci, "lastk_per_seed": per_seed}


def _summarise_runs(
    runs: Sequence[RunResult], label: str, last_k: int | None = None, aborted: Sequence[str] = ()
) -> dict[str, Any]:
    finals = [r.final for r in runs]
    per_seed_means = [f.mean_score for f in finals]
    mean, ci = mean_ci95(per_seed_means)  # the seed is the unit of replication
    counts = [len(f.episodes) for f in finals]
    # Episode-level statistics over all seeds pooled; same code path as a
    # baseline row. The pooled mean and CI of this object are NOT used.
    pooled = EvalResult(
        policy=label, env=finals[0].env, seed=finals[0].seed, episodes=[e for f in finals for e in f.episodes]
    ).summary()
    summary = {
        "label": label,
        "n_seeds": len(runs),
        "seeds": [r.seed for r in runs],
        "n_episodes": counts[0] if len(set(counts)) == 1 else None,
        "per_seed_n_episodes": counts,
        "mean_score": mean,
        "ci95": ci,
        "ci_kind": "seeds",
        "per_seed_mean_scores": per_seed_means,
        "median_score": pooled["median_score"],
        "max_score": pooled["max_score"],
        "mean_steps": pooled["mean_steps"],
        "cause_fractions": pooled["cause_fractions"],
        "n_aborted": len(aborted),
        "aborted": list(aborted),
    }
    if last_k is not None:
        summary.update(_last_k_summary(runs, last_k))
    return summary


def summarise_experiment(
    exp_dir: str | Path, label: str | None = None, last_k: int | None = None
) -> dict[str, Any]:
    """Across-seed summary of the final evaluation of one experiment.

    ``mean_score`` is the mean over seeds of the per-seed mean score and
    ``ci95`` the Student-t half-width over those per-seed means (nan for one
    seed). ``n_episodes`` is the episode count per seed, or ``None`` if the
    seeds disagree (``per_seed_n_episodes`` always lists them). Median, max,
    mean steps and cause fractions are over all pooled episodes. ``n_aborted``
    and ``aborted`` count and name the run directories of the experiment
    that hold ``aborted.json`` (a diverged DQN run; not in any statistic).

    With ``last_k`` the dict gains four keys: ``last_k``; ``lastk_per_seed``,
    each seed's mean of ``eval_mean_score`` over the last ``last_k`` rows of
    its ``metrics.csv`` (seed order); ``lastk_mean_score``, the mean of those
    per-seed numbers; ``lastk_ci95``, their Student-t half-width (nan for one
    seed). ``ValueError`` if ``last_k < 1`` or a run has fewer rows. Without
    ``last_k`` the dict is exactly the one described above.
    """
    if last_k is not None:
        _check_last_k(last_k)  # before any file is read
    exp_dir = Path(exp_dir)
    scan = scan_experiment(exp_dir, stacklevel=3)
    return _summarise_runs(scan.runs, label if label is not None else exp_dir.resolve().name, last_k, scan.aborted)


def summarise_baseline(json_path: str | Path, label: str | None = None) -> dict[str, Any]:
    """The same keys for one ``EvalResult`` JSON of a non-learning player.

    A baseline has no training seed: ``n_seeds`` is 1 (one evaluation),
    ``seeds`` is empty and ``ci95`` is the episode-level normal-approximation
    half-width of that evaluation (``ci_kind = "episodes"``). The default
    label is the file stem, e.g. ``bfs_W10_relative``.
    """
    json_path = Path(json_path)
    result = EvalResult.load(json_path)
    if not result.episodes:
        raise ValueError(f"{json_path}: no episodes")
    s = result.summary()
    return {
        "label": label if label is not None else json_path.stem,
        "n_seeds": 1,
        "seeds": [],
        "n_episodes": s["n_episodes"],
        "per_seed_n_episodes": [s["n_episodes"]],
        "mean_score": s["mean_score"],
        "ci95": s["ci95"],
        "ci_kind": "episodes",
        "per_seed_mean_scores": [s["mean_score"]],
        "median_score": s["median_score"],
        "max_score": s["max_score"],
        "mean_steps": s["mean_steps"],
        "cause_fractions": s["cause_fractions"],
        "n_aborted": 0,
        "aborted": [],
    }


def results_table(rows: Sequence[dict[str, Any]]) -> str:
    """Markdown table of summary rows (``summarise_experiment`` / ``summarise_baseline``).

    A CI that does not exist (one seed) is printed as ``n/a``, never as a
    number. Seeds with different episode counts are printed as ``min-max``.
    A row with aborted runs (``n_aborted > 0``) prints its seeds cell as
    ``<n_seeds> (+<n_aborted> aborted)``; without any, the table is unchanged.

    If at least one row carries ``last_k`` (``summarise_experiment(...,
    last_k=k)``), a last column ``last-<k> evals mean +- 95% CI`` is appended;
    rows without it (baselines) show ``n/a`` there. Rows with different ``k``
    in one table are a ``ValueError``. Without any such row the table is the
    twelve-column table, unchanged.
    """
    last_ks = sorted({int(r["last_k"]) for r in rows if r.get("last_k") is not None})
    if len(last_ks) > 1:
        raise ValueError(f"rows carry different last_k values {last_ks}; one table has one last-k column")
    head = (
        "| label | seeds | episodes | score mean +- 95% CI | CI over | median | max | mean steps | "
        + " | ".join(CAUSES)
        + " |"
    )
    if last_ks:
        head += f" last-{last_ks[0]} evals mean +- 95% CI |"
    lines = [head, "|" + "---|" * (8 + len(CAUSES) + len(last_ks))]
    for r in rows:
        ci = r["ci95"]
        ci_text = f"{ci:.2f}" if ci is not None and math.isfinite(ci) else "n/a"
        if r["n_episodes"] is not None:
            episodes = str(r["n_episodes"])
        else:
            episodes = f"{min(r['per_seed_n_episodes'])}-{max(r['per_seed_n_episodes'])}"
        frac = " | ".join(f"{r['cause_fractions'][c]:.2f}" for c in CAUSES)
        label = str(r["label"]).replace("|", "\\|")
        n_aborted = int(r.get("n_aborted") or 0)
        seeds_text = f"{r['n_seeds']} (+{n_aborted} aborted)" if n_aborted else str(r["n_seeds"])
        line = (
            f"| {label} | {seeds_text} | {episodes} | {r['mean_score']:.2f} +- {ci_text} | {r['ci_kind']} | "
            f"{r['median_score']:.1f} | {r['max_score']} | {r['mean_steps']:.1f} | {frac} |"
        )
        if last_ks:
            if r.get("last_k") is None:
                line += " n/a |"
            else:
                k_ci = r["lastk_ci95"]
                k_ci_text = f"{k_ci:.2f}" if k_ci is not None and math.isfinite(k_ci) else "n/a"
                line += f" {r['lastk_mean_score']:.2f} +- {k_ci_text} |"
        lines.append(line)
    return "\n".join(lines)


# -------------------------------------------------------------------- curves
def _curve_from_runs(runs: Sequence[RunResult]) -> pd.DataFrame:
    series = []
    for r in runs:
        missing = {"env_steps", "eval_mean_score"} - set(r.metrics.columns)
        if missing:
            raise ValueError(f"{r.run_dir / METRICS_FILE}: missing column(s) {sorted(missing)}")
        s = r.metrics.set_index("env_steps")["eval_mean_score"]
        if not s.index.is_unique:
            raise ValueError(f"{r.run_dir / METRICS_FILE}: duplicate env_steps")
        series.append(s)
    step_sets = [set(s.index) for s in series]
    common = sorted(set.intersection(*step_sets))
    n_dropped = len(set.union(*step_sets)) - len(common)
    if n_dropped:
        warnings.warn(
            f"{runs[0].run_dir.parent}: {n_dropped} env_steps value(s) are not present in all "
            f"{len(runs)} finished runs and are left out of the curve",
            AggregateWarning,
            stacklevel=3,
        )
    # values[i, j] = eval_mean_score of seed i at common[j]
    values = np.array([s.loc[common].to_numpy(dtype=np.float64) for s in series]).reshape(len(runs), len(common))
    return pd.DataFrame(
        {
            "env_steps": np.array(common, dtype=np.int64),
            "n_seeds": len(runs),
            "mean": values.mean(axis=0),
            "ci95": _t_half_width(values, axis=0),
            "min": values.min(axis=0),
            "max": values.max(axis=0),
        },
        columns=list(CURVE_COLUMNS),
    )


def learning_curve(exp_dir: str | Path) -> pd.DataFrame:
    """Evaluation curve across seeds, one row per ``env_steps`` value.

    Only ``env_steps`` values present in ALL finished runs are used, so every
    row averages the same seeds. Columns: ``env_steps``, ``n_seeds``,
    ``mean`` (of ``eval_mean_score`` across seeds), ``ci95`` (Student t across
    seeds, nan for one seed), ``min``, ``max``.
    """
    return _curve_from_runs(load_runs(exp_dir))


# ------------------------------------------------------- death causes (M8)
#: Columns of :func:`death_cause_curve`: per cause, the mean over seeds of
#: the evaluation fraction and its Student-t 95 % half-width.
DEATH_CAUSE_COLUMNS = (
    "env_steps", "n_seeds", "wall_mean", "wall_ci95", "self_mean", "self_ci95", "starve_mean", "starve_ci95",
    "win_mean", "win_ci95",
)
_DEATH_CAUSES = ("wall", "self", "starve", "win")


def death_cause_curve(exp_dir: str | Path) -> pd.DataFrame:
    """How the evaluation episodes end over training, across seeds.

    One row per ``env_steps`` value present in all finished runs (the rows of
    :func:`learning_curve`, same warning). For every cause ``wall`` /
    ``self`` / ``starve`` / ``win``: the mean over seeds of the run's
    ``eval_frac_<cause>`` column and the Student-t 95 % half-width over
    seeds (nan for one seed). The four means of a row sum to 1. A run
    without one of the columns raises ``ValueError``.
    """
    runs = load_runs(exp_dir)
    curve = _curve_from_runs(runs)
    common = curve["env_steps"].to_numpy(dtype=np.int64)
    out: dict[str, Any] = {"env_steps": common, "n_seeds": len(runs)}
    for cause in _DEATH_CAUSES:
        column = f"eval_frac_{cause}"
        rows = []
        for r in runs:
            if column not in r.metrics.columns:
                raise ValueError(f"{r.run_dir / METRICS_FILE}: missing column {column!r}")
            rows.append(r.metrics.set_index("env_steps")[column].loc[common].to_numpy(dtype=np.float64))
        values = np.array(rows).reshape(len(runs), len(common))
        out[f"{cause}_mean"] = values.mean(axis=0)
        out[f"{cause}_ci95"] = _t_half_width(values, axis=0)
    return pd.DataFrame(out, columns=list(DEATH_CAUSE_COLUMNS))


# -------------------------------------------- value estimates vs returns (M7)
def _value_series(run: RunResult, which: str) -> pd.DataFrame:
    """``env_steps``-indexed frame with columns ``value`` and ``return`` of one run.

    From the ``eval_value_<which>`` / ``eval_return_<which>`` columns of
    ``metrics.csv`` when both are present and complete; otherwise from
    ``values.csv`` (``scripts/backfill_values.py``); otherwise ``ValueError``.
    """
    columns = [f"eval_value_{which}", f"eval_return_{which}"]
    frame = run.metrics
    if not (set(columns) <= set(frame.columns) and frame[columns].notna().all().all()):
        values_path = run.run_dir / VALUES_FILE
        if not values_path.is_file():
            raise ValueError(
                f"{run.run_dir}: no value estimates ({', '.join(columns)} in {METRICS_FILE}, or {VALUES_FILE}): "
                f"train with the M7 code or run scripts/backfill_values.py on the run's checkpoints"
            )
        frame = pd.read_csv(values_path, float_precision="round_trip")
        missing = {"env_steps", *columns} - set(frame.columns)
        if missing or frame[columns].isna().any().any():
            raise ValueError(f"{values_path}: missing column(s) or values {sorted(missing)}")
    series = frame.set_index("env_steps")[columns]
    if not series.index.is_unique:
        raise ValueError(f"{run.run_dir}: duplicate env_steps")
    return series.rename(columns=dict(zip(columns, ("value", "return"))))


def value_curve(exp_dir: str | Path, which: str = "visited") -> pd.DataFrame:
    """Value estimates against realised returns across seeds (M7), one row per ``env_steps``.

    ``which``: ``"visited"`` (all states the greedy policy visits on the
    evaluation boards) or ``"start"`` (the first state of each episode).
    Finished runs only; only ``env_steps`` present in all of them (the same
    rule and warning as :func:`learning_curve`). Columns
    (``VALUE_CURVE_COLUMNS``): ``value_mean`` / ``value_ci95`` of the
    estimate ``max_a Q``, ``return_mean`` / ``return_ci95`` of the realised
    discounted return, ``gap_mean`` / ``gap_ci95`` of the per-seed difference
    ``value - return`` (the overestimation); Student-t 95 % half-widths with
    the seed as the unit, nan for one seed.
    """
    if which not in ("visited", "start"):
        raise ValueError(f'which must be "visited" or "start", got {which!r}')
    runs = load_runs(exp_dir)
    series = [_value_series(r, which) for r in runs]
    step_sets = [set(s.index) for s in series]
    common = sorted(set.intersection(*step_sets))
    n_dropped = len(set.union(*step_sets)) - len(common)
    if n_dropped:
        warnings.warn(
            f"{runs[0].run_dir.parent}: {n_dropped} env_steps value(s) are not present in all "
            f"{len(runs)} finished runs and are left out of the value curve",
            AggregateWarning,
            stacklevel=2,
        )
    values = np.array([s.loc[common, "value"].to_numpy(dtype=np.float64) for s in series]).reshape(len(runs), -1)
    returns = np.array([s.loc[common, "return"].to_numpy(dtype=np.float64) for s in series]).reshape(len(runs), -1)
    gaps = values - returns
    return pd.DataFrame(
        {
            "env_steps": np.array(common, dtype=np.int64),
            "n_seeds": len(runs),
            "value_mean": values.mean(axis=0),
            "value_ci95": _t_half_width(values, axis=0),
            "return_mean": returns.mean(axis=0),
            "return_ci95": _t_half_width(returns, axis=0),
            "gap_mean": gaps.mean(axis=0),
            "gap_ci95": _t_half_width(gaps, axis=0),
        },
        columns=list(VALUE_CURVE_COLUMNS),
    )


# ----------------------------------------------------------------------- CLI
def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Results table and learning curves from run directories.")
    p.add_argument("--baseline", nargs="+", action="extend", default=[], metavar="FILE",
                   help="EvalResult JSON files written by python -m snake_rl.evaluate")
    p.add_argument("--exp", nargs="+", action="extend", default=[], metavar="DIR",
                   help="experiment directories <runs_dir>/<exp> holding one sub-directory per seed")
    p.add_argument("--out", metavar="FILE.md", help="also write the table to this file")
    p.add_argument("--curves-dir", metavar="DIR", help="write <exp name>_curve.csv per experiment")
    p.add_argument("--last-k", type=int, default=None, metavar="K",
                   help="add a column: mean +- 95%% CI over seeds of each seed's mean score over its last K "
                        "evaluations (default: off)")
    args = p.parse_args(argv)
    if not args.baseline and not args.exp:
        p.error("nothing to aggregate: give --baseline FILE and/or --exp DIR")
    if args.last_k is not None and args.last_k < 1:
        p.error(f"--last-k must be >= 1, got {args.last_k}")

    rows = [summarise_baseline(f) for f in args.baseline]
    curves: dict[str, pd.DataFrame] = {}
    for d in args.exp:
        name = Path(d).resolve().name
        scan = scan_experiment(d)
        rows.append(_summarise_runs(scan.runs, name, args.last_k, scan.aborted))
        if args.curves_dir:
            if name in curves:
                p.error(f"two experiments are named {name!r}: their curve files would overwrite each other")
            curves[name] = _curve_from_runs(scan.runs)

    table = results_table(rows)
    print(table)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table + "\n", encoding="utf-8", newline="\n")
    for name, curve in curves.items():
        curves_dir = Path(args.curves_dir)
        curves_dir.mkdir(parents=True, exist_ok=True)
        curve.to_csv(curves_dir / f"{name}_curve.csv", index=False, lineterminator="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

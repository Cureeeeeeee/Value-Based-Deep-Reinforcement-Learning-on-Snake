"""Value estimates against realised returns for runs trained before M7, from
their checkpoints (PLAN.md milestone M7: the overestimation measurement).

Usage, from the repo root::

    python scripts/backfill_values.py --exp runs/dqn_A_W8 [runs/dqn_B8_W8 ...]
        [--device auto|cpu|cuda] [--force] [--quiet]

A DQN run trained with the M7 code logs, at every evaluation, the network's
value estimate ``max_a Q`` on the states its greedy policy visits and the
discounted return it then realises from them (the ``eval_value_*`` /
``eval_return_*`` columns of ``metrics.csv``; ``snake_rl.evaluate``). A run
trained earlier has the same networks on disk: one weights-only checkpoint
per evaluation (``checkpoints/step_<env_steps>.pt``). This script evaluates
each checkpoint with the run's own configuration (boards, episodes, gamma)
and writes ``<run>/values.csv`` with the columns::

    env_steps,eval_mean_score,eval_value_start,eval_return_start,eval_value_visited,eval_return_visited

one row per checkpoint in step order. ``eval_mean_score`` is the score of
that evaluation; it must equal the ``metrics.csv`` row of the same
``env_steps`` exactly (the checkpoint IS the network that produced the row),
which the script checks: a mismatch is reported and the exit code is 1
(every run is still processed). ``aggregate.value_curve`` reads
``values.csv`` when ``metrics.csv`` has no value columns.

A run that already has ``values.csv`` is skipped unless ``--force``; a
directory without ``checkpoints/`` (tabular runs, ``keep_checkpoints:
false``) is skipped. No other file of a run is touched. The run's own
``agent.torch_threads`` is applied before its checkpoints are evaluated, so a
CPU run's cells come out exactly as the trainer wrote them (a different
thread count changes the rounding of a batched forward pass in the last
bits).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snake_rl.agents.dqn.agent import DQNAgent  # noqa: E402
from snake_rl.analysis.aggregate import VALUES_FILE  # noqa: E402
from snake_rl.config import load_config  # noqa: E402
from snake_rl.evaluate import evaluate_batched  # noqa: E402
from snake_rl.train import CHECKPOINT_DIR, CONFIG_FILE, METRICS_FILE, VALUE_COLUMNS, value_columns  # noqa: E402

COLUMNS = ("env_steps", "eval_mean_score", *VALUE_COLUMNS)


def checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    """``(env_steps, file)`` of every ``checkpoints/step_<digits>.pt``, in step order."""
    found = []
    for path in (run_dir / CHECKPOINT_DIR).glob("step_*.pt"):
        digits = path.stem[len("step_"):]
        if digits.isdigit():
            found.append((int(digits), path))
    return sorted(found)


def metrics_scores(run_dir: Path) -> dict[int, float]:
    """``env_steps -> eval_mean_score`` of ``metrics.csv`` (empty if absent)."""
    path = run_dir / METRICS_FILE
    if not path.is_file():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {int(row["env_steps"]): float(row["eval_mean_score"]) for row in csv.DictReader(f)}


def backfill_run(run_dir: Path, device: str, force: bool) -> tuple[str, bool]:
    """Write ``values.csv`` for one run; returns ``(status line, consistent)``."""
    out = run_dir / VALUES_FILE
    if not (run_dir / CONFIG_FILE).is_file() or not (run_dir / CHECKPOINT_DIR).is_dir():
        return f"{run_dir}: no {CONFIG_FILE} or no {CHECKPOINT_DIR}/, skipped", True
    if out.is_file() and not force:
        return f"{run_dir}: {VALUES_FILE} exists, skipped (--force to redo)", True
    cfg = load_config(run_dir / CONFIG_FILE)
    if cfg.agent_kind != "dqn":
        return f"{run_dir}: agent.kind {cfg.agent_kind}, skipped", True
    files = checkpoints(run_dir)
    if not files:
        return f"{run_dir}: no checkpoint files, skipped", True
    if cfg.agent.torch_threads is not None:
        torch.set_num_threads(cfg.agent.torch_threads)  # the trainer's setting: same rounding as its rows
    scores = metrics_scores(run_dir)
    rows, mismatches = [], []
    for env_steps, path in files:
        agent = DQNAgent.load(path, device=device)
        result = evaluate_batched(agent, cfg.env, cfg.train.eval_episodes, cfg.train.eval_seed, gamma=cfg.agent.gamma)
        rows.append({"env_steps": env_steps, "eval_mean_score": result.mean_score, **value_columns(result)})
        if env_steps in scores and scores[env_steps] != result.mean_score:
            mismatches.append(f"step {env_steps}: checkpoint scores {result.mean_score!r}, metrics.csv {scores[env_steps]!r}")
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow([str(row["env_steps"])] + [repr(float(row[name])) for name in COLUMNS[1:]])
    line = f"{run_dir}: {len(rows)} checkpoints -> {VALUES_FILE}"
    if mismatches:
        line += "\n  MISMATCH with metrics.csv: " + "; ".join(mismatches)
    return line, not mismatches


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Value estimates vs realised returns from the checkpoints of finished runs.")
    p.add_argument("--exp", nargs="+", action="extend", required=True, metavar="DIR",
                   help="experiment directories <runs_dir>/<exp>; every <seed>/ sub-directory is a run")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"), help="where to evaluate (default: auto)")
    p.add_argument("--force", action="store_true", help="rewrite an existing values.csv")
    p.add_argument("--quiet", action="store_true", help="no per-run lines")
    args = p.parse_args(argv)

    consistent = True
    for exp in args.exp:
        exp_dir = Path(exp)
        if not exp_dir.is_dir():
            print(f"error: {exp_dir} is not a directory", file=sys.stderr)
            return 2
        for run_dir in sorted(d for d in exp_dir.iterdir() if d.is_dir()):
            line, ok = backfill_run(run_dir, args.device, args.force)
            consistent &= ok
            if not args.quiet or not ok:
                print(line, flush=True)
    return 0 if consistent else 1


if __name__ == "__main__":
    raise SystemExit(main())

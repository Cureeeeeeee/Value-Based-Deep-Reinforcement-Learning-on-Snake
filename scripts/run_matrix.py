"""Batch runner: seeds x grid of overrides, one ``snake_rl.train`` process per run.

Every multi-seed batch of the project is launched through this script (five
seeds per configuration is a project rule). Usage, from the repo root::

    python scripts/run_matrix.py --config configs/x.yaml --seeds 0 1 2 3 4
        [--grid agent.alpha=0.05,0.1,0.2 agent.gamma=0.9,0.99] [--set key=value ...]
        [--exp-prefix NAME] [--runs-dir runs] [--jobs 4] [--dry-run]

There are two modes:

* Run now (default): the runs are started from this process, at most
  ``--jobs`` at a time (default 1), and a status line per run is printed at
  the end. For batches of a few minutes::

      python scripts/run_matrix.py --config configs/tabular_q_8x8.yaml --seeds 0 1 2 3 4 --jobs 5

* ``--dry-run``: prints the commands of the runs that would be started, one
  per line on stdout, and starts nothing. Long batches are launched by the
  user in a separate terminal (project rule: training of more than ten
  minutes is not run in the assistant's foreground), either by pasting these
  lines or by running this script there without ``--dry-run``::

      python scripts/run_matrix.py --config configs/tabular_q_10x10.yaml --seeds 0 1 2 3 4
          --grid agent.alpha=0.05,0.1,0.2 agent.gamma=0.9,0.99 --dry-run

The matrix
----------
The runs are the cartesian product of the ``--grid`` axes times the seeds:
grid point by grid point (first axis slowest), the seeds of a grid point next
to each other, so a sequential batch completes one experiment after the other.
An axis is ``dotted.key=value1,value2,...``; ``--set`` overrides apply to every
run and are applied before the grid values (``snake_rl.config.load_config``
describes the keys).

Each grid point is its own experiment, named
``<base>__<lastkey>=<value>__...``: the last component of each dotted key with
the value as typed, axes in the order given, no suffix without ``--grid``.
``<base>`` is ``exp`` of the config file, or ``--exp-prefix``. Example: the
grid above on ``exp: tabular_q_W10`` gives ``tabular_q_W10__alpha=0.05__gamma=0.9``
and five more. A value that cannot be part of a directory name (anything but
letters, digits and ``. _ - = +``) is an error.

Each run is one process, written to ``<runs-dir>/<exp>/<seed>/``::

    <python> -m snake_rl.train --config F --seed S --exp NAME [--set ...] --runs-dir D --quiet

so a run started here is the run that the same command line gives by hand:
the result depends on the configuration and the seed only, never on ``--jobs``
or on what else is running.

Checked before anything starts
------------------------------
Every generated configuration is loaded with ``load_config`` (same overrides
as the process will get), so a typo in ``--grid`` or ``--set`` fails in a
second with the offending key and not after the first job. Also errors:
the same run twice (a repeated seed or grid value), and an existing run
directory under a planned name whose ``config.yaml`` is another configuration
(one experiment name must not mix configurations; pick a new ``--exp-prefix``).

Existing runs are never touched
-------------------------------
``done``: ``final_eval.json`` exists; the run is skipped. ``incomplete``:
``config.yaml`` without ``final_eval.json``, i.e. a run that is still going in
another terminal or one that was interrupted; it is NOT started and nothing is
deleted (``snake_rl.train`` refuses the directory anyway). The user decides:
wait, or move the directory away and call this script again. Calling the
script again is therefore how a partly finished batch is continued.

Output and exit code
--------------------
One line when a run finishes, then one status line per run, ``<run dir>:
<status>`` with status ``ok``, ``failed (exit code N)``, ``done`` or
``incomplete``, and a line of counts. Whatever a failing process printed goes
straight to the terminal. Exit code 1 if any run failed, 2 if the matrix is
invalid (nothing was started), 130 after Ctrl-C (queued runs are not started,
the processes in flight are killed and stay behind as incomplete runs), else 0;
``done`` and ``incomplete`` runs do not change the exit code.

``--dry-run`` prints the commands in the quoting of the platform's shell
(cmd.exe on Windows, POSIX sh elsewhere), with paths as given, so they are
meant for the directory this script was called from. Runs that are ``done``
or ``incomplete`` have no command; they are listed on stderr.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shlex
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from snake_rl.config import ExperimentConfig, config_to_dict, load_config
from snake_rl.train import CONFIG_FILE, FINAL_EVAL_FILE

PENDING, OK, FAILED, DONE, INCOMPLETE = "pending", "ok", "failed", "done", "incomplete"
#: Seconds between two looks at the running processes.
POLL_SECONDS = 0.05


@dataclass
class Run:
    """One (grid point, seed) of the matrix."""

    exp: str
    seed: int
    run_dir: Path
    command: list[str]
    #: ``pending`` (to be started) becomes ``ok`` or ``failed``; ``done`` and
    #: ``incomplete`` runs exist already and are never started.
    status: str
    returncode: int | None = None

    @property
    def label(self) -> str:
        return self.run_dir.as_posix()

    def describe(self) -> str:
        return f"{FAILED} (exit code {self.returncode})" if self.status == FAILED else self.status


# ------------------------------------------------------------------ planning
def parse_axis(text: str) -> tuple[str, list[str]]:
    """``dotted.key=v1,v2,...`` -> ``(key, [v1, v2, ...])``."""
    key, sep, tail = text.partition("=")
    key, values = key.strip(), [value.strip() for value in tail.split(",")]
    if not sep or not all(key.split(".")) or not all(values):
        raise ValueError(f"--grid {text!r} is not of the form dotted.key=value1,value2,...")
    if key == "exp":
        raise ValueError("--grid exp=...: the names are generated from the grid; --exp-prefix sets the base name")
    return key, values


def plan_runs(
    config: str | Path,
    seeds: Sequence[int],
    grid: Sequence[str] = (),
    sets: Sequence[str] = (),
    exp_prefix: str | None = None,
    runs_dir: str | Path = "runs",
) -> list[Run]:
    """Expand the matrix into runs and check all of them (module docstring).

    Nothing is started or written. ``ValueError`` names the first problem;
    ``OSError`` if the config file cannot be read.
    """
    axes = [parse_axis(text) for text in grid]
    keys = [key for key, _ in axes]
    repeated = sorted({key for key in keys if keys.count(key) > 1})
    if repeated:
        raise ValueError(f"--grid has more than one axis for {', '.join(repeated)}")

    runs: list[Run] = []
    for point in itertools.product(*([(key, value) for value in values] for key, values in axes)):
        grid_overrides = [f"{key}={value}" for key, value in point]
        overrides = [*sets, *grid_overrides]
        try:
            base = exp_prefix if exp_prefix is not None else load_config(config, overrides).exp
            exp = "__".join([base, *(f"{key.rpartition('.')[2]}={value}" for key, value in point)])
            # The override list snake_rl.train builds from "--set ... --exp NAME".
            cfg = load_config(config, [*overrides, f"exp={json.dumps(exp)}"])
        except ValueError as err:
            where = f"grid point {' '.join(grid_overrides)}: " if point else ""
            raise ValueError(f"{where}{err}") from err
        for seed in seeds:
            run_dir = Path(runs_dir) / exp / str(seed)
            command = [sys.executable, "-m", "snake_rl.train", "--config", str(config), "--seed", str(seed)]
            command += ["--exp", exp, *(["--set", *overrides] if overrides else [])]
            command += ["--runs-dir", str(runs_dir), "--quiet"]
            runs.append(Run(exp, seed, run_dir, command, existing_status(run_dir, cfg)))

    twice = [label for label, n in Counter(run.label for run in runs).items() if n > 1]
    if twice:
        raise ValueError(f"the same run appears more than once (repeated seed or grid value): {', '.join(twice)}")
    return runs


def existing_status(run_dir: Path, cfg: ExperimentConfig) -> str:
    """``done`` / ``incomplete`` / ``pending`` from the files in ``run_dir``.

    ``ValueError`` if the directory holds a run of another configuration than
    ``cfg``: it can neither count as this run nor be started under this name.
    """
    recorded = run_dir / CONFIG_FILE
    if recorded.exists():
        there, here = (dict(_flatten(config_to_dict(c))) for c in (load_config(recorded), cfg))
        differences = [
            f"{key}: {there.get(key)!r} there, {here.get(key)!r} requested"
            for key in there | here
            if there.get(key) != here.get(key)
        ]
        if differences:
            raise ValueError(
                f"{run_dir.as_posix()} holds a run of another configuration ({'; '.join(differences)}); "
                f"runs are never overwritten - give this matrix another name with --exp-prefix"
            )
    if (run_dir / FINAL_EVAL_FILE).exists():
        return DONE
    return INCOMPLETE if recorded.exists() else PENDING


def _flatten(node: dict[str, Any], prefix: str = "") -> Iterator[tuple[str, Any]]:
    """``{"agent": {"alpha": 0.1}}`` -> ``("agent.alpha", 0.1)``, ..."""
    for key, value in node.items():
        if isinstance(value, dict):
            yield from _flatten(value, f"{prefix}{key}.")
        else:
            yield f"{prefix}{key}", value


# ----------------------------------------------------------------- execution
def run_pending(runs: Sequence[Run], jobs: int) -> None:
    """Start every pending run, at most ``jobs`` processes at a time, and
    record how each one ended. The processes inherit stdout and stderr.

    Everything happens in this thread, so an exception (Ctrl-C) stops the
    launching at once; the processes still running are then killed.
    """
    queue = [run for run in runs if run.status == PENDING]
    total = len(queue)
    active: list[tuple[Run, subprocess.Popen, float]] = []
    try:
        while queue or active:
            while queue and len(active) < jobs:
                run = queue.pop(0)
                active.append((run, subprocess.Popen(run.command), time.perf_counter()))
            time.sleep(POLL_SECONDS)
            for job in [job for job in active if job[1].poll() is not None]:
                active.remove(job)
                run, process, started = job
                run.returncode = process.returncode
                run.status = OK if process.returncode == 0 else FAILED
                finished = total - len(queue) - len(active)
                seconds = time.perf_counter() - started
                print(f"[{finished}/{total}] {run.label}: {run.describe()} ({seconds:.1f} s)", flush=True)
    finally:
        for _, process, _ in active:  # not empty only when an exception interrupted the loop
            process.kill()
            process.wait()


def format_command(command: Sequence[str]) -> str:
    """The command as one shell line: cmd.exe quoting on Windows (PowerShell
    reads it too, unless the first word needed quotes), POSIX quoting elsewhere."""
    return subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)


# ----------------------------------------------------------------------- CLI
def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Run or print a batch of training runs: the cartesian product of the --grid axes times the seeds.",
        allow_abbrev=False,
    )
    p.add_argument("--config", required=True, help="YAML experiment file (see snake_rl/config.py)")
    p.add_argument("--seeds", type=int, nargs="+", required=True, help="run seeds, e.g. 0 1 2 3 4")
    p.add_argument(
        "--grid",
        nargs="*",
        action="append",
        default=[],
        metavar="KEY=V1,V2",
        help="grid axes, e.g. agent.alpha=0.05,0.1,0.2 agent.gamma=0.9,0.99; each grid point is its own experiment",
    )
    p.add_argument(
        "--set",
        dest="sets",
        nargs="*",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="overrides for every run, applied before the grid values",
    )
    p.add_argument("--exp-prefix", help="base experiment name instead of exp of the config file")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--jobs", type=int, default=1, help="processes at a time (default 1)")
    p.add_argument("--dry-run", action="store_true", help="print the commands, one per line, and start nothing")
    args = p.parse_args(argv)
    if args.jobs < 1:
        p.error("--jobs must be at least 1")
    if min(args.seeds) < 0:
        p.error("--seeds must be >= 0")

    grid = [axis for group in args.grid for axis in group]
    sets = [item for group in args.sets for item in group]
    try:
        runs = plan_runs(args.config, args.seeds, grid, sets, args.exp_prefix, args.runs_dir)
    except (OSError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2

    if args.dry_run:
        for run in runs:
            if run.status == PENDING:
                print(format_command(run.command))
            else:
                print(f"no command for {run.label}: {run.status}", file=sys.stderr)
        return 0

    n_pending = sum(run.status == PENDING for run in runs)
    print(f"{len(runs)} runs, {n_pending} to start, at most {args.jobs} at a time", flush=True)
    try:
        run_pending(runs, args.jobs)
    except KeyboardInterrupt:
        print(
            "interrupted: queued runs were not started; the processes in flight were killed and their run "
            "directories are left as they are (incomplete)",
            file=sys.stderr,
        )
        return 130

    for run in runs:
        print(f"{run.label}: {run.describe()}")
    counts = Counter(run.status for run in runs)
    print(f"{len(runs)} runs: " + ", ".join(f"{counts[s]} {s}" for s in (OK, FAILED, DONE, INCOMPLETE) if counts[s]))
    if counts[INCOMPLETE]:
        print(
            "incomplete = config.yaml without final_eval.json (still running elsewhere, or interrupted): not started, "
            "nothing deleted. To run it again, move that run directory away first."
        )
    return 1 if counts[FAILED] else 0


if __name__ == "__main__":
    raise SystemExit(main())

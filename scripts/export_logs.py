"""Pack the light-weight training logs of ``runs/`` into one zip for the
results notebook (M10).

``runs/`` is not in git and holds about 1 GB with checkpoints; the files the
analysis reads are small (about 12 MB for 300 runs): per run ``config.yaml``,
``metrics.csv``, ``final_eval.json``, ``values.csv`` (backfilled value
estimates) and ``aborted.json``; the baseline evaluations
``runs/baselines/*.json``; optionally the behaviour analyses
(``runs/behaviour/**``: tables, summaries, heat-maps and recordings, no
weights). Weight files (``*.pt``, ``*.npz``, anything under ``checkpoints/``)
and the regenerable outputs (``figures``, ``tables``, ``report``, ``review``,
``bench``, ``recordings``, ``_interrupted``) are never included.

Entries are ``runs/<...>`` paths relative to the parent of the runs
directory, so unzipping at the repo root recreates ``runs/`` as the analysis
expects it::

    python scripts/export_logs.py [--runs-dir runs] [--out runs_logs.zip] [--include-behaviour] [--list]
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path
from typing import Sequence

#: Per-run files the analysis reads (any depth: sweeps nest grid points).
LOG_FILES = ("config.yaml", "metrics.csv", "final_eval.json", "values.csv", "aborted.json")
#: Top-level directories of ``runs/`` that are never exported.
EXCLUDED_DIRS = ("_interrupted", "bench", "figures", "recordings", "report", "review", "tables")
#: What is taken from ``runs/behaviour`` with ``--include-behaviour``.
BEHAVIOUR_SUFFIXES = (".json", ".csv", ".png", ".md", ".npy")
BASELINES_DIR = "baselines"
BEHAVIOUR_DIR = "behaviour"
CHECKPOINTS_DIR = "checkpoints"


def select_entries(runs_dir: str | Path, include_behaviour: bool = False) -> list[Path]:
    """The files to export, as paths relative to ``runs_dir``, sorted."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        raise ValueError(f"{runs_dir} is not a directory")
    entries: list[Path] = []
    for root, dirs, files in os.walk(runs_dir):
        rel_root = Path(root).relative_to(runs_dir)
        top = rel_root.parts[0] if rel_root.parts else None
        if top is None:
            dirs[:] = [
                d for d in dirs if d not in EXCLUDED_DIRS and (include_behaviour or d != BEHAVIOUR_DIR)
            ]
        dirs[:] = sorted(d for d in dirs if d != CHECKPOINTS_DIR)
        for name in sorted(files):
            suffix = Path(name).suffix.lower()
            if suffix in (".pt", ".npz"):
                continue
            if top == BASELINES_DIR:
                keep = suffix == ".json" and len(rel_root.parts) == 1  # directly in runs/baselines
            elif top == BEHAVIOUR_DIR:
                keep = suffix in BEHAVIOUR_SUFFIXES
            else:
                keep = name in LOG_FILES
            if keep:
                entries.append(rel_root / name)
    return sorted(entries, key=lambda p: p.as_posix())


def export_logs(runs_dir: str | Path = "runs", out: str | Path = "runs_logs.zip", include_behaviour: bool = False) -> dict[str, int]:
    """Write the zip; returns ``{"files", "runs", "bytes"}`` (``runs``: run
    directories that contributed a ``config.yaml`` or ``final_eval.json``,
    ``bytes``: uncompressed size)."""
    runs_dir, out = Path(runs_dir), Path(out)
    entries = select_entries(runs_dir, include_behaviour)
    out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    run_dirs: set[Path] = set()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in entries:
            path = runs_dir / rel
            total += path.stat().st_size
            if rel.name in ("config.yaml", "final_eval.json"):
                run_dirs.add(rel.parent)
            zf.write(path, (Path(runs_dir.name) / rel).as_posix())
    return {"files": len(entries), "runs": len(run_dirs), "bytes": total}


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Pack the training logs of runs/ (no weights) into one zip.")
    p.add_argument("--runs-dir", default="runs", metavar="DIR")
    p.add_argument("--out", default="runs_logs.zip", metavar="FILE.zip")
    p.add_argument("--include-behaviour", action="store_true", help="also the behaviour analyses (tables, PNGs, recordings)")
    p.add_argument("--list", action="store_true", help="print the entries, write nothing")
    args = p.parse_args(argv)
    try:
        if args.list:
            for rel in select_entries(args.runs_dir, args.include_behaviour):
                print((Path(args.runs_dir).name / rel).as_posix())
            return 0
        counts = export_logs(args.runs_dir, args.out, args.include_behaviour)
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    print(f"{counts['files']} files from {counts['runs']} run directories, {counts['bytes'] / 1e6:.1f} MB uncompressed -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

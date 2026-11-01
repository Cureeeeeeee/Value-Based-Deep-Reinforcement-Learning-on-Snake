"""Pack what a live demonstration needs from ``runs/`` (not in git) into one zip.

Usage::

    python scripts/make_demo_pack.py                       # -> runs/export/demo_pack.zip
    python scripts/make_demo_pack.py --out /path/demo.zip --runs-dir runs

The archive stores ``runs/<relative path>`` entries, so unpacking it at the
root of a clone of the repository puts every file where the GUI, the replay
command and the report figures expect it::

    unzip demo_pack.zip -d <clone>      # macOS / Linux
    Expand-Archive demo_pack.zip <clone> # Windows

What goes in (``MANIFEST``): the final weights and configuration of the two
10 x 10 agents the demonstration plays (the CNN at gamma 0.99 and the MLP),
two early checkpoints of the CNN run for the "fumbling -> eating -> avoiding
itself" progression, the tabular table (advisor mode), the behaviour analyses
whose ``hack_*.json`` recordings the GUI replays, the regenerated report
(tables, figures, index, manifest), the Part D screenshots, the baseline
evaluations and the exported log archive for the notebook. Checkpoint
directories are NOT taken whole (2.1 GB for one CNN run); patterns marked
required abort the pack when they match nothing, the others only warn.
``scripts/demo_check.py`` verifies the unpacked result on the target machine
(``docs/DEMO-macOS.md``).
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

# (glob pattern relative to the runs directory, required)
MANIFEST: tuple[tuple[str, bool], ...] = (
    ("dqn_B8_W10_gamma099/0/config.yaml", True),
    ("dqn_B8_W10_gamma099/0/model.pt", True),
    ("dqn_B8_W10_gamma099/0/final_eval.json", False),
    ("dqn_B8_W10_gamma099/0/metrics.csv", False),
    ("dqn_B8_W10_gamma099/0/checkpoints/step_00012800.pt", False),
    ("dqn_B8_W10_gamma099/0/checkpoints/step_00140800.pt", False),
    ("dqn_A_W10/0/config.yaml", True),
    ("dqn_A_W10/0/model.pt", True),
    ("dqn_A_W10/0/final_eval.json", False),
    ("dqn_A_W10/0/metrics.csv", False),
    ("dqn_A_W10/0/checkpoints/step_00012800.pt", False),
    ("tabular_q_W10/0/**/*", True),
    ("behaviour/dqn_B8_W10_gamma099_R3_final/**/*", True),
    ("behaviour/dqn_B8_W10_gamma099_early/**/*", False),
    ("report/**/*", False),
    ("export/runs_logs.zip", False),
    ("figures/partD_*", False),
    ("baselines/*.json", False),
)


def select_files(runs_dir: Path, manifest=MANIFEST) -> tuple[list[Path], list[str], list[str]]:
    """Resolve the manifest against ``runs_dir``.

    Returns ``(files, missing_required, missing_optional)``; ``files`` are the
    matching regular files, sorted, without duplicates.
    """
    files: set[Path] = set()
    missing_required: list[str] = []
    missing_optional: list[str] = []
    for pattern, required in manifest:
        matches = [p for p in runs_dir.glob(pattern) if p.is_file()]
        if not matches:
            (missing_required if required else missing_optional).append(pattern)
        files.update(matches)
    return sorted(files), missing_required, missing_optional


def write_pack(files: list[Path], runs_dir: Path, out: Path) -> int:
    """Write ``files`` to the zip ``out`` as ``runs/<path relative to runs_dir>``; returns the byte total."""
    out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            arcname = Path("runs") / path.relative_to(runs_dir)
            zf.write(path, arcname.as_posix())
            total += path.stat().st_size
    return total


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out", default="runs/export/demo_pack.zip")
    args = p.parse_args(argv)
    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_dir():
        print(f"no runs directory at {runs_dir}", file=sys.stderr)
        return 1
    files, missing_required, missing_optional = select_files(runs_dir)
    for pattern in missing_optional:
        print(f"warning: nothing matches {pattern} (optional, skipped)")
    if missing_required:
        for pattern in missing_required:
            print(f"error: nothing matches {pattern} (required)", file=sys.stderr)
        return 1
    total = write_pack(files, runs_dir, Path(args.out))
    zipped = Path(args.out).stat().st_size
    print(f"{len(files)} files, {total / 1e6:.1f} MB -> {args.out} ({zipped / 1e6:.1f} MB zipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

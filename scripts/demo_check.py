"""Run every command of the live demonstration headless and report PASS / FAIL.

Usage::

    python scripts/demo_check.py                 # items -> demo_check/<name>.png
    python scripts/demo_check.py --skip-train    # GUI items only
    python scripts/demo_check.py --out-dir /tmp/check --runs-dir runs

Meant for the machine the demonstration runs on (``docs/DEMO-macOS.md``),
after a clone of the repository has received the demo pack
(``scripts/make_demo_pack.py``). Each GUI item is the demonstration command
with ``--headless --steps N --screenshot``, so a PASS means the player loaded,
the environment stepped and a frame rendered; open the PNGs to see what the
audience will see. The training item trains ``configs/smoke_8x8.yaml`` into
``<out-dir>/runs`` (about ten seconds on a laptop CPU) and the next item plays
the result in the GUI, which is the live-training segment of the
demonstration. Exit code 1 if anything fails; the last lines of the failing
command's output are printed.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

CNN_RUN = "dqn_B8_W10_gamma099/0"
MLP_RUN = "dqn_A_W10/0"
TABULAR_RUN = "tabular_q_W10/0"
REPLAY = "behaviour/dqn_B8_W10_gamma099_R3_final/1/hack_1_10029.json"
REPORT_FIGURES = (
    "report/figures/gamma_W10_curve.png",
    "report/figures/components_B8_W8_curve.png",
    "report/figures/score_distributions.png",
    "report/index.md",
)
SMOKE_CONFIG = "configs/smoke_8x8.yaml"
TIMEOUT_S = 600


@dataclass
class Item:
    name: str
    argv: list[str] = field(default_factory=list)   # empty: a file-existence item
    files: list[Path] = field(default_factory=list)


@dataclass
class Result:
    name: str
    ok: bool
    seconds: float
    detail: str = ""


def gui(name: str, runs_dir: Path, out_dir: Path, *args: str, steps: int) -> Item:
    argv = [sys.executable, "-m", "snake_rl.gui.app", *args, "--headless",
            "--steps", str(steps), "--screenshot", str(out_dir / f"{name}.png")]
    return Item(name, argv)


def build_items(runs_dir: Path, out_dir: Path, *, train: bool = True, train_seed: int = 0) -> list[Item]:
    """The demonstration, item by item, in the order the runbook uses them."""
    r = runs_dir
    items = [
        Item("files_report", files=[r / f for f in REPORT_FIGURES] + [r / REPLAY]),
        gui("gui_cnn_W10", r, out_dir, "--player", str(r / CNN_RUN), steps=60),
        gui("gui_cnn_W10_early_ckpt", r, out_dir,
            "--player", str(r / CNN_RUN / "checkpoints" / "step_00012800.pt"), steps=60),
        gui("gui_mlp_W10", r, out_dir, "--player", str(r / MLP_RUN), steps=60),
        gui("gui_human_cnn_advisor", r, out_dir,
            "--mode", "human", "--player", str(r / CNN_RUN), "--seed", "10000", steps=3),
        gui("gui_human_tabular_advisor", r, out_dir,
            "--mode", "human", "--player", str(r / TABULAR_RUN), steps=3),
        gui("gui_bfs", r, out_dir, "--player", "bfs", "--seed", "10000", steps=100),
        gui("gui_hamiltonian_W8", r, out_dir, "--player", "hamiltonian", "--W", "8", "--fps", "60", steps=300),
        gui("gui_replay_R3_loop", r, out_dir, "--replay", str(r / REPLAY), steps=40),
    ]
    if train:
        train_runs = out_dir / "runs"
        items.append(Item("train_smoke_8x8", [
            sys.executable, "-m", "snake_rl.train", "--config", SMOKE_CONFIG,
            "--seed", str(train_seed), "--runs-dir", str(train_runs), "--quiet"]))
        items.append(gui("gui_play_smoke_run", r, out_dir,
                         "--player", str(train_runs / "smoke_dqn_W8" / str(train_seed)), steps=100))
    return items


def run_subprocess(argv: list[str]) -> tuple[int, str]:
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=TIMEOUT_S)
    return proc.returncode, (proc.stdout + proc.stderr)


def run_item(item: Item, runner: Callable[[list[str]], tuple[int, str]] = run_subprocess) -> Result:
    t0 = time.perf_counter()
    if not item.argv:
        missing = [str(f) for f in item.files if not f.is_file()]
        return Result(item.name, not missing, time.perf_counter() - t0,
                      "" if not missing else "missing: " + ", ".join(missing))
    try:
        code, output = runner(item.argv)
    except subprocess.TimeoutExpired:
        return Result(item.name, False, time.perf_counter() - t0, f"timeout after {TIMEOUT_S} s")
    tail = "\n".join(output.strip().splitlines()[-6:])
    return Result(item.name, code == 0, time.perf_counter() - t0, "" if code == 0 else f"exit {code}\n{tail}")


def report(results: list[Result]) -> int:
    width = max(len(r.name) for r in results)
    for r in results:
        print(f"{'PASS' if r.ok else 'FAIL'}  {r.name:<{width}}  {r.seconds:6.1f} s" + (f"\n      {r.detail}" if r.detail else ""))
    failed = [r for r in results if not r.ok]
    print(f"\n{len(results) - len(failed)} / {len(results)} passed")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out-dir", default="demo_check")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--train-seed", type=int, default=0, help="the smoke run directory is never overwritten: change the seed to rerun")
    args = p.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = build_items(Path(args.runs_dir), out_dir, train=not args.skip_train, train_seed=args.train_seed)
    results: list[Result] = []
    for item in items:
        result = run_item(item)
        results.append(result)
        print(f"{'PASS' if result.ok else 'FAIL'}  {item.name}  ({result.seconds:.1f} s)", flush=True)
    print()
    return report(results)


if __name__ == "__main__":
    sys.exit(main())

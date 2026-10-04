"""Batch runner ``scripts/run_matrix.py`` (milestone M4).

The script plans and launches; the learning is in ``snake_rl.train``. The
tests therefore pin the bookkeeping a five-seed batch relies on:

* the matrix: one run per (grid point, seed), experiment names and ``--set``
  arguments for zero, one and two grid axes, ``--exp-prefix``;
* an invalid matrix (typo, bad value, value unusable in a name, repeated run)
  is refused before any process starts;
* a real batch of 2 grid points x 2 seeds with ``--jobs 2`` writes its four
  run directories, and each is identical to the same run trained alone, one
  after the other: running in parallel does not change a result;
* existing runs are never touched: ``done`` runs are skipped, ``incomplete``
  ones are reported and left exactly as they are;
* a failing process gives a ``failed`` status line and exit code 1.

Processes are real wherever the claim is about what ends up on disk. The
scheduling claims (what is started, in which order, how many at a time, what
Ctrl-C does) use ``FakeProcesses``, which replaces ``subprocess`` inside the
script, records the commands and starts nothing. Every run directory is under
``tmp_path``.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Sequence

import numpy as np
import pytest
import yaml

from snake_rl import train as train_module
from snake_rl.config import ExperimentConfig, load_config

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_matrix.py"
CONFIG = REPO / "configs" / "smoke_tabular.yaml"
BASE = "smoke_tabular"  # exp of CONFIG
#: Shrinks the smoke configuration to 20 vec-steps and three small evaluations.
TINY = ["train.total_env_steps=320", "train.eval_every=160", "train.eval_episodes=4"]
RUN_FILES = {"config.yaml", "metrics.csv", "q_table.npz", "final_eval.json"}


def _load_script() -> ModuleType:
    """``scripts/`` is not a package: import the file by its path."""
    spec = importlib.util.spec_from_file_location("run_matrix_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rm = _load_script()


# ------------------------------------------------------------------- helpers
def run_main(*argv: object) -> tuple[int, str, str]:
    """``main(argv)`` in this process -> (exit code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = rm.main([str(arg) for arg in argv])
        except SystemExit as exit_:  # argparse usage error
            code = exit_.code
    return code, out.getvalue(), err.getvalue()


def command(runs_dir: Path | str, exp: str, seed: int, overrides: Sequence[str] = ()) -> list[str]:
    """The process of one run, written out independently of the script."""
    return [
        *[sys.executable, "-m", "snake_rl.train", "--config", str(CONFIG), "--seed", str(seed), "--exp", exp],
        *(["--set", *overrides] if overrides else []),
        *["--runs-dir", str(runs_dir), "--quiet"],
    ]


def config_of(cmd: list[str]) -> ExperimentConfig:
    """The configuration ``snake_rl.train`` builds from a command: its
    ``--set`` overrides in order, then ``--exp``."""
    sets = cmd[cmd.index("--set") + 1 : cmd.index("--runs-dir")] if "--set" in cmd else []
    exp = cmd[cmd.index("--exp") + 1]
    return load_config(cmd[cmd.index("--config") + 1], [*sets, f"exp={json.dumps(exp)}"])


def label(runs_dir: Path, exp: str, seed: int) -> str:
    return (runs_dir / exp / str(seed)).as_posix()


STATUS_LINE = re.compile(r"(?P<run>.+): (?P<status>ok|done|incomplete|failed \(exit code -?\d+\))")


def statuses(out: str) -> dict[str, str]:
    """The final status lines of a batch, ``{run directory: status}`` in
    printed order. (The progress lines end with the run time and do not match.)"""
    found: dict[str, str] = {}
    for line in out.splitlines():
        match = STATUS_LINE.fullmatch(line)
        if match:
            assert match["run"] not in found, f"two status lines for {match['run']}"
            found[match["run"]] = match["status"]
    return found


def train_run(runs_dir: Path, exp: str, seed: int, overrides: Sequence[str] = TINY) -> Path:
    """Train one run in this process through the command line of ``snake_rl.train``."""
    assert train_module.main(command(runs_dir, exp, seed, overrides)[3:]) == 0
    return runs_dir / exp / str(seed)


def snapshot(root: Path) -> dict[str, tuple[int, bytes] | None]:
    """Everything under ``root``: modification time and content of each file
    (``None`` for a directory)."""
    return {
        path.relative_to(root).as_posix(): (path.stat().st_mtime_ns, path.read_bytes()) if path.is_file() else None
        for path in sorted(root.rglob("*"))
    }


def assert_same_run(run_dir: Path, reference: Path) -> None:
    """Same Q-table, same final evaluation episodes, same learning curve
    (every column of metrics.csv except the wall time, which is the last)."""
    got, ref = np.load(run_dir / "q_table.npz"), np.load(reference / "q_table.npz")
    assert sorted(got.files) == sorted(ref.files)
    for name in ref.files:
        np.testing.assert_array_equal(got[name], ref[name], err_msg=f"{run_dir}: {name}")
    episodes = [json.loads((d / "final_eval.json").read_text())["episodes"] for d in (run_dir, reference)]
    assert episodes[0] == episodes[1]
    curves = [
        [line.rsplit(",", 1)[0] for line in (d / "metrics.csv").read_text().splitlines()] for d in (run_dir, reference)
    ]
    assert curves[0] == curves[1] and len(curves[0]) == 4  # header + 3 evaluations


class FakeProcesses:
    """Stands in for the ``subprocess`` module inside the script: records
    every command and starts nothing. A fake process is alive for ``polls``
    looks of the scheduler, then exits with the code listed for its seed
    (0 if none is listed)."""

    list2cmdline = staticmethod(subprocess.list2cmdline)

    def __init__(self) -> None:
        self.polls = 2
        self.exit_codes: dict[int, int] = {}  # seed -> exit code
        self.commands: list[list[str]] = []  # every command started, in order
        self.killed: list[list[str]] = []
        self.alive = 0
        self.max_alive = 0

    def Popen(self, cmd: list[str]) -> "_FakeProcess":
        self.commands.append(list(cmd))
        self.alive += 1
        self.max_alive = max(self.max_alive, self.alive)
        return _FakeProcess(self, list(cmd))


class _FakeProcess:
    def __init__(self, owner: FakeProcesses, cmd: list[str]) -> None:
        self.owner, self.cmd = owner, cmd
        self.polls_left = owner.polls
        self.returncode: int | None = None

    def _exit(self, code: int) -> None:
        self.returncode = code
        self.owner.alive -= 1

    def poll(self) -> int | None:
        if self.returncode is None:
            self.polls_left -= 1
            if self.polls_left <= 0:
                seed = int(self.cmd[self.cmd.index("--seed") + 1])
                self._exit(self.owner.exit_codes.get(seed, 0))
        return self.returncode

    def kill(self) -> None:
        self.owner.killed.append(self.cmd)
        if self.returncode is None:
            self._exit(-9)

    def wait(self) -> int | None:
        return self.returncode


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeProcesses:
    """From here on the script cannot start a process; it talks to a recorder."""
    processes = FakeProcesses()
    monkeypatch.setattr(rm, "subprocess", processes)
    monkeypatch.setattr(rm, "POLL_SECONDS", 0.0)
    return processes


# ------------------------------------------------- the matrix and its names
#: (--grid, --set, --exp-prefix, expected (experiment name, --set arguments) per grid point in order)
MATRICES = [
    pytest.param([], [], None, [(BASE, [])], id="no grid: the name of the config file, no --set"),
    pytest.param(
        [],
        ["agent.gamma=0.9", "train.n_envs=8"],
        None,
        [(BASE, ["agent.gamma=0.9", "train.n_envs=8"])],
        id="no grid: --set is passed on",
    ),
    pytest.param(
        ["agent.alpha=0.05,0.1,0.2"],
        [],
        None,
        [(f"{BASE}__alpha={a}", [f"agent.alpha={a}"]) for a in ("0.05", "0.1", "0.2")],
        id="one axis",
    ),
    pytest.param(
        ["agent.alpha=0.1,0.5", "env.reward.variant=R0,R1,R3"],
        ["agent.gamma=0.9"],
        None,
        [
            (f"{BASE}__alpha={a}__variant={v}", ["agent.gamma=0.9", f"agent.alpha={a}", f"env.reward.variant={v}"])
            for a in ("0.1", "0.5")
            for v in ("R0", "R1", "R3")
        ],
        id="two axes: first axis slowest, grid values after --set",
    ),
    pytest.param(
        ["env.reward.variant=R1,R0", "agent.alpha=0.5,0.1"],
        [],
        None,
        [
            (f"{BASE}__variant={v}__alpha={a}", [f"env.reward.variant={v}", f"agent.alpha={a}"])
            for v in ("R1", "R0")
            for a in ("0.5", "0.1")
        ],
        id="axes and values in the order given",
    ),
    pytest.param(
        ["env.starve_limit=null,50", "agent.alpha=1.0e-2"],
        [],
        None,
        [
            (f"{BASE}__starve_limit={h}__alpha=1.0e-2", [f"env.starve_limit={h}", "agent.alpha=1.0e-2"])
            for h in ("null", "50")
        ],
        id="the value text as typed",
    ),
    pytest.param([], [], "sweep.v2", [("sweep.v2", [])], id="--exp-prefix without a grid"),
    pytest.param(
        ["agent.alpha=0.1,0.5"],
        ["train.n_envs=8"],
        "sweep.v2",
        [(f"sweep.v2__alpha={a}", ["train.n_envs=8", f"agent.alpha={a}"]) for a in ("0.1", "0.5")],
        id="--exp-prefix replaces the base name",
    ),
    pytest.param(
        ["agent.alpha=0.1,0.5"],
        ["exp=renamed"],
        None,
        [(f"renamed__alpha={a}", ["exp=renamed", f"agent.alpha={a}"]) for a in ("0.1", "0.5")],
        id="--set exp= is the base name",
    ),
    pytest.param(
        ["agent.alpha=0.1,0.5"],
        ["exp=renamed"],
        "sweep.v2",
        [(f"sweep.v2__alpha={a}", ["exp=renamed", f"agent.alpha={a}"]) for a in ("0.1", "0.5")],
        id="--exp-prefix wins over --set exp=",
    ),
]


@pytest.mark.parametrize(("grid", "sets", "prefix", "points"), MATRICES)
def test_one_command_per_grid_point_and_seed_with_the_right_name_and_overrides(
    tmp_path, fake, grid, sets, prefix, points
):
    runs_dir = tmp_path / "runs"
    seeds = [4, 0, 11]
    expected = [command(runs_dir, exp, seed, overrides) for exp, overrides in points for seed in seeds]

    runs = rm.plan_runs(CONFIG, seeds, grid, sets, prefix, runs_dir)
    assert [run.command for run in runs] == expected
    assert [(run.exp, run.seed) for run in runs] == [(exp, seed) for exp, _ in points for seed in seeds]
    assert [run.run_dir for run in runs] == [runs_dir / exp / str(seed) for exp, _ in points for seed in seeds]
    assert {run.status for run in runs} == {"pending"}
    assert len({run.label for run in runs}) == len(points) * len(seeds)  # no two runs share a directory

    argv = ["--config", CONFIG, "--seeds", *seeds, "--runs-dir", runs_dir, "--dry-run"]
    argv += (["--grid", *grid] if grid else []) + (["--set", *sets] if sets else [])
    argv += ["--exp-prefix", prefix] if prefix else []
    code, out, err = run_main(*argv)
    assert (code, err) == (0, "")
    assert out.splitlines() == [rm.format_command(cmd) for cmd in expected]  # exactly one line per run
    assert fake.commands == []  # a dry run starts nothing ...
    assert list(tmp_path.iterdir()) == []  # ... and creates nothing


def test_each_command_trains_the_grid_point_its_name_promises(tmp_path):
    grid = ["agent.alpha=0.1,0.5", "env.reward.variant=R0,R1"]
    runs = rm.plan_runs(CONFIG, [0], grid, ["agent.alpha=0.3", "agent.gamma=0.9", *TINY], None, tmp_path)
    seen = set()
    for run in runs:
        cfg = config_of(run.command)
        assert cfg.exp == run.exp == f"{BASE}__alpha={cfg.agent.alpha}__variant={cfg.env.reward.variant}"
        assert cfg.agent.gamma == 0.9 and cfg.train.total_env_steps == 320  # --set reaches every run
        seen.add((cfg.agent.alpha, cfg.env.reward.variant))  # the grid value wins over --set agent.alpha=0.3
    assert seen == {(0.1, "R0"), (0.1, "R1"), (0.5, "R0"), (0.5, "R1")}


def test_repeated_set_and_grid_options_accumulate(tmp_path, fake):
    argv = ["--config", CONFIG, "--seeds", 0, "--runs-dir", tmp_path / "runs", "--dry-run"]
    argv += ["--set", "agent.gamma=0.9", "--grid", "agent.alpha=0.1,0.5", "--set", *TINY, "--grid", "train.n_envs=8,16"]
    code, out, err = run_main(*argv)
    sets = ["agent.gamma=0.9", *TINY]  # all --set first, then the axes in the order given
    names = {(a, n): f"{BASE}__alpha={a}__n_envs={n}" for a in ("0.1", "0.5") for n in ("8", "16")}
    expected = [
        command(tmp_path / "runs", name, 0, [*sets, f"agent.alpha={a}", f"train.n_envs={n}"])
        for (a, n), name in names.items()
    ]
    assert (code, err) == (0, "")
    assert out.splitlines() == [rm.format_command(cmd) for cmd in expected]


def test_default_runs_dir_is_runs_under_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # planning looks at ./runs: keep it away from the repo
    (run,) = rm.plan_runs(CONFIG, [7])
    assert run.run_dir == Path("runs") / BASE / "7"
    assert run.command == command("runs", BASE, 7)
    assert list(tmp_path.iterdir()) == []


# ------------------------------------------- refused before anything starts
#: (what is wrong, the extra arguments, what the error message must mention)
INVALID = [
    ("typo in a grid key", ["--grid", "agent.alhpa=0.1,0.2"], ["agent.alhpa"]),
    ("bad value in the last point", ["--grid", "agent.alpha=0.1,0.2,7"], ["grid point agent.alpha=7", "agent.alpha"]),
    (
        "bad combination in one point",
        ["--grid", "agent.alpha=0.1,0.5", "train.n_envs=16,7"],
        ["grid point agent.alpha=0.1 train.n_envs=7", "train.eval_every"],
    ),
    ("tabular Q needs observation A", ["--grid", "env.obs.kind=A,B"], ["grid point env.obs.kind=B", "env.obs.kind"]),
    ("YAML scientific notation", ["--grid", "agent.alpha=1e-3"], ["agent.alpha", "1.0e-3"]),
    ("bad --set", ["--set", "train.n_envs=0"], ["train.n_envs"]),
    ("typo in --set", ["--set", "trian.n_envs=16"], ["trian"]),
    ("value unusable in a name", ["--grid", "env.starve_limit=50,~"], ["exp", f"{BASE}__starve_limit=~"]),
    ("value with a space", ["--grid", "env.reward.variant=R0,R 1"], ["grid point env.reward.variant=R 1"]),
    ("bad prefix", ["--exp-prefix", "my sweep", "--grid", "agent.alpha=0.1"], ["exp", "my sweep__alpha=0.1"]),
    ("axis without =", ["--grid", "agent.alpha"], ["--grid 'agent.alpha'"]),
    ("axis without a key", ["--grid", "=0.1,0.2"], ["--grid '=0.1,0.2'"]),
    ("axis with an empty key part", ["--grid", "agent..alpha=0.1"], ["--grid 'agent..alpha=0.1'"]),
    ("axis without values", ["--grid", "agent.alpha="], ["--grid 'agent.alpha='"]),
    ("axis with an empty value", ["--grid", "agent.alpha=0.1,,0.2"], ["--grid 'agent.alpha=0.1,,0.2'"]),
    ("exp is not a grid axis", ["--grid", "exp=a,b"], ["--exp-prefix"]),
    ("axis twice", ["--grid", "agent.alpha=0.1", "agent.alpha=0.2"], ["more than one axis", "agent.alpha"]),
    ("value twice", ["--grid", "agent.alpha=0.1,0.5,0.1"], ["more than once", f"{BASE}__alpha=0.1/0"]),
    ("seed twice", ["--seeds", "0", "1", "0"], ["more than once", f"{BASE}/0"]),
    ("negative seed", ["--seeds", "0", "-1"], ["--seeds"]),
    ("seed that is not an integer", ["--seeds", "0", "1.5"], ["--seeds"]),
    ("no jobs", ["--jobs", "0"], ["--jobs"]),
    ("missing config file", ["--config", "no_such_file.yaml"], ["no_such_file.yaml"]),
    ("unknown option", ["--girdd", "agent.alpha=0.1"], ["--girdd"]),
]


@pytest.mark.parametrize(("extra", "mentions"), [pytest.param(extra, texts, id=what) for what, extra, texts in INVALID])
def test_an_invalid_matrix_is_refused_before_anything_starts(tmp_path, fake, extra, mentions):
    runs_dir = tmp_path / "runs"
    for mode in ([], ["--dry-run"]):
        # Later options win in argparse, so ``extra`` can replace --config and --seeds.
        code, out, err = run_main("--config", CONFIG, "--seeds", 0, 1, "--runs-dir", runs_dir, *mode, *extra)
        assert code == 2
        assert "error" in err
        for text in mentions:
            assert text in err
        assert out == ""
        assert fake.commands == []
        assert list(tmp_path.iterdir()) == []


def test_missing_required_options_are_usage_errors(tmp_path, fake):
    assert run_main("--config", CONFIG)[0] == 2  # five seeds are typed, never implied
    assert run_main("--seeds", 0)[0] == 2
    assert fake.commands == []


# ------------------------------------------------------------ a real batch
SEEDS = (3, 1)  # not 0, 1: a run must be named by its seed, not by its position
ALPHAS = ("0.1", "0.5")


@pytest.fixture(scope="module")
def batch(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """2 grid points x 2 seeds, four real processes, two at a time."""
    runs_dir = tmp_path_factory.mktemp("batch") / "runs"
    argv = ["--config", CONFIG, "--seeds", *SEEDS, "--grid", f"agent.alpha={','.join(ALPHAS)}", "--set", *TINY]
    argv += ["--runs-dir", runs_dir, "--jobs", 2]
    code, out, err = run_main(*argv)
    runs = [(f"{BASE}__alpha={alpha}", alpha, seed) for alpha in ALPHAS for seed in SEEDS]
    return SimpleNamespace(argv=argv, runs_dir=runs_dir, code=code, out=out, err=err, runs=runs)


def test_batch_writes_one_finished_run_directory_per_run(batch):
    assert batch.code == 0, batch.err
    assert statuses(batch.out) == {label(batch.runs_dir, exp, seed): "ok" for exp, _, seed in batch.runs}
    assert "4 runs: 4 ok" in batch.out.splitlines()
    assert {path.name for path in batch.runs_dir.iterdir()} == {exp for exp, _, _ in batch.runs}
    for exp, alpha, seed in batch.runs:
        run_dir = batch.runs_dir / exp / str(seed)
        assert {path.name for path in run_dir.iterdir()} == RUN_FILES
        record = yaml.safe_load((run_dir / "config.yaml").read_text())
        assert record["exp"] == exp
        assert record["agent"]["alpha"] == float(alpha)  # the grid override of this point
        assert record["train"]["total_env_steps"] == 320 and record["train"]["eval_episodes"] == 4  # --set
        assert record["run"]["seed"] == seed
        assert record["run"]["argv"] == command(batch.runs_dir, exp, seed, [*TINY, f"agent.alpha={alpha}"])[3:]


def test_batch_runs_equal_the_same_runs_trained_one_after_the_other(batch, tmp_path):
    for exp, alpha, seed in batch.runs:
        reference = train_run(tmp_path / "sequential", exp, seed, [*TINY, f"agent.alpha={alpha}"])
        assert_same_run(batch.runs_dir / exp / str(seed), reference)

    # The comparison can fail: seeds and grid points give different tables.
    tables = [np.load(batch.runs_dir / exp / str(seed) / "q_table.npz") for exp, _, seed in batch.runs]
    assert all(int(np.count_nonzero(table["visits"])) > 50 for table in tables)
    for i, first in enumerate(tables):
        for second in tables[i + 1 :]:
            assert not np.array_equal(first["q"], second["q"])


def test_a_second_invocation_reports_done_and_starts_nothing(batch, fake):
    before = snapshot(batch.runs_dir)
    code, out, err = run_main(*batch.argv)
    assert code == 0
    assert statuses(out) == {label(batch.runs_dir, exp, seed): "done" for exp, _, seed in batch.runs}
    assert "4 runs: 4 done" in out.splitlines()
    assert fake.commands == []
    assert snapshot(batch.runs_dir) == before  # same files, same modification times, same bytes


def test_a_later_invocation_starts_only_the_missing_runs(batch, fake):
    before = snapshot(batch.runs_dir)
    argv = [str(arg) for arg in batch.argv]
    argv[argv.index("--seeds") + 1 : argv.index("--grid")] = ["1", "2", "3"]  # seed 2 is new
    new = [command(batch.runs_dir, f"{BASE}__alpha={alpha}", 2, [*TINY, f"agent.alpha={alpha}"]) for alpha in ALPHAS]

    code, out, err = run_main(*argv, "--dry-run")
    assert code == 0
    assert out.splitlines() == [rm.format_command(cmd) for cmd in new]  # commands of the pending runs only
    assert [line.rsplit(": ", 1)[1] for line in err.splitlines()] == ["done"] * 4
    assert fake.commands == []

    code, out, err = run_main(*argv)
    assert code == 0
    assert fake.commands == new
    assert list(statuses(out).values()) == ["done", "ok", "done", "done", "ok", "done"]  # seeds 1 2 3 per grid point
    assert snapshot(batch.runs_dir) == before


# --------------------------------------------- existing runs are left alone
def test_incomplete_runs_are_reported_not_started_and_not_touched(tmp_path, fake):
    runs_dir = tmp_path / "runs"
    for seed, leftovers in ((0, {"config.yaml"}), (3, RUN_FILES - {"final_eval.json"})):
        run_dir = train_run(runs_dir, BASE, seed)
        for name in RUN_FILES - leftovers:  # what a run interrupted early / late leaves behind
            (run_dir / name).unlink()
    train_run(runs_dir, BASE, 2)
    before = snapshot(runs_dir)

    argv = ["--config", CONFIG, "--seeds", 0, 1, 2, 3, "--set", *TINY, "--runs-dir", runs_dir, "--jobs", 4]
    code, out, err = run_main(*argv)
    assert statuses(out) == {
        label(runs_dir, BASE, 0): "incomplete",
        label(runs_dir, BASE, 1): "ok",
        label(runs_dir, BASE, 2): "done",
        label(runs_dir, BASE, 3): "incomplete",
    }
    assert "4 runs: 1 ok, 1 done, 2 incomplete" in out.splitlines()
    assert fake.commands == [command(runs_dir, BASE, 1, TINY)]  # the one missing run, nothing else
    assert code == 0  # only a failed run changes the exit code
    assert snapshot(runs_dir) == before  # nothing modified, nothing deleted, nothing added

    code, out, err = run_main(*argv, "--dry-run")
    assert out.splitlines() == [rm.format_command(command(runs_dir, BASE, 1, TINY))]
    assert [line.rsplit(": ", 1)[1] for line in err.splitlines()] == ["incomplete", "done", "incomplete"]
    assert snapshot(runs_dir) == before


def test_an_existing_run_of_another_configuration_stops_the_batch(tmp_path, fake):
    runs_dir = tmp_path / "runs"
    train_run(runs_dir, BASE, 1)  # trained with the gamma of the config file, 0.99
    before = snapshot(runs_dir)
    argv = ["--config", CONFIG, "--seeds", 0, 1, "--set", *TINY, "agent.gamma=0.9", "--runs-dir", runs_dir]

    code, out, err = run_main(*argv)
    assert code == 2 and out == ""
    assert label(runs_dir, BASE, 1) in err
    assert "agent.gamma: 0.99 there, 0.9 requested" in err and "--exp-prefix" in err
    assert fake.commands == []  # seed 0 is not started either: one name, one configuration
    assert snapshot(runs_dir) == before

    code, out, err = run_main(*argv, "--exp-prefix", "gamma09")  # under its own name the batch is fine
    assert code == 0
    assert fake.commands == [command(runs_dir, "gamma09", seed, [*TINY, "agent.gamma=0.9"]) for seed in (0, 1)]


# ------------------------------------------------------------------ failures
def test_a_failing_run_gets_a_failed_status_and_the_exit_code_is_1(tmp_path):
    runs_dir = tmp_path / "runs"
    (runs_dir / BASE).mkdir(parents=True)
    (runs_dir / BASE / "1").write_text("in the way")  # seed 1 cannot create its run directory

    argv = ["--config", CONFIG, "--seeds", 0, 1, 2, "--set", *TINY, "--runs-dir", runs_dir, "--jobs", 3]
    code, out, err = run_main(*argv)
    assert statuses(out) == {
        label(runs_dir, BASE, 0): "ok",
        label(runs_dir, BASE, 1): "failed (exit code 1)",
        label(runs_dir, BASE, 2): "ok",
    }
    assert "3 runs: 2 ok, 1 failed" in out.splitlines()
    assert code == 1
    assert (runs_dir / BASE / "0" / "final_eval.json").exists() and (runs_dir / BASE / "2" / "final_eval.json").exists()
    assert (runs_dir / BASE / "1").read_text() == "in the way"


def test_a_runs_dir_that_is_a_file_fails_the_run(tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.write_text("not a directory")
    code, out, err = run_main("--config", CONFIG, "--seeds", 0, "--set", *TINY, "--runs-dir", runs_dir)
    assert statuses(out) == {label(runs_dir, BASE, 0): "failed (exit code 1)"}
    assert code == 1
    assert runs_dir.read_text() == "not a directory"


def test_the_exit_code_of_each_process_decides_its_status(tmp_path, fake):
    fake.exit_codes = {1: 3, 2: -9}
    runs_dir = tmp_path / "runs"
    code, out, err = run_main("--config", CONFIG, "--seeds", 0, 1, 2, 3, "--runs-dir", runs_dir, "--jobs", 2)
    assert statuses(out) == {
        label(runs_dir, BASE, 0): "ok",
        label(runs_dir, BASE, 1): "failed (exit code 3)",
        label(runs_dir, BASE, 2): "failed (exit code -9)",
        label(runs_dir, BASE, 3): "ok",
    }
    assert "4 runs: 2 ok, 2 failed" in out.splitlines()
    assert code == 1
    assert len(fake.commands) == 4  # a failure does not stop the rest of the batch


# ---------------------------------------------------------------- scheduling
@pytest.mark.parametrize(("jobs", "at_a_time"), [(None, 1), (1, 1), (2, 2), (4, 4), (6, 6), (9, 6)])
def test_at_most_jobs_processes_run_at_a_time_in_matrix_order(tmp_path, fake, jobs, at_a_time):
    runs_dir = tmp_path / "runs"
    argv = ["--config", CONFIG, "--seeds", 5, 0, 2, "--grid", "agent.alpha=0.5,0.1", "--runs-dir", runs_dir]
    code, out, err = run_main(*argv, *([] if jobs is None else ["--jobs", jobs]))
    assert code == 0
    assert fake.max_alive == at_a_time  # never more than --jobs, and the slots are used
    assert fake.commands == [
        command(runs_dir, f"{BASE}__alpha={alpha}", seed, [f"agent.alpha={alpha}"])
        for alpha in ("0.5", "0.1")
        for seed in (5, 0, 2)
    ]
    assert list(statuses(out).values()) == ["ok"] * 6
    assert fake.alive == 0 and fake.killed == []


def test_ctrl_c_stops_the_launching_and_kills_the_processes_in_flight(tmp_path, fake, monkeypatch):
    def interrupted(seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(rm, "time", SimpleNamespace(sleep=interrupted, perf_counter=time.perf_counter))
    runs_dir = tmp_path / "runs"
    code, out, err = run_main("--config", CONFIG, "--seeds", 0, 1, 2, 3, 4, "--runs-dir", runs_dir, "--jobs", 2)
    in_flight = [command(runs_dir, BASE, seed) for seed in (0, 1)]
    assert code == 130
    assert fake.commands == in_flight  # seeds 2, 3, 4 were never started
    assert fake.killed == in_flight and fake.alive == 0
    assert "interrupted" in err


# ------------------------------------------------------- the script itself
def test_the_script_runs_as_a_script_from_any_directory(tmp_path):
    runs_dir = tmp_path / "runs"
    argv = [sys.executable, str(SCRIPT), "--config", str(CONFIG), "--seeds", "0", "1", "--runs-dir", str(runs_dir)]

    dry = subprocess.run(
        [*argv, "--grid", "agent.alpha=0.1,0.5", "--dry-run"], capture_output=True, text=True, cwd=tmp_path
    )
    assert dry.returncode == 0, dry.stderr
    assert dry.stdout.splitlines() == [
        rm.format_command(command(runs_dir, f"{BASE}__alpha={alpha}", seed, [f"agent.alpha={alpha}"]))
        for alpha in ("0.1", "0.5")
        for seed in (0, 1)
    ]
    assert dry.stderr == ""

    typo = subprocess.run([*argv, "--grid", "agent.alhpa=0.1,0.5"], capture_output=True, text=True, cwd=tmp_path)
    assert typo.returncode == 2
    assert "agent.alhpa" in typo.stderr and typo.stdout == ""
    assert list(tmp_path.iterdir()) == []


def test_a_printed_command_trains_that_run_when_given_to_the_shell(tmp_path, fake):
    runs_dir = tmp_path / "runs"
    argv = ["--config", CONFIG, "--seeds", 5, "--grid", "agent.alpha=0.5", "--set", *TINY, "--runs-dir", runs_dir]
    code, out, err = run_main(*argv, "--dry-run")
    (line,) = out.splitlines()

    assert subprocess.run(line, shell=True, cwd=tmp_path).returncode == 0  # the real subprocess, not the recorder
    run_dir = runs_dir / f"{BASE}__alpha=0.5" / "5"
    assert {path.name for path in run_dir.iterdir()} == RUN_FILES
    record = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert (record["exp"], record["agent"]["alpha"], record["run"]["seed"]) == (f"{BASE}__alpha=0.5", 0.5, 5)

    code, out, err = run_main(*argv)  # and the batch runner counts it as done
    assert statuses(out) == {run_dir.as_posix(): "done"}
    assert fake.commands == []

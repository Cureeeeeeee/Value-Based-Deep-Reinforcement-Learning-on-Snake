"""``scripts/bench_update.py`` (PLAN.md milestone M6, spec D7): the timing of
one gradient step and one ``act_epsilon`` call of a configuration on a device,
used for the device decision of the CNN files. Two tests only: ``--help`` and
one tiny CPU invocation that writes nothing.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "bench_update.py"
CONFIG = REPO_ROOT / "configs" / "dqn_B_8x8.yaml"
#: ``<name>: <number> ms`` for the two measurements.
MEASUREMENT = re.compile(r"\b(update|act):\s*([0-9]+(?:\.[0-9]+)?)\s*ms\b")


def run(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=cwd, capture_output=True, text=True, timeout=300)


def test_help(tmp_path):
    done = run("--help", cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    for flag in ("--config", "--set", "--device", "--threads", "--updates", "--warmup", "--seed"):
        assert flag in done.stdout, flag
    assert list(tmp_path.iterdir()) == []


def test_tiny_cpu_invocation_prints_the_two_timings_and_writes_nothing(tmp_path):
    """The shipped CNN file shrunk with ``--set``: a few boards, a tiny CNN, a
    small batch; two warm-up and three timed updates on the CPU with one thread."""
    done = run(
        "--config", str(CONFIG), "--set", "train.n_envs=4", "agent.conv_channels=[2]", "agent.hidden=[8]",
        "agent.batch_size=8", "agent.buffer_size=8192", "agent.learning_starts=8", "--device", "cpu",
        "--threads", "1", "--updates", "3", "--warmup", "2", "--seed", "0", cwd=tmp_path,
    )
    assert done.returncode == 0, done.stderr
    found = {name: float(value) for name, value in MEASUREMENT.findall(done.stdout)}
    assert set(found) == {"update", "act"}, done.stdout
    assert all(value >= 0.0 for value in found.values())
    assert list(tmp_path.iterdir()) == []  # nothing written next to the caller (no runs/ directory)
    assert not (REPO_ROOT / "runs" / "dqn_B8_W8" / "bench").exists()

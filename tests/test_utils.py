"""``snake_rl.utils.git_hash``: resolved once per process.

A run records the code it started from. Training evaluates many times and
every evaluation result carries the hash, so asking git each time would cost
two processes per evaluation and could stamp a later evaluation of the same
run with ``+dirty`` (or a newer commit) when the tree is edited while a long
run trains.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import snake_rl.utils as utils
from snake_rl.config import load_config
from snake_rl.train import train

REPO = Path(__file__).resolve().parents[1]
SMOKE = REPO / "configs" / "smoke_tabular.yaml"
TINY = ["train.total_env_steps=320", "train.eval_every=160", "train.eval_episodes=4"]


def test_git_is_asked_once_per_process(monkeypatch, tmp_path):
    spawned: list[list[str]] = []
    real = subprocess.check_output

    def counting(args, *a, **k):
        spawned.append(list(args))
        return real(args, *a, **k)

    monkeypatch.setattr(utils.subprocess, "check_output", counting)
    utils.git_hash.cache_clear()
    try:
        train(load_config(SMOKE, TINY), 0, tmp_path, quiet=True)  # 3 evaluations + the run record
        # One git_hash(): rev-parse + status in a repository, rev-parse alone (it fails) outside one.
        assert 1 <= len(spawned) <= 2 and all(args[0] == "git" for args in spawned)
        assert len(spawned) == 1 or "--no-optional-locks" in spawned[1]
        first = utils.git_hash()
        assert utils.git_hash() == first and 1 <= len(spawned) <= 2  # cached: no further process
    finally:
        utils.git_hash.cache_clear()

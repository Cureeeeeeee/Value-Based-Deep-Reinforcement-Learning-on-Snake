"""Small shared helpers (reproducibility bookkeeping)."""

from __future__ import annotations

import functools
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@functools.lru_cache(maxsize=None)
def git_hash(short: bool = True) -> str:
    """Commit hash (with ``+dirty`` if the tree has changes), or ``"unknown"``.

    Resolved once per process, at the first call: a run records the code it
    started from, also when the tree changes while it trains.
    """
    try:
        args = ["git", "rev-parse", "--short" if short else "--verify", "HEAD"]
        h = subprocess.check_output(args, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(
            ["git", "--no-optional-locks", "status", "--porcelain"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        )
        return h + ("+dirty" if status.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"

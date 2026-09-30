"""Small shared helpers (reproducibility bookkeeping)."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def git_hash(short: bool = True) -> str:
    """Current commit hash (with ``+dirty`` if the tree has changes), or ``"unknown"``."""
    try:
        args = ["git", "rev-parse", "--short" if short else "--verify", "HEAD"]
        h = subprocess.check_output(args, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        )
        return h + ("+dirty" if status.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"

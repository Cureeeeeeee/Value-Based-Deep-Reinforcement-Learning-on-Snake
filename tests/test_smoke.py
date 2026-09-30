"""M0 smoke test: the package and every module of the PLAN.md section 2 layout import."""

from __future__ import annotations

import importlib

import pytest

MODULES = [
    "snake_rl",
    "snake_rl.env",
    "snake_rl.env.common",
    "snake_rl.env.core",
    "snake_rl.env.obs",
    "snake_rl.env.rewards",
    "snake_rl.env.vec_env",
    "snake_rl.agents",
    "snake_rl.agents.baselines",
    "snake_rl.agents.tabular_q",
    "snake_rl.agents.dqn",
    "snake_rl.agents.dqn.networks",
    "snake_rl.agents.dqn.replay",
    "snake_rl.agents.dqn.agent",
    "snake_rl.train",
    "snake_rl.evaluate",
    "snake_rl.analysis",
    "snake_rl.analysis.aggregate",
    "snake_rl.analysis.plots",
    "snake_rl.gui",
    "snake_rl.gui.app",
]


@pytest.mark.parametrize("name", MODULES)
def test_import(name):
    importlib.import_module(name)


def test_version():
    import snake_rl

    assert snake_rl.__version__

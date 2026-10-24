"""``notebooks/results.ipynb`` (M10 spec section 6; Part E): the notebook that
carries the code of the study, trains in the session and regenerates every
table and figure of the report from the training logs.

``scripts/build_notebook.py`` writes the notebook from the package sources:
the environment, the players, the DQN, the experiment configuration, the
evaluation protocol and the training loop are the modules of ``snake_rl``
inlined one per code cell (module docstring, import statements and the
command-line wrappers removed, every import gathered in one cell). The
committed notebook must equal the builder's output, so it cannot drift from
the code; the inlined cells must contain the module text unchanged (lines
removed, none rewritten).

``nbformat`` is not a dependency of the project, so the file is read as
JSON and the nbformat-4 rules that matter are checked by hand: every cell
has a type, an id and a non-empty source; code cells are committed clean
(no outputs, no execution count); the code calls the report pipeline
(``snake_rl.analysis.matrix``, ``snake_rl.analysis.report``,
``snake_rl.analysis.report_tables``), reads the logs from ``runs_logs.zip``
and clones the repository; every ``Table N.`` and ``Figure N.`` of
``docs/PartE-final-report.md`` is shown. The executed copy under
``notebooks/executed/`` must carry no error, the three training runs, the
environment equivalence check and the check cell's verdict.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "results.ipynb"
EXECUTED = REPO / "notebooks" / "executed" / "results.ipynb"
REPORT_MD = REPO / "docs" / "PartE-final-report.md"
BUILDER = REPO / "scripts" / "build_notebook.py"
REPO_URL = "https://github.com/Cureeeeeeee/Value-Based-Deep-Reinforcement-Learning-on-Snake.git"
VERDICT = "RESULT: every table and figure of the report is reproduced"
#: Classes and functions of the study that must be defined by the notebook's own cells.
STUDY_NAMES = (
    "class SnakeEnv", "class VecSnakeEnv", "def encode", "def compute_reward", "class RandomPolicy",
    "class BFSGreedyPolicy", "class HamiltonianPolicy", "class TabularQAgent", "class DQNConfig",
    "class ReplayBuffer", "class PrioritizedReplayBuffer", "class MLP", "class CNN", "class DuelingMLP",
    "def build_q_network", "class DQNAgent", "def load_config", "def evaluate(", "def evaluate_batched",
    "def train(", "def _train_tabular_q", "def _train_dqn", "def linear_schedule",
)


def _load_builder() -> ModuleType:
    """``scripts/`` is not a package: import the file by its path."""
    spec = importlib.util.spec_from_file_location("build_notebook_script", BUILDER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = _load_builder()


@pytest.fixture(scope="module")
def nb() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def source_of(cell: dict) -> str:
    source = cell["source"]
    return "".join(source) if isinstance(source, list) else source


def code_cells(nb: dict) -> list[str]:
    return [source_of(c) for c in nb["cells"] if c["cell_type"] == "code"]


def code_of(nb: dict) -> str:
    return "\n".join(code_cells(nb))


def report_items() -> list[str]:
    """``Table N`` / ``Figure N`` names of every caption of the report source, in order."""
    names = []
    for line in REPORT_MD.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^((?:Table|Figure) [A-Z]?\d+)\.", line)
        if m:
            names.append(m.group(1))
    return names


# ------------------------------------------------------------------ the file
def test_the_file_is_nbformat_4_json(nb):
    assert NOTEBOOK.is_file()
    assert nb["nbformat"] == 4
    assert nb["nbformat_minor"] == 5
    assert nb["metadata"]["kernelspec"]["name"] == "python3"  # Colab's kernel
    assert isinstance(nb["cells"], list) and len(nb["cells"]) >= 60


def test_every_cell_is_well_formed(nb):
    ids = []
    for cell in nb["cells"]:
        assert cell["cell_type"] in ("markdown", "code")
        assert source_of(cell).strip() != ""
        assert isinstance(cell["id"], str) and cell["id"]
        assert isinstance(cell["metadata"], dict)
        ids.append(cell["id"])
    assert len(set(ids)) == len(ids)


def test_code_cells_are_committed_clean(nb):
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(cells) >= 40
    for cell in cells:
        assert cell["outputs"] == []
        assert cell["execution_count"] is None


# ------------------------------------------------------------------ the code of the study
def test_the_notebook_is_the_builders_output():
    assert builder.render(builder.build()) == NOTEBOOK.read_text(encoding="utf-8"), \
        "notebooks/results.ipynb differs from the package sources: run python scripts/build_notebook.py"


def test_every_module_of_the_study_is_one_code_cell_in_order(nb):
    cells = code_cells(nb)
    positions = []
    for path, comment, drop in builder.MODULES:
        expected, _ = builder.module_cell(path, comment, drop)
        assert expected in cells, path
        positions.append(cells.index(expected))
        assert expected.startswith(f"# {comment}\nfrom __future__ import annotations\n"), path
    assert positions == sorted(positions)
    assert len(positions) == len(set(positions)) == len(builder.MODULES) >= 16
    assert {p for p, _, _ in builder.MODULES} >= {
        "snake_rl/env/common.py", "snake_rl/env/obs.py", "snake_rl/env/rewards.py", "snake_rl/env/core.py",
        "snake_rl/env/vec_env.py", "snake_rl/agents/baselines.py", "snake_rl/agents/tabular_q.py",
        "snake_rl/agents/dqn/replay.py", "snake_rl/agents/dqn/networks.py", "snake_rl/agents/dqn/agent.py",
        "snake_rl/config.py", "snake_rl/evaluate.py", "snake_rl/train.py"}


def test_the_inlined_cells_keep_the_module_text_unchanged():
    """Only lines are removed (docstring, imports, ``main``, the ``__main__`` guard); every kept line is a line of the module."""
    for path, comment, drop in builder.MODULES:
        module_lines = (REPO / path).read_text(encoding="utf-8").splitlines()
        cell, _ = builder.module_cell(path, comment, drop)
        body = cell.split("\n", 2)[2].splitlines()
        kept = [line for line in body if line.strip()]
        assert kept, path
        assert all(line in module_lines for line in kept), path
        # the removals: no import statement and no command-line wrapper remain at module level
        assert not any(re.match(r"^(from|import) ", line) for line in body), path
        assert not any(line.startswith("if __name__") for line in body), path
        for name in drop:
            assert f"def {name}(" not in cell, (path, name)
        assert "snake_rl." not in "\n".join(line for line in body if not line.lstrip().startswith("#")) or path.endswith("train.py"), path


def test_the_study_is_defined_by_the_notebook_itself(nb):
    code = code_of(nb)
    for name in STUDY_NAMES:
        assert name in code, name
    assert "from snake_rl.env" not in code and "from snake_rl.agents" not in code and "import snake_rl.train" not in code
    assert "from snake_rl.analysis import report_tables as rt" in code  # the analysis modules come from the repository


def test_the_import_cell_gathers_every_import_once(nb):
    cells = code_cells(nb)
    imports = cells[1]
    assert imports.startswith("# Every import of the modules below")
    lines = [line for line in imports.splitlines() if re.match(r"^(from|import) ", line)]
    assert len(lines) == len(set(lines)) >= 25
    assert "import torch" in lines and "import numpy as np" in lines and "import gymnasium as gym" in lines
    assert "from snake_rl.utils import git_hash" in lines


def test_the_builder_refuses_shadowed_names():
    names = set()
    for path, comment, drop in builder.MODULES:
        cell, _ = builder.module_cell(path, comment, drop)
        module_names = builder.top_level_names(cell.split("\n", 2)[2])
        assert not (names & module_names), (path, names & module_names)
        names |= module_names
    assert {"SnakeEnv", "VecSnakeEnv", "DQNAgent", "TabularQAgent", "train", "evaluate", "load_config"} <= names
    assert "main" not in names


# ------------------------------------------------------------------ the layout
def test_the_title_and_the_introduction_open_the_notebook(nb):
    title, intro = nb["cells"][0], nb["cells"][1]
    assert title["cell_type"] == "markdown" and intro["cell_type"] == "markdown"
    assert source_of(title).startswith("# 43008 Reinforcement Learning -- Assignment 3, Part E")
    assert "14191136" in source_of(title) and "25211451" in source_of(title) and "26145324" in source_of(title)
    text = source_of(intro)
    for needle in ("snake_rl.analysis.report", "snake_rl.analysis.report_tables", "runs_logs.zip", "export_logs.py",
                   "scripts/build_notebook.py", "tests/test_m10_notebook.py", "TRAIN_MODE", REPO_URL[:-4]):
        assert needle in text, needle


def test_the_sections_are_numbered_in_order(nb):
    headings = [source_of(c).strip().splitlines()[0] for c in nb["cells"]
                if c["cell_type"] == "markdown" and source_of(c).startswith("### ")]
    assert [h.split()[1] for h in headings] == [f"{k}." for k in range(1, len(headings) + 1)]
    assert len(headings) == 11
    sub = [source_of(c).strip().splitlines()[0] for c in nb["cells"]
           if c["cell_type"] == "markdown" and source_of(c).startswith("#### ")]
    assert all(re.match(r"^#### \d+\.\d+ ", h) for h in sub) and len(sub) >= 35
    numbers = [tuple(int(x) for x in h.split()[1].split(".")) for h in sub]
    assert numbers == sorted(numbers)


def test_the_code_calls_the_report_pipeline(nb):
    code = code_of(nb)
    for needle in ("snake_rl.analysis.report ", "snake_rl.analysis.matrix", "report_tables", "runs_logs.zip", REPO_URL,
                   "--status", "pip install", "clone", "snake-rl", "LOGS_URL", "LOGS_ZIP", "zipfile", "urllib.request",
                   "google.colab", "snake_rl.analysis.behaviour", "os.path.exists", "IPython.display", "Markdown",
                   "Image", "sys.executable", "report_extra_figures", "gui_screenshots.py", "docs/PartE-final-report.md",
                   "report_captions", "report_figures", "compare_with_report", "appendix_table",
                   # training in the session
                   "TRAIN_MODE", "runs_notebook", "train_or_reuse", "configs/tabular_q_8x8.yaml", "configs/smoke_8x8.yaml",
                   "configs/smoke_B_8x8.yaml", "configs/dqn_A_8x8.yaml", "configs/dqn_B_8x8.yaml", "load_config(",
                   "evaluate(make_policy(", "BASELINE_NAMES", "markdown_table(", "plot_learning_curves(",
                   "EvalResult.load(", "DQNAgent.load(", "VecSnakeEnv(", "render_mode=\"ansi\""):
        assert needle in code, needle
    assert "!python " not in code  # the CLI calls use the kernel's interpreter


def test_every_table_and_figure_of_the_report_is_shown(nb):
    code = code_of(nb)
    names = report_items()
    assert len(names) >= 20
    for name in names:
        kind = "show_table" if name.startswith("Table") else "show_figure"
        assert f'{kind}("{name}"' in code or f'"{name}"' in code, name
    shown = [m.group(1) for m in re.finditer(r'show_(?:table|figure)\("((?:Table|Figure) [A-Z]?\d+)"', code)]
    # the direct calls come in the report's order (the appendix tables are shown by a loop)
    assert shown == [n for n in names if n in shown]


def test_the_cells_come_in_the_specified_order(nb):
    cells = nb["cells"]
    needles = (("install", "pip install"), ("imports", "REPORT_MD ="), ("env", "class SnakeEnv"),
               ("vec", "class VecSnakeEnv"), ("equivalence", "mismatches"), ("baselines_code", "class BFSGreedyPolicy"),
               ("tabular", "class TabularQAgent"), ("dqn", "class DQNAgent"), ("config", "def load_config"),
               ("evaluate", "def evaluate("), ("train", "def train("), ("settings", "TRAIN_MODE ="),
               ("baselines", "evaluate(make_policy("), ("runs", "train_or_reuse(*CONFIGS"),
               ("curves", "plot_learning_curves("), ("episode", "DQNAgent.load("), ("logs", "LOGS_ZIP"),
               ("matrix", "snake_rl.analysis.matrix"), ("report", "snake_rl.analysis.report "),
               ("composite", "report_extra_figures"), ("tables", "rt.build"), ("helpers", "report_captions"),
               ("check", "compare_with_report"), ("behaviour", "RUN_DIR"))
    code_index = {}
    for i, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        text = source_of(cell)
        for key, needle in needles:
            if needle in text:
                code_index.setdefault(key, i)
    keys = [k for k, _ in needles]
    assert set(keys) <= set(code_index), set(keys) - set(code_index)
    order = [code_index[k] for k in keys]
    assert order == sorted(order) and len(set(order)) == len(order), code_index
    assert cells[0]["cell_type"] == "markdown" and code_index["install"] >= 2
    assert cells[code_index["behaviour"] - 1]["cell_type"] == "markdown"


def test_every_code_cell_runs_the_notebooks_namespace_only(nb):
    """A cell of the notebook's own code must not rebind a module-level name of the inlined modules."""
    module_cells = {builder.module_cell(path, comment, drop)[0] for path, comment, drop in builder.MODULES}
    defined = set()
    for cell in module_cells:
        defined |= builder.top_level_names(cell.split("\n", 2)[2])
    for cell in code_cells(nb):
        if cell in module_cells:
            continue
        assert not (builder._notebook_names(cell) & defined), cell[:60]


# ------------------------------------------------------------------ the executed copy
def test_the_executed_copy_reproduces_the_report():
    nb = json.loads(EXECUTED.read_text(encoding="utf-8"))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert [source_of(c) for c in cells] == code_cells(json.loads(NOTEBOOK.read_text(encoding="utf-8")))
    assert not any(o.get("output_type") == "error" for c in cells for o in c["outputs"])
    streams = "".join("".join(o.get("text", "")) for c in cells for o in c["outputs"] if o.get("output_type") == "stream")
    assert VERDICT in streams
    assert "identical to the report" in streams and "differences" not in streams
    assert "4 boards x 300 steps: identical" in streams
    for exp in ("tabular_q_W8", "smoke_dqn_W8", "smoke_dqn_B8_W8"):
        assert f"[{exp}/0] finished" in streams, exp  # "finished ->" (trained here) or "finished run found" (reused)
    images = sum(1 for c in cells for o in c["outputs"] if "image/png" in o.get("data", {}))
    figures = len([n for n in report_items() if n.startswith("Figure")])
    assert images >= figures + 4  # the report's figures, two learning curves, heat-map and last steps
    tables = sum(1 for c in cells for o in c["outputs"] if "text/markdown" in o.get("data", {}))
    assert tables >= len([n for n in report_items() if n.startswith("Table")]) + 2

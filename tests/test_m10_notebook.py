"""``notebooks/results.ipynb`` (M10 spec section 6; Part E): the notebook that
regenerates every table and figure of the report from the training logs,
shows them in the report's order with the report's captions and checks the
computed tables against the report source.

``nbformat`` is not a dependency of the project, so the file is read as
JSON and the nbformat-4 rules that matter are checked by hand: every cell
has a type, an id and a non-empty source; code cells are committed clean
(no outputs, no execution count); the code calls the report pipeline
(``snake_rl.analysis.matrix``, ``snake_rl.analysis.report``,
``snake_rl.analysis.report_tables``), reads the logs from ``runs_logs.zip``
and clones the repository; every ``Table N.`` and ``Figure N.`` of
``docs/PartE-final-report.md`` is shown. The executed copy under
``notebooks/executed/`` must carry no error and the check cell's verdict.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "results.ipynb"
EXECUTED = REPO / "notebooks" / "executed" / "results.ipynb"
REPORT_MD = REPO / "docs" / "PartE-final-report.md"
REPO_URL = "https://github.com/Cureeeeeeee/Value-Based-Deep-Reinforcement-Learning-on-Snake.git"
VERDICT = "RESULT: every table and figure of the report is reproduced"


@pytest.fixture(scope="module")
def nb() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def source_of(cell: dict) -> str:
    source = cell["source"]
    return "".join(source) if isinstance(source, list) else source


def code_of(nb: dict) -> str:
    return "\n".join(source_of(c) for c in nb["cells"] if c["cell_type"] == "code")


def report_items() -> list[str]:
    """``Table N`` / ``Figure N`` names of every caption of the report source, in order."""
    names = []
    for line in REPORT_MD.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^((?:Table|Figure) [A-Z]?\d+)\.", line)
        if m:
            names.append(m.group(1))
    return names


def test_the_file_is_nbformat_4_json(nb):
    assert NOTEBOOK.is_file()
    assert nb["nbformat"] == 4
    assert nb["nbformat_minor"] == 5
    assert nb["metadata"]["kernelspec"]["name"] == "python3"  # Colab's kernel
    assert isinstance(nb["cells"], list) and len(nb["cells"]) >= 20


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
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) >= 10
    for cell in code_cells:
        assert cell["outputs"] == []
        assert cell["execution_count"] is None


def test_the_title_and_the_introduction_open_the_notebook(nb):
    title, intro = nb["cells"][0], nb["cells"][1]
    assert title["cell_type"] == "markdown" and intro["cell_type"] == "markdown"
    assert source_of(title).startswith("# 43008 Reinforcement Learning -- Assignment 3, Part E")
    assert "14191136" in source_of(title) and "25211451" in source_of(title) and "26145324" in source_of(title)
    text = source_of(intro)
    for needle in ("snake_rl.analysis.report", "snake_rl.analysis.report_tables", "runs_logs.zip", "export_logs.py"):
        assert needle in text, needle


def test_the_sections_are_numbered_in_order(nb):
    headings = [source_of(c).strip().splitlines()[0] for c in nb["cells"]
                if c["cell_type"] == "markdown" and source_of(c).startswith("### ")]
    assert [h.split()[1] for h in headings] == [f"{k}." for k in range(1, len(headings) + 1)]
    assert len(headings) >= 7
    sub = [source_of(c).strip().splitlines()[0] for c in nb["cells"]
           if c["cell_type"] == "markdown" and source_of(c).startswith("#### ")]
    assert all(re.match(r"^#### \d+\.\d+ ", h) for h in sub) and len(sub) >= 15


def test_the_code_calls_the_report_pipeline(nb):
    code = code_of(nb)
    for needle in ("snake_rl.analysis.report ", "snake_rl.analysis.matrix", "report_tables", "runs_logs.zip", REPO_URL,
                   "--status", "pip install", "clone", "snake-rl", "LOGS_URL", "LOGS_ZIP", "zipfile", "urllib.request",
                   "google.colab", "snake_rl.analysis.behaviour", "os.path.exists", "IPython.display", "Markdown",
                   "Image", "sys.executable", "report_extra_figures", "gui_screenshots.py", "docs/PartE-final-report.md",
                   "report_captions", "report_figures", "compare_with_report", "appendix_table"):
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
    code_index = {}
    for i, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        text = source_of(cell)
        for key, needle in (("install", "pip install"), ("logs", "LOGS_ZIP"), ("matrix", "snake_rl.analysis.matrix"),
                            ("report", "snake_rl.analysis.report "), ("composite", "report_extra_figures"),
                            ("tables", "rt.build"), ("helpers", "report_captions"), ("check", "compare_with_report"),
                            ("behaviour", "RUN_DIR")):
            if needle in text:
                code_index.setdefault(key, i)
    keys = ("install", "logs", "matrix", "report", "composite", "tables", "helpers", "check", "behaviour")
    order = [code_index[k] for k in keys]
    assert order == sorted(order) and len(set(order)) == len(order), code_index
    assert cells[0]["cell_type"] == "markdown" and code_index["install"] >= 2
    assert cells[code_index["behaviour"] - 1]["cell_type"] == "markdown"


def test_the_executed_copy_reproduces_the_report():
    nb = json.loads(EXECUTED.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert [source_of(c) for c in code_cells] == [source_of(c) for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
                                                  if c["cell_type"] == "code"]
    assert not any(o.get("output_type") == "error" for c in code_cells for o in c["outputs"])
    streams = "".join("".join(o.get("text", "")) for c in code_cells for o in c["outputs"] if o.get("output_type") == "stream")
    assert VERDICT in streams
    assert "identical to the report" in streams and "differences" not in streams
    images = sum(1 for c in code_cells for o in c["outputs"] if "image/png" in o.get("data", {}))
    assert images >= len([n for n in report_items() if n.startswith("Figure")])

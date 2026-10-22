"""``notebooks/results.ipynb`` (M10 spec section 6): the Colab notebook that
regenerates every table and figure of the report from the training logs.

``nbformat`` is not a dependency of the project, so the file is read as
JSON and the nbformat-4 rules that matter are checked by hand: every cell
has a type, an id and a non-empty source; code cells are committed clean
(no outputs, no execution count); the code calls the report pipeline
(``snake_rl.analysis.matrix``, ``snake_rl.analysis.report``), reads the
logs from ``runs_logs.zip`` and the manifest, and clones the repository.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "results.ipynb"
REPO_URL = "https://github.com/Cureeeeeeee/Value-Based-Deep-Reinforcement-Learning-on-Snake.git"


@pytest.fixture(scope="module")
def nb() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def source_of(cell: dict) -> str:
    source = cell["source"]
    return "".join(source) if isinstance(source, list) else source


def code_of(nb: dict) -> str:
    return "\n".join(source_of(c) for c in nb["cells"] if c["cell_type"] == "code")


def test_the_file_is_nbformat_4_json(nb):
    assert NOTEBOOK.is_file()
    assert nb["nbformat"] == 4
    assert nb["nbformat_minor"] == 5
    assert isinstance(nb["metadata"], dict)
    assert isinstance(nb["cells"], list) and len(nb["cells"]) >= 7


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
    assert len(code_cells) >= 5
    for cell in code_cells:
        assert cell["outputs"] == []
        assert cell["execution_count"] is None


def test_the_first_cell_is_the_markdown_introduction(nb):
    first = nb["cells"][0]
    assert first["cell_type"] == "markdown"
    text = source_of(first)
    assert "results notebook" in text and "Part E" in text
    assert "snake_rl.analysis.report" in text
    assert "runs_logs.zip" in text and "export_logs.py" in text


def test_the_code_calls_the_report_pipeline(nb):
    code = code_of(nb)
    for needle in ("snake_rl.analysis.report", "snake_rl.analysis.matrix", "runs_logs.zip", "manifest.json", REPO_URL):
        assert needle in code, needle
    assert "--status" in code
    assert "pip install" in code and "clone" in code and "snake-rl" in code
    assert "LOGS_URL" in code and "LOGS_ZIP" in code
    assert "zipfile" in code and "urllib.request" in code and "google.colab" in code
    assert "snake_rl.analysis.behaviour" in code and "os.path.exists" in code
    assert "IPython.display" in code and "Markdown" in code and "Image" in code


def test_the_cells_come_in_the_specified_order(nb):
    cells = nb["cells"]
    code_index = {}
    for i, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        text = source_of(cell)
        for key, needle in (("install", "pip install"), ("logs", "LOGS_ZIP"), ("matrix", "snake_rl.analysis.matrix"),
                            ("report", "snake_rl.analysis.report"), ("display", "manifest.json"),
                            ("behaviour", "snake_rl.analysis.behaviour")):
            if needle in text:
                code_index.setdefault(key, i)
    order = [code_index[k] for k in ("install", "logs", "matrix", "report", "display", "behaviour")]
    assert order == sorted(order) and len(set(order)) == len(order), code_index
    assert cells[0]["cell_type"] == "markdown" and code_index["install"] >= 1
    # the behaviour cell is introduced by a Markdown cell
    assert cells[code_index["behaviour"] - 1]["cell_type"] == "markdown"

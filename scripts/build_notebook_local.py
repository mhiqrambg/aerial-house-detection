"""Assemble notebooks/aerial_house_detection_colab.ipynb from the exported cell dumps.

The dumps are verbatim copies of the 36 cells that ran in Colab. This script only
packages them into nbformat 4.5 JSON so the notebook can be stored in git.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
OUT = NB_DIR / "aerial_house_detection_colab.ipynb"
HEADER = re.compile(r"^@@CELL (\d+) (markdown|code)$")

cells: dict[int, dict] = {}
current: dict | None = None
buf: list[str] = []

for part in sorted(NB_DIR.glob("_cells_*.txt")):
    for line in part.read_text().splitlines():
        match = HEADER.match(line)
        if match:
            if current is not None:
                cells[current["index"]] = {**current, "source": "\n".join(buf).strip("\n")}
            current = {"index": int(match.group(1)), "type": match.group(2)}
            buf = []
        else:
            buf.append(line)
if current is not None:
    cells[current["index"]] = {**current, "source": "\n".join(buf).strip("\n")}

assert sorted(cells) == list(range(36)), f"expected 36 cells, got {sorted(cells)}"

notebook_cells = []
for i in range(36):
    cell = cells[i]
    entry = {
        "cell_type": cell["type"],
        "id": f"colab-{i:02d}",
        "metadata": {},
        "source": cell["source"],
    }
    if cell["type"] == "code":
        entry["execution_count"] = None
        entry["outputs"] = []
    notebook_cells.append(entry)

notebook = {
    "cells": notebook_cells,
    "metadata": {
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False))
print("saved:", OUT, round(OUT.stat().st_size / 1e3, 1), "KB")
print("cells:", len(notebook_cells), "| code:", sum(c["cell_type"] == "code" for c in notebook_cells))

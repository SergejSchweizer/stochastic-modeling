"""Validate the checked-in notebook, narrative sidecar, and presentation export."""

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "MScFE622_GWP1.ipynb"
README = ROOT / "README.md"
HTML = ROOT / "outputs" / "MScFE622_GWP1.html"


def main() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code_cells, "notebook has no executable cells"
    assert all("hide-input" in cell.metadata.get("tags", []) for cell in code_cells)
    assert not [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.output_type == "error"
    ]
    readme_text = README.read_text(encoding="utf-8").lower()
    assert "code" not in readme_text and "notebook" not in readme_text
    html_text = HTML.read_text(encoding="utf-8")
    for marker in ("from dataclasses", "def heston_cf", "differential_evolution"):
        assert marker not in html_text


if __name__ == "__main__":
    main()

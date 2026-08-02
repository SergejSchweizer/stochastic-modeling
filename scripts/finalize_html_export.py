"""Apply accessibility metadata that nbconvert does not preserve from image outputs."""

import html
import re
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "MScFE622_GWP1.ipynb"
HTML = ROOT / "outputs" / "MScFE622_GWP1.html"
MISSING_ALT = 'alt="No description has been provided for this image"'


def _plain_text(fragment: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", fragment)).strip()


def main() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    descriptions = [
        _plain_text(output.data["text/html"])
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
        if output.output_type in ("display_data", "execute_result")
        and "text/html" in output.data
        and "plot-description" in output.data["text/html"]
    ]
    source = HTML.read_text(encoding="utf-8")
    if source.count(MISSING_ALT) != len(descriptions):
        raise ValueError("HTML image count does not match the notebook plot descriptions.")
    for description in descriptions:
        replacement = f'alt="{html.escape(description, quote=True)}"'
        source = source.replace(MISSING_ALT, replacement, 1)
    HTML.write_text(source, encoding="utf-8")
    print(f"Added descriptive alternative text to {len(descriptions)} HTML plots.")


if __name__ == "__main__":
    main()

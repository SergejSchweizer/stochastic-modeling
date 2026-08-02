"""Regenerate the call-only MScFE 622 coursework notebook."""

import ast
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "MScFE622_GWP1.ipynb"
CALLS = (
    "show_data_quality_checks",
    "show_pricing_method_configuration",
    "show_heston_lewis_calibration",
    "show_heston_lewis_fit",
    "show_heston_carr_madan_calibration",
    "show_asian_call_valuation",
    "show_bates_lewis_calibration",
    "show_bates_carr_madan_calibration",
    "show_seventy_day_put_valuation",
    "show_cir_calibration",
    "show_cir_scenarios",
    "refresh_narrative_sidecar",
)
CELL_SOURCES = (
    'get_ipython().run_line_magic("load_ext", "stochastic_modeling.notebook_runtime")',
    *(f"{name}()" for name in CALLS),
)


def validate_function_call(source: str) -> str:
    """Validate and return one expression containing only a function invocation."""
    tree = ast.parse(source)
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr)
    assert isinstance(tree.body[0].value, ast.Call)
    return source


def main() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if len(code_cells) != len(CELL_SOURCES):
        raise ValueError(f"Expected {len(CELL_SOURCES)} executable cells, found {len(code_cells)}.")

    for cell, source in zip(code_cells, CELL_SOURCES, strict=True):
        cell.source = validate_function_call(source)
        cell.outputs = []
        cell.execution_count = None
        cell.metadata = {"tags": ["hide-input"], "jupyter": {"source_hidden": True}}

    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3.14 (.venv)",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3.14"}
    nbformat.write(notebook, NOTEBOOK)
    print(NOTEBOOK)


if __name__ == "__main__":
    main()

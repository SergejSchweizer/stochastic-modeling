import ast
import re
import subprocess
from pathlib import Path

import nbformat


def test_no_generated_junk_is_tracked():
    root = Path(__file__).resolve().parents[1]
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    forbidden_parts = {".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "tmp", "htmlcov"}
    assert not [path for path in tracked if forbidden_parts.intersection(Path(path).parts)]
    assert not [path for path in tracked if path.endswith((".log", ".pyc", ".coverage"))]


def test_notebook_cells_are_call_only_and_have_outputs():
    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(root / "notebooks" / "MScFE622_GWP1.ipynb", as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]

    assert code_cells
    assert all(cell.outputs for cell in code_cells)
    for cell in code_cells:
        tree = ast.parse(cell.source)
        assert len(tree.body) == 1
        assert isinstance(tree.body[0], ast.Expr)
        assert isinstance(tree.body[0].value, ast.Call)
        assert not any(
            isinstance(node, (ast.Assign, ast.Import, ast.ImportFrom, ast.FunctionDef, ast.Lambda))
            for node in ast.walk(tree)
        )


def test_notebook_has_numbered_eight_point_equation_and_plot_annotations():
    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(root / "notebooks" / "MScFE622_GWP1.ipynb", as_version=4)
    markdown = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
    descriptions = [
        output.data["text/html"]
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
        if output.output_type in ("display_data", "execute_result")
        and "text/html" in output.data
        and "plot-description" in output.data["text/html"]
    ]

    assert "font-size: 8pt !important" in markdown
    equations = re.findall(r"\$\$(.*?)\$\$", markdown, flags=re.DOTALL)
    assert len(equations) == 25
    assert all(r"\tag{" in equation for equation in equations)
    assert r"\[" not in markdown and r"\]" not in markdown
    assert len(descriptions) == 9
    assert all("font-size: 8pt !important" in description for description in descriptions)
    assert [
        re.search(r"<strong>Plot (\d+)\.</strong>", description).group(1)
        for description in descriptions
    ] == [str(number) for number in range(1, 10)]

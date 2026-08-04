"""Validate the checked-in notebook, narrative sidecar, and presentation export."""

import ast
import re
from pathlib import Path

import nbformat
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "MScFE622_GWP1.ipynb"
README = ROOT / "README.md"
HTML = ROOT / "outputs" / "MScFE622_GWP1.html"
PDF = ROOT / "outputs" / "pdf" / "Stochastic_Modeling_GWP1.pdf"


def main() -> None:
    assert not (ROOT / "output").exists(), "legacy output directory must not exist"
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    markdown_text = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )
    assert code_cells, "notebook has no executable cells"
    assert all("hide-input" in cell.metadata.get("tags", []) for cell in code_cells)
    assert all(cell.get("outputs") for cell in code_cells), (
        "every notebook function call must have a captured output"
    )
    for cell in code_cells:
        tree = ast.parse(cell.source)
        assert len(tree.body) == 1, "each executable cell must contain one function call"
        statement = tree.body[0]
        assert isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call), (
            "notebook implementation must be limited to function calls"
        )
        assert not any(
            isinstance(node, (ast.Assign, ast.Import, ast.ImportFrom, ast.FunctionDef, ast.Lambda))
            for node in ast.walk(tree)
        ), "notebook cells may not contain implementation statements"
    assert not [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.output_type == "error"
    ]
    display_equations = re.findall(r"\$\$(.*?)\$\$", markdown_text, flags=re.DOTALL)
    assert display_equations, "notebook has no display equations"
    assert len(display_equations) == 25
    assert all("\\tag*{" in equation for equation in display_equations)
    assert r"\[" not in markdown_text and r"\]" not in markdown_text
    equation_numbers = [
        int(number)
        for number in re.findall(r"\\tag\*\{\$\\scriptstyle \((\d+)\)\$\}", markdown_text)
    ]
    assert equation_numbers == list(range(1, 26))
    parameter_blocks = {
        "**Parameters (Equations 1-3).**": 1,
        "**Parameters (Equation 4).**": 4,
        "**Parameters (Equations 5-6).**": 5,
        "**Parameters (Equation 7).**": 7,
        "**Parameters (Equations 8-9).**": 8,
        "**Parameters (Equations 10-13).**": 10,
        "**Parameters (Equations 14-16).**": 14,
        "**Parameters (Equation 17).**": 17,
        "**Parameters (Equations 18-23).**": 18,
        "**Parameters (Equations 24-25).**": 24,
    }
    for label, first_equation in parameter_blocks.items():
        assert label in markdown_text
        equation_tag = rf"\tag*{{$\scriptstyle ({first_equation})$}}"
        assert markdown_text.index(label) < markdown_text.index(equation_tag)
    assert not any(character in markdown_text for character in ("\a", "\b", "\f", "\v"))
    assert "font-size: 10pt !important" in markdown_text
    for explanation_label in (
        "**Heston model purpose.**",
        "**Lewis representation purpose.**",
        "**Carr-Madan method purpose.**",
        "**Asian valuation purpose.**",
        "**Bates model purpose.**",
        "**Bates-Carr-Madan purpose.**",
        "**Put-pricing purpose.**",
        "**CIR model purpose.**",
        "**CIR simulation purpose.**",
    ):
        assert explanation_label in markdown_text
    plot_descriptions = [
        output.data["text/html"]
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.output_type in ("display_data", "execute_result")
        and "text/html" in output.data
        and "plot-description" in output.data["text/html"]
    ]
    assert len(plot_descriptions) == 9
    assert all("plot-description" in description for description in plot_descriptions)
    assert [
        re.search(r"<strong>Plot (\d+)\.</strong>", description).group(1)
        for description in plot_descriptions
    ] == [str(number) for number in range(1, 10)]
    discussion_headings = {
        output.data["text/markdown"].splitlines()[0]
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.output_type in ("display_data", "execute_result")
        and "text/markdown" in output.data
    }
    assert discussion_headings == {
        "#### Calibration discussion",
        "#### Lewis versus Carr-Madan",
        "#### Valuation and client recommendation",
        "#### Bates calibration discussion",
        "#### Bates pricing-method comparison",
        "#### Pricing interpretation",
        "#### CIR calibration discussion",
        "#### Rate-scenario discussion",
    }
    for citation in (
        "(Heston 327-343)",
        "(Carr and Madan 61-73)",
        "(Bates 69-107)",
        "(Cox, Ingersoll, and Ross 385-407)",
        "(WorldQuant University 1-2)",
    ):
        assert citation in markdown_text
    assert "## Works Cited" in markdown_text
    readme_text = README.read_text(encoding="utf-8").lower()
    assert "code" not in readme_text and "notebook" not in readme_text
    assert "## works cited" in readme_text
    html_text = HTML.read_text(encoding="utf-8")
    assert html_text.count("plot-description") >= 9
    assert "No description has been provided for this image" not in html_text
    for marker in ("from dataclasses", "def heston_cf", "differential_evolution"):
        assert marker not in html_text
    pdf = PdfReader(PDF)
    assert len(pdf.pages) >= 2
    pdf_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    participants = [
        "Umuhoza Denyse Graine",
        "Opeyemi Waliyilah Oladipupo",
        "Sergej Schweizer",
    ]
    positions = [pdf_text.index(participant) for participant in participants]
    assert positions == sorted(positions)
    assert pdf.metadata.title == "Stochastic Modeling Group Work Project 1"
    assert "16855" in pdf_text
    assert "code" not in pdf_text.lower() and "notebook" not in pdf_text.lower()
    assert all(f"Plot {number}." in pdf_text for number in range(1, 10))


if __name__ == "__main__":
    main()

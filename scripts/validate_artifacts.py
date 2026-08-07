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
        "### Calibration discussion",
        "### Lewis versus Carr-Madan",
        "### Valuation and client recommendation",
        "### Bates calibration discussion",
        "### Bates pricing-method comparison",
        "### Pricing interpretation",
        "### CIR calibration discussion",
        "### Rate-scenario discussion",
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
    normalized_pdf_text = " ".join(pdf_text.split())
    participants = [
        "Umuhoza Denyse Graine",
        "Opeyemi Waliyilah Oladipupo",
        "Sergej Schweizer",
    ]
    positions = [pdf_text.index(participant) for participant in participants]
    assert positions == sorted(positions)
    assert pdf.metadata.title == "Stochastic Modeling Group Work Project 1"
    cover_text = pdf.pages[0].extract_text() or ""
    assert re.search(r"\b1\s+16855\b", cover_text)
    assert "16855" in pdf_text
    assert "code" not in pdf_text.lower() and "notebook" not in pdf_text.lower()
    assert all(f"Plot {number}." in pdf_text for number in range(1, 10))
    plot_explanations = (
        "remaining gaps reflect quote inconsistency",
        "no single directional bias explains the errors",
        "agreement in prices despite materially different calibrated parameters",
        "highlighted median path gives a typical scenario",
        "confidence band narrows as Monte Carlo uncertainty declines",
        "limitations of fitting the inconsistent call-put cross-section",
        "supporting numerical consistency across methods",
        "small deviations summarized by the curve RMSE",
        "zoom excludes only the highest 10% from view",
    )
    assert all(explanation in normalized_pdf_text for explanation in plot_explanations)
    assert all(f"({number})" in pdf_text for number in range(1, 26))
    assert not re.search(r"^Equation \d+$", pdf_text, flags=re.MULTILINE)
    section_equations = {
        "1(i) Heston calibration using Lewis (2001)": (
            "Heston represents equity returns",
            range(1, 8),
        ),
        "1(ii) Carr-Madan comparison": (
            "Carr-Madan damps the option-price function",
            range(8, 10),
        ),
        "1(iii) 20-day ATM Asian call": (
            "The arithmetic-average Asian payoff",
            range(10, 14),
        ),
        "2(i) Bates calibration using Lewis": (
            "Bates extends Heston",
            range(14, 17),
        ),
        "2(ii) Carr-Madan comparison": (
            "This specification retains the Bates jump",
            range(8, 10),
        ),
        "2(iii) 70-day 95%-moneyness put": (
            "The project specifies a strike",
            range(17, 18),
        ),
        "3(i) CIR calibration": (
            "CIR models a non-negative",
            range(18, 24),
        ),
        "3(ii) One-year rate scenarios": (
            "Daily CIR simulation turns",
            range(24, 26),
        ),
    }
    section_positions = [pdf_text.index(heading) for heading in section_equations]
    section_ends = section_positions[1:] + [pdf_text.index("Works Cited")]
    for (heading, (introduction, equation_numbers)), start, end in zip(
        section_equations.items(), section_positions, section_ends, strict=True
    ):
        section_text = pdf_text[start:end]
        assert all(f"({number})" in section_text for number in equation_numbers), heading
        assert section_text.index(introduction) < section_text.index(
            f"({equation_numbers.start})"
        ), heading
    assert "Model equations and parameter descriptions" not in pdf_text
    assert "Equation 1 description." not in pdf_text
    for parameter_label in (
        "Parameters (Equations 1-3).",
        "Parameters (Equation 4).",
        "Parameters (Equations 5-6).",
        "Parameters (Equation 7).",
        "Parameters (Equations 8-9).",
        "Parameters (Equations 10-13).",
        "Parameters (Equations 14-16).",
        "Parameters (Equation 17).",
        "Parameters (Equations 18-23).",
        "Parameters (Equations 24-25).",
    ):
        assert parameter_label in pdf_text


if __name__ == "__main__":
    main()

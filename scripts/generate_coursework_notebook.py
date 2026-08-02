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
NOTEBOOK_STYLE = """<style>
/* WQU presentation typography: equation labels and figure descriptions use 8 pt. */
.jp-RenderedHTMLCommon mjx-container[jax="CHTML"][display="true"] mjx-labels,
.jp-RenderedHTMLCommon mjx-container[jax="CHTML"][display="true"] mjx-labels *,
.jp-RenderedHTMLCommon .MathJax .mjx-tag,
.jp-RenderedHTMLCommon .MathJax_Display .mjx-tag,
.jp-RenderedHTMLCommon .mtd[id^="mjx-eqn"],
.jp-RenderedHTMLCommon [id^="mjx-eqn"],
.plot-description {
  font-size: 8pt !important;
}
.plot-description {
  line-height: 1.3;
  color: #444;
  margin: 0.25rem 0 1rem;
}
</style>"""

MODEL_EXPLANATIONS = {
    "## 1(a) Heston (1993) calibrated with the Lewis (2001) representation": (
        "**Heston model purpose.** The Heston model represents equity returns with a variance "
        "that changes randomly through time and can be correlated with stock-price shocks. It "
        "therefore captures volatility clustering, mean reversion, and the implied-volatility "
        "skew that a constant-volatility model cannot reproduce."
    ),
    "The Lewis representation used for calls is (Lewis):": (
        "**Lewis representation purpose.** Lewis converts the Heston characteristic function "
        "into a stable one-dimensional Fourier integral for European call values; put values "
        "then follow from put-call parity."
    ),
    "## 1(b) Heston calibrated with Carr-Madan (1999)": (
        "**Carr-Madan method purpose.** Carr-Madan damps the call-price function so its Fourier "
        "transform is integrable, providing an independent numerical route for pricing and "
        "calibrating the same Heston dynamics."
    ),
    "## 1(c) Twenty-day ATM Asian call": (
        "**Asian valuation purpose.** The arithmetic-average payoff reduces sensitivity to a "
        "single terminal stock price. Because the average depends on the complete simulated "
        "path, Monte Carlo valuation is used under the calibrated Heston dynamics."
    ),
    "## 2(a) Bates (1996) calibrated with Lewis (2001)": (
        "**Bates model purpose.** Bates extends Heston by adding random discontinuous price "
        "jumps. The diffusion describes ordinary volatility dynamics, while jump intensity and "
        "jump-size parameters capture rare, abrupt moves that matter more at 60 days."
    ),
    "## 2(b) Bates (1996) calibrated with Carr-Madan (1999)": (
        "**Bates-Carr-Madan purpose.** This specification keeps the Bates jump and stochastic-"
        "variance dynamics but evaluates prices with the damped Carr-Madan transform, providing "
        "a numerical cross-check against Lewis pricing."
    ),
    "## 2(c) Seventy-day, 95% moneyness vanilla put": (
        "**Put-pricing purpose.** The calibrated Bates model converts the requested strike and "
        "70-day horizon into a downside-protection value while retaining both stochastic "
        "volatility and jump risk."
    ),
    "## 3(a) CIR (1985) calibration to the Euribor term structure": (
        "**CIR model purpose.** CIR models a non-negative, mean-reverting short rate. Its "
        "closed-form zero-coupon bond prices connect the short-rate parameters to the observed "
        "Euribor term structure."
    ),
    "## 3(b) One-year daily CIR simulation": (
        "**CIR simulation purpose.** Daily simulation propagates the calibrated rate dynamics "
        "one year forward, producing an expected terminal rate and a distribution for "
        "interest-rate risk rather than a single deterministic forecast."
    ),
}


def validate_function_call(source: str) -> str:
    """Validate and return one expression containing only a function invocation."""
    tree = ast.parse(source)
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr)
    assert isinstance(tree.body[0].value, ast.Call)
    return source


def apply_presentation_markdown(notebook) -> None:
    """Add model explanations and exact presentation styles idempotently."""
    markdown_cells = [cell for cell in notebook.cells if cell.cell_type == "markdown"]
    if not markdown_cells:
        raise ValueError("Notebook has no narrative cells.")
    if NOTEBOOK_STYLE not in markdown_cells[0].source:
        markdown_cells[0].source = f"{NOTEBOOK_STYLE}\n\n{markdown_cells[0].source}"

    for anchor, explanation in MODEL_EXPLANATIONS.items():
        matching_cells = [cell for cell in markdown_cells if anchor in cell.source]
        if len(matching_cells) != 1:
            raise ValueError(f"Expected one markdown anchor for {anchor!r}.")
        cell = matching_cells[0]
        if explanation not in cell.source:
            cell.source = cell.source.replace(anchor, f"{anchor}\n\n{explanation}", 1)


def main() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    apply_presentation_markdown(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if len(code_cells) != len(CELL_SOURCES):
        raise ValueError(f"Expected {len(CELL_SOURCES)} executable cells, found {len(code_cells)}.")

    for cell, source in zip(code_cells, CELL_SOURCES, strict=True):
        cell.source = validate_function_call(source)
        cell.outputs = []
        cell.execution_count = None
        cell.metadata = {"tags": ["hide-input"], "jupyter": {"source_hidden": True}}

    nbformat.write(notebook, NOTEBOOK)
    print(NOTEBOOK)


if __name__ == "__main__":
    main()

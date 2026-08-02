"""Regenerate the call-only MScFE 622 coursework notebook."""

import ast
import re
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
NOTEBOOK_STYLE = """<!-- NOTEBOOK_THEME_START -->
<style>
/* Professional WQU notebook theme. */
.jp-Notebook {
  max-width: 1120px;
  margin: 0 auto;
}
.jp-RenderedHTMLCommon {
  color: #243447;
  font-family: Arial, Calibri, sans-serif;
  font-size: 10pt;
  line-height: 1.55;
    text-align: justify;
    text-align-last: center;
    width: 100%;
    box-sizing: border-box;
}
.jp-RenderedHTMLCommon h1,
.jp-RenderedHTMLCommon h2,
.jp-RenderedHTMLCommon h3,
.jp-RenderedHTMLCommon h4 {
    text-align: center;
    text-align-last: center;
}
.jp-RenderedHTMLCommon p,
.jp-RenderedHTMLCommon li {
    font-size: 10pt;
    text-align: justify;
    text-align-last: center;
}
.jp-RenderedHTMLCommon h1 {
  color: #28345c;
  border-bottom: 2px solid #465b8c;
  padding-bottom: 0.3rem;
}
.jp-RenderedHTMLCommon h2 {
  color: #1f6f9f;
  margin-top: 1.7rem;
}
.jp-RenderedHTMLCommon h3,
.jp-RenderedHTMLCommon h4 {
  color: #355070;
}
.jp-RenderedHTMLCommon blockquote {
  background: #f4f7fb;
  border-left: 4px solid #4b7f9f;
  color: #263849;
    font-size: 10pt;
  margin: 1rem 0;
  padding: 0.7rem 1rem;
}
.jp-RenderedHTMLCommon table {
  border-collapse: collapse;
    font-size: 10pt !important;
    margin-left: auto;
    margin-right: auto;
    width: 100%;
}
.jp-RenderedHTMLCommon th,
.jp-RenderedHTMLCommon td {
    font-size: 10pt !important;
    text-align: center !important;
    text-align-last: center !important;
}
.jp-RenderedHTMLCommon th {
  background: #e9eef6;
  color: #28345c;
}
.jp-RenderedHTMLCommon img {
    display: block;
    margin-left: auto;
    margin-right: auto;
}
.jp-RenderedHTMLCommon mjx-container[display="true"],
.jp-RenderedHTMLCommon .MathJax_Display {
  margin: 1.1rem 0 !important;
  overflow-x: auto;
  overflow-y: hidden;
}
.jp-RenderedHTMLCommon mjx-container[jax="CHTML"][display="true"] mjx-labels,
.jp-RenderedHTMLCommon mjx-container[jax="CHTML"][display="true"] mjx-labels *,
.jp-RenderedHTMLCommon .MathJax .mjx-tag,
.jp-RenderedHTMLCommon .MathJax_Display .mjx-tag,
.jp-RenderedHTMLCommon .mtd[id^="mjx-eqn"],
.jp-RenderedHTMLCommon [id^="mjx-eqn"] {
    font-size: 8pt !important;
}
.jp-RenderedHTMLCommon p.plot-description {
    font-size: 8pt !important;
  line-height: 1.3;
  color: #444;
  margin: 0.25rem 0 1rem;
        text-align: center !important;
        text-align-last: center !important;
}
</style>
<!-- NOTEBOOK_THEME_END -->"""

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

SECTION_APPENDICES = {
    "## 1(a) Heston (1993) calibrated with the Lewis (2001) representation": r"""
### Calibration procedure

1. Retain exactly the five calls and five puts with 15 trading days to maturity and set $T=15/250=0.06$ years.
2. For each trial parameter vector, evaluate calls from Equation (5) and derive puts from Equation (6), using the common spot and 1.50% risk-free rate.
3. Minimize Equation (7) in dollar-price space. A seeded differential-evolution search first explores the bounded parameter region; bounded nonlinear least squares then refines the candidate using the ten individual residuals.
4. Assess the result using MSE, RMSE, option-level residuals, put-call parity, parameter bounds, and the Feller diagnostic $2\kappa\theta-\sigma^2$. This separates numerical fit from data inconsistency and parameter-identification risk.
""".strip(),
    "## 1(b) Heston calibrated with Carr-Madan (1999)": r"""
### Comparison procedure

The maturity, observations, parameter bounds, parity convention, optimizer sequence, and unweighted MSE are held fixed. Only the Fourier pricing representation changes. This controlled design makes fitted prices, objective values, and direct same-parameter price differences valid numerical comparisons; parameter equality is a weaker criterion when the short-maturity objective is flat.
""".strip(),
    "## 1(c) Twenty-day ATM Asian call": r"""
### Monte Carlo procedure

The selected Heston calibration is simulated under the risk-neutral measure for 20 daily steps of $\Delta t=1/250$. Full truncation replaces negative Euler variance inputs by zero, correlated Gaussian shocks preserve the fitted leverage effect, and today's spot is included before the 20 simulated observations. The analysis uses 200,000 paths with a fixed seed, discounts every payoff at the risk-free rate, and reports both estimator uncertainty and the required commercial fee.
""".strip(),
    "## 2(a) Bates (1996) calibrated with Lewis (2001)": r"""
### Calibration procedure

The ten 60-day options are converted to $T=60/250=0.24$ years. The five Heston parameters and three jump parameters are estimated jointly by the same seeded global-search/local-refinement sequence and regular price MSE used in Step 1(a). Calls use the Lewis integral with the Bates characteristic function; puts use parity. Residual, parity, boundary, and Feller diagnostics are reviewed because adding jump flexibility does not make inconsistent quotes jointly attainable.
""".strip(),
    "## 2(b) Bates (1996) calibrated with Carr-Madan (1999)": r"""
### Comparison procedure

The Carr-Madan exercise changes only the numerical pricing transform. The 60-day sample, eight-parameter bounds, optimizer tolerances, parity construction, and objective remain unchanged. Similar MSE with different parameter vectors indicates weak identification or multiple near-minima; it does not by itself imply a pricing-engine error.
""".strip(),
    "## 3(a) CIR (1985) calibration to the Euribor term structure": r"""
### Curve-construction and calibration procedure

The five supplied annual rates are mapped to 7, 30, 90, 180, and 365 days. A natural cubic spline produces 52 weekly nodes from day 7 through day 364, with negative interpolated values floored at zero to remain consistent with CIR support. The one-week observation initializes the short rate. Differential evolution then minimizes the mean squared difference between the weekly spline and the continuously compounded CIR zero rates from Equations (19)-(23).
""".strip(),
    "## 3(b) One-year daily CIR simulation": r"""
### Scenario procedure

Starting from the calibrated initial rate, 100,000 paths are advanced through 250 daily steps with $\Delta t=1/250$ and a fixed random seed. Full truncation evaluates the square-root diffusion at the non-negative part of the current rate and floors terminal rates at zero. The 2.5th and 97.5th percentiles form the selected 95% range; the sample mean estimates the expected terminal rate.
""".strip(),
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
    first_source = re.sub(
        r"<!-- NOTEBOOK_THEME_START -->.*?<!-- NOTEBOOK_THEME_END -->\s*",
        "",
        markdown_cells[0].source,
        flags=re.DOTALL,
    )
    first_source = re.sub(
        r"<style>\s*/\* WQU presentation typography:.*?</style>\s*",
        "",
        first_source,
        flags=re.DOTALL,
    )
    markdown_cells[0].source = f"{NOTEBOOK_STYLE}\n\n{first_source.lstrip()}"

    for anchor, explanation in MODEL_EXPLANATIONS.items():
        matching_cells = [cell for cell in markdown_cells if anchor in cell.source]
        if len(matching_cells) != 1:
            raise ValueError(f"Expected one markdown anchor for {anchor!r}.")
        cell = matching_cells[0]
        styled_explanation = f"> {explanation}"
        if styled_explanation not in cell.source:
            if explanation in cell.source:
                cell.source = cell.source.replace(explanation, styled_explanation, 1)
            else:
                cell.source = cell.source.replace(anchor, f"{anchor}\n\n{styled_explanation}", 1)

    for anchor, appendix in SECTION_APPENDICES.items():
        matching_cells = [cell for cell in markdown_cells if anchor in cell.source]
        if len(matching_cells) != 1:
            raise ValueError(f"Expected one markdown section for {anchor!r}.")
        cell = matching_cells[0]
        appendix_heading = appendix.splitlines()[0]
        if appendix_heading not in cell.source:
            cell.source = f"{cell.source.rstrip()}\n\n{appendix}"

    for cell in markdown_cells:
        cell.source = (
            cell.source.replace(r"\[", "$$")
            .replace(r"\]", "$$")
            .replace(r"\(", "$")
            .replace(r"\)", "$")
        )


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

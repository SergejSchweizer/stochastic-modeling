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
.jp-RenderedHTMLCommon p:not(.plot-description),
.jp-RenderedHTMLCommon blockquote,
.jp-RenderedHTMLCommon li,
.jp-RenderedHTMLCommon table,
.jp-RenderedHTMLCommon th,
.jp-RenderedHTMLCommon td {
    font-family: Arial, Calibri, sans-serif !important;
    font-size: 10pt !important;
    font-style: normal !important;
    line-height: 1.55 !important;
    letter-spacing: 0px !important;
}
.jp-RenderedHTMLCommon h1,
.jp-RenderedHTMLCommon h2,
.jp-RenderedHTMLCommon h3,
.jp-RenderedHTMLCommon h4 {
    text-align: center;
    text-align-last: center;
}
.jp-RenderedHTMLCommon p {
    text-align: justify;
    text-align-last: center;
}
.jp-RenderedHTMLCommon ul,
.jp-RenderedHTMLCommon ol {
    padding-left: 0;
    text-align: center;
    text-align-last: center;
}
.jp-RenderedHTMLCommon li {
    list-style-position: inside;
    text-align: center;
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
  margin: 1rem 0;
  padding: 0.7rem 1rem;
}
.jp-RenderedHTMLCommon table {
  border-collapse: collapse;
    margin-left: auto;
    margin-right: auto;
        text-align: center !important;
        text-align-last: center !important;
    width: 100%;
}
.jp-RenderedHTMLCommon th,
.jp-RenderedHTMLCommon td {
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
.jp-RenderedHTMLCommon .mjx-label .mjx-mstyle > .mjx-mrow {
    font-size: 85.5% !important;
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
    "## 1(i) Heston (1993) calibrated with the Lewis (2001) representation": (
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
    "## 1(ii) Heston calibrated with Carr-Madan (1999)": (
        "**Carr-Madan method purpose.** Carr-Madan damps the call-price function so its Fourier "
        "transform is integrable, providing an independent numerical route for pricing and "
        "calibrating the same Heston dynamics."
    ),
    "## 1(iii) Twenty-day ATM Asian call": (
        "**Asian valuation purpose.** The arithmetic-average payoff reduces sensitivity to a "
        "single terminal stock price. Because the average depends on the complete simulated "
        "path, Monte Carlo valuation is used under the calibrated Heston dynamics."
    ),
    "## 2(i) Bates (1996) calibrated with Lewis (2001)": (
        "**Bates model purpose.** Bates extends Heston by adding random discontinuous price "
        "jumps. The diffusion describes ordinary volatility dynamics, while jump intensity and "
        "jump-size parameters capture rare, abrupt moves that matter more at 60 days."
    ),
    "## 2(ii) Bates (1996) calibrated with Carr-Madan (1999)": (
        "**Bates-Carr-Madan purpose.** This specification keeps the Bates jump and stochastic-"
        "variance dynamics but evaluates prices with the damped Carr-Madan transform, providing "
        "a numerical cross-check against Lewis pricing."
    ),
    "## 2(iii) Seventy-day, 95% moneyness vanilla put": (
        "**Put-pricing purpose.** The calibrated Bates model converts the requested strike and "
        "70-day horizon into a downside-protection value while retaining both stochastic "
        "volatility and jump risk."
    ),
    "## 3(i) CIR (1985) calibration to the Euribor term structure": (
        "**CIR model purpose.** CIR models a non-negative, mean-reverting short rate. Its "
        "closed-form zero-coupon bond prices connect the short-rate parameters to the observed "
        "Euribor term structure."
    ),
    "## 3(ii) One-year daily CIR simulation": (
        "**CIR simulation purpose.** Daily simulation propagates the calibrated rate dynamics "
        "one year forward, producing an expected terminal rate and a distribution for "
        "interest-rate risk rather than a single deterministic forecast."
    ),
}

PARAMETER_DESCRIPTIONS = {
    "Under the risk-neutral measure, the stock and instantaneous variance follow the Heston diffusion (Heston 327-343):": (
        "**Parameters (Equations 1-3).** $S_t$ is the stock price and $v_t$ is its "
        "instantaneous variance at time $t$; $r$ is the continuously compounded risk-free "
        "rate and $dt$ is an infinitesimal time increment. $W_t^S$ and $W_t^v$ are Brownian "
        "motions driving price and variance. The variance parameters are mean-reversion speed "
        "$\\kappa$, long-run variance $\\theta$, volatility of variance $\\sigma$, correlation "
        "$\\rho$ between the Brownian shocks, and initial variance $v_0$."
    ),
    "The parameter vector is $(\\kappa,\\theta,\\sigma,\\rho,v_0)$. Define the characteristic function of the log return as": (
        "**Parameters (Equation 4).** $\\varphi_T(z)$ is the characteristic function of the "
        "log return over maturity $T$ at complex Fourier argument $z$; $i=\\sqrt{-1}$, "
        "$S_0$ and $S_T$ are the initial and terminal stock prices, and "
        "$\\mathbb{E}^{\\mathbb{Q}}$ denotes expectation under the risk-neutral measure "
        "$\\mathbb{Q}$."
    ),
    "> **Lewis representation purpose.** Lewis converts the Heston characteristic function into a stable one-dimensional Fourier integral for European call values; put values then follow from put-call parity.": (
        "**Parameters (Equations 5-6).** $C_0$ and $P_0$ are today's European call and put "
        "values; $S_0$ is spot, $K$ is strike, $T$ is time to maturity, and $r$ is the "
        "risk-free rate. $u$ is the Fourier integration variable, $i=\\sqrt{-1}$, "
        "$\\operatorname{Re}[\\cdot]$ takes the real part, $\\pi$ is the circle constant, and "
        "$\\varphi_T$ is the Heston characteristic function from Equation (4)."
    ),
    "The calibration uses all five 15-day calls and all five 15-day puts with the requested unweighted dollar-price MSE:": (
        "**Parameters (Equation 7).** $\\Theta$ is the trial Heston parameter vector; $N=10$ "
        "is the number of option quotes and $i$ indexes them. $V_i^{\\mathrm{model}}(\\Theta)$ "
        "and $V_i^{\\mathrm{market}}$ are the model and observed USD prices, respectively, and "
        "$\\operatorname{MSE}$ is their mean squared pricing error."
    ),
    "Carr-Madan prices the same characteristic function through a damped Fourier transform (Carr and Madan 61-73). For damping parameter $\\alpha>0$, the transformed call value is": (
        "**Parameters (Equations 8-9).** $\\psi_T(v)$ is the damped transform at Fourier "
        "frequency $v$ and maturity $T$; $\\alpha>0$ is the damping parameter, $r$ is the "
        "risk-free rate, and $i=\\sqrt{-1}$. $\\phi_{\\log S_T}$ is the characteristic "
        "function of the terminal log price. $C_0(K,T)$ is today's call value, $K$ is strike, "
        "and $\\operatorname{Re}[\\cdot]$ takes the real part of the integrand."
    ),
    "The client requests an arithmetic-average Asian call with 20 trading days to maturity and strike equal to today's spot. The current spot is included as the first observation in the average, as required by the project brief (WorldQuant University 1-2):": (
        "**Parameters (Equations 10-13).** $A_T$ is the arithmetic average through maturity "
        "$T$; $n=20$ is the number of future daily observations, $j$ indexes observation dates "
        "$t_j$, and $S_{t_j}$ is the stock price at each date, including $S_{t_0}=S_0$. "
        "$\\Pi_T$ is the Asian-call payoff and $K=S_0$ is its at-the-money strike. "
        "$\\widehat{V}_0$ is the discounted Monte Carlo value, $M=200{,}000$ is the number of "
        "paths, $m$ indexes paths, $\\Pi_T^{(m)}$ is path $m$'s payoff, and $r$ is the "
        "risk-free rate. $V_{\\mathrm{client}}$ is the quoted value after multiplying fair "
        "value by $1.04$ to include the 4% fee."
    ),
    "The Bates model augments the Heston process with a compound-Poisson jump component (Bates 69-107):": (
        "**Parameters (Equations 14-16).** $S_{t^-}$ is the stock price immediately before a "
        "jump; $r$, $v_t$, and $W_t^S$ retain their Heston meanings. $N_t$ is a Poisson jump "
        "count with intensity $\\lambda$, $J=e^Y$ is the multiplicative jump size, and "
        "$Y$ is normal with mean $\\mu_J$ and standard deviation $\\delta_J$. The compensator "
        "$\\kappa_J=\\mathbb{E}[J-1]$ preserves the risk-neutral drift. $\\varphi_J(u)$ is the "
        "jump characteristic-function multiplier at Fourier argument $u$ over maturity $T$, "
        "with $i=\\sqrt{-1}$."
    ),
    "The required strike is 95% of the current stock price (WorldQuant University 3):": (
        "**Parameters (Equation 17).** $K$ is the put strike, $S_0=232.90$ is the current SM "
        "Energy spot price, and $0.95$ is the assignment's 95% moneyness factor, giving "
        "$K=221.255$."
    ),
    "The supplied annualized Euribor observations are: 1 week 0.648%, 1 month 0.679%, 3 months 1.173%, 6 months 1.809%, and 12 months 2.556%. A cubic spline interpolates weekly annualized zero rates across one year, as required by the project brief (WorldQuant University 3-4). The CIR short-rate dynamics are (Cox, Ingersoll, and Ross 385-407):": (
        "**Parameters (Equations 18-23).** $r_t$ is the short rate at time $t$, $r_0$ is its "
        "initial value, $a$ is mean-reversion speed, $b$ is the long-run rate, $\\eta$ is rate "
        "volatility, and $W_t$ is a Brownian motion. $P(0,T)$ is today's zero-coupon bond price "
        "for maturity $T$; $A(T)$ and $B(T)$ are the CIR affine bond coefficients and "
        "$\\gamma=\\sqrt{a^2+2\\eta^2}$ is an auxiliary coefficient. $R(0,T)$ is the "
        "continuously compounded zero rate, while $\\exp$ and $\\log$ denote the exponential "
        "and natural logarithm."
    ),
    "One hundred thousand risk-neutral daily paths are simulated for one year using full truncation (WorldQuant University 4). The selected confidence level is 95%; the reported range is": (
        "**Parameters (Equations 24-25).** $r_{1\\mathrm{y}}$ is the simulated terminal rate "
        "after one year and $Q_p(\\cdot)$ is the $p$-quantile operator, so $Q_{0.025}$ and "
        "$Q_{0.975}$ form the central 95% range. $\\widehat{\\mathbb{E}}[r_{1\\mathrm{y}}]$ "
        "is the Monte Carlo estimate of the expected terminal rate; $M=100{,}000$ is the "
        "number of paths, $m$ indexes paths, and $r_{1\\mathrm{y}}^{(m)}$ is path $m$'s "
        "terminal rate."
    ),
}

SECTION_APPENDICES = {
    "## 1(i) Heston (1993) calibrated with the Lewis (2001) representation": r"""
### Calibration procedure

1. Retain exactly the five calls and five puts with 15 trading days to maturity and set $T=15/250=0.06$ years.
2. For each trial parameter vector, evaluate calls from Equation (5) and derive puts from Equation (6), using the common spot and 1.50% risk-free rate.
3. Minimize Equation (7) in dollar-price space. A seeded differential-evolution search first explores the bounded parameter region; bounded nonlinear least squares then refines the candidate using the ten individual residuals.
4. Assess the result using MSE, RMSE, option-level residuals, put-call parity, parameter bounds, and the Feller diagnostic $2\kappa\theta-\sigma^2$. This separates numerical fit from data inconsistency and parameter-identification risk.
""".strip(),
    "## 1(ii) Heston calibrated with Carr-Madan (1999)": r"""
### Comparison procedure

The maturity, observations, parameter bounds, parity convention, optimizer sequence, and unweighted MSE are held fixed. Only the Fourier pricing representation changes. This controlled design makes fitted prices, objective values, and direct same-parameter price differences valid numerical comparisons; parameter equality is a weaker criterion when the short-maturity objective is flat.
""".strip(),
    "## 1(iii) Twenty-day ATM Asian call": r"""
### Monte Carlo procedure

The selected Heston calibration is simulated under the risk-neutral measure for 20 daily steps of $\Delta t=1/250$. Full truncation replaces negative Euler variance inputs by zero, correlated Gaussian shocks preserve the fitted leverage effect, and today's spot is included before the 20 simulated observations. The analysis uses 200,000 paths with a fixed seed, discounts every payoff at the risk-free rate, and reports both estimator uncertainty and the required commercial fee.
""".strip(),
    "## 2(i) Bates (1996) calibrated with Lewis (2001)": r"""
### Calibration procedure

The ten 60-day options are converted to $T=60/250=0.24$ years. The five Heston parameters and three jump parameters are estimated jointly by the same seeded global-search/local-refinement sequence and regular price MSE used in Step 1(i). Calls use the Lewis integral with the Bates characteristic function; puts use parity. Residual, parity, boundary, and Feller diagnostics are reviewed because adding jump flexibility does not make inconsistent quotes jointly attainable.
""".strip(),
    "## 2(ii) Bates (1996) calibrated with Carr-Madan (1999)": r"""
### Comparison procedure

The Carr-Madan exercise changes only the numerical pricing transform. The 60-day sample, eight-parameter bounds, optimizer tolerances, parity construction, and objective remain unchanged. Similar MSE with different parameter vectors indicates weak identification or multiple near-minima; it does not by itself imply a pricing-engine error.
""".strip(),
    "## 3(i) CIR (1985) calibration to the Euribor term structure": r"""
### Curve-construction and calibration procedure

The five supplied annual rates are mapped to 7, 30, 90, 180, and 365 days. A natural cubic spline produces 52 weekly nodes from day 7 through day 364, with negative interpolated values floored at zero to remain consistent with CIR support. The one-week observation initializes the short rate. Differential evolution then minimizes the mean squared difference between the weekly spline and the continuously compounded CIR zero rates from Equations (19)-(23).
""".strip(),
    "## 3(ii) One-year daily CIR simulation": r"""
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
    sub_exercise_labels = {
        "1(a)": "1(i)",
        "1(b)": "1(ii)",
        "1(c)": "1(iii)",
        "2(a)": "2(i)",
        "2(b)": "2(ii)",
        "2(c)": "2(iii)",
        "3(a)": "3(i)",
        "3(b)": "3(ii)",
    }
    for cell in markdown_cells:
        for old_label, new_label in sub_exercise_labels.items():
            cell.source = cell.source.replace(old_label, new_label)

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

    for anchor, description in PARAMETER_DESCRIPTIONS.items():
        matching_cells = [cell for cell in markdown_cells if anchor in cell.source]
        if len(matching_cells) != 1:
            raise ValueError(f"Expected one parameter-description anchor for {anchor!r}.")
        cell = matching_cells[0]
        if description not in cell.source:
            cell.source = cell.source.replace(anchor, f"{anchor}\n\n{description}", 1)

    for cell in markdown_cells:
        cell.source = (
            cell.source.replace(r"\[", "$$")
            .replace(r"\]", "$$")
            .replace(r"\(", "$")
            .replace(r"\)", "$")
        )
        cell.source = re.sub(
            r"\\tag\{(?:\\scriptsize\s+)?(\d+)\}",
            r"\\tag*{$\\scriptstyle (\1)$}",
            cell.source,
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

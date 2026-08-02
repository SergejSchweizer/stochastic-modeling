"""Presentation runtime for the call-only coursework notebook.

The notebook intentionally contains only calls into this module.  Keeping the
orchestration here makes the submitted notebook readable while retaining a
reproducible, testable project implementation outside the presentation layer.
"""

import warnings
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

from stochastic_modeling import (
    CalibrationService,
    CarrMadanPricer,
    EuriborCurve,
    LewisPricer,
    calibrate_cir,
    cir_zero_rate,
    estimate_asian_call,
    load_option_data,
    model_prices,
    simulate_cir_terminal,
    simulate_heston_asian,
)

_WORKFLOW: SimpleNamespace | None = None


def _context() -> SimpleNamespace:
    if _WORKFLOW is None:
        raise RuntimeError("Call initialize_workflow() before running an analysis section.")
    return _WORKFLOW


def _save_figure(name: str) -> None:
    context = _context()
    plt.tight_layout()
    plt.savefig(context.figure_dir / name, dpi=180, bbox_inches="tight")
    plt.show()


def _prices(frame: pd.DataFrame, model, method: str = "lewis") -> np.ndarray:
    context = _context()
    return model_prices(frame, model, context.pricers[method])


def _calibrate(frame: pd.DataFrame, kind: str, method: str, seed: int):
    context = _context()
    result = CalibrationService(context.pricers[method]).calibrate(frame, kind, seed)
    return result.parameters, result.mse, result.residuals


def initialize_workflow() -> None:
    """Initialize shared assumptions and presentation styling."""
    global _WORKFLOW

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    sns.set_theme(style="whitegrid", context="notebook")
    root = Path.cwd()
    if not (root / "data").exists():
        root = root.parent
    figure_dir = root / "outputs" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    _WORKFLOW = SimpleNamespace(
        root=root,
        data_dir=root / "data",
        figure_dir=figure_dir,
        spot=232.90,
        rate=0.015,
        days_per_year=250,
        pricers={"lewis": LewisPricer(), "carr_madan": CarrMadanPricer()},
    )
    print("Workflow initialized with the project assumptions and pricing engines.")


def show_data_quality_checks() -> None:
    """Load and display the supplied option observations."""
    context = _context()
    files = sorted(context.data_dir.rglob("*.csv"))
    if not files:
        raise FileNotFoundError("Place the supplied option CSV in data/.")
    option_file = next((path for path in files if "option" in path.name.lower()), files[0])
    context.options = load_option_data(option_file)
    display(context.options)
    print(f"Source: {option_file.name}; observations: {len(context.options)}")


def show_pricing_method_configuration() -> None:
    """Display the common calibration conventions used by both Fourier methods."""
    display(
        pd.DataFrame(
            [
                {
                    "pricing method": "Lewis (2001)",
                    "put treatment": "put-call parity",
                    "calibration objective": "unweighted price MSE",
                },
                {
                    "pricing method": "Carr-Madan (1999)",
                    "put treatment": "put-call parity",
                    "calibration objective": "unweighted price MSE",
                },
            ]
        )
    )


def show_heston_lewis_calibration() -> None:
    """Calibrate Heston-Lewis jointly to the 15-day calls and puts."""
    context = _context()
    context.short = (
        context.options.query("days == 15").sort_values(["type", "strike"]).reset_index(drop=True)
    )
    calls = context.short.query("type == 'C'").sort_values("strike").set_index("strike")["price"]
    puts = context.short.query("type == 'P'").sort_values("strike").set_index("strike")["price"]
    parity_value = context.spot - calls.index.to_numpy() * np.exp(
        -context.rate * 15 / context.days_per_year
    )
    context.parity_gap = calls.to_numpy() - puts.to_numpy() - parity_value
    parity = pd.DataFrame(
        {
            "strike": calls.index,
            "C - P": calls.to_numpy() - puts.to_numpy(),
            "parity value": parity_value,
            "gap": context.parity_gap,
        }
    )
    display(parity.round(4))
    parity_rmse = np.sqrt(np.mean(context.parity_gap**2))
    print(
        f"Put-call parity RMSE: ${parity_rmse:.4f}; "
        f"irreducible joint-price RMSE floor: ${parity_rmse / 2:.4f}"
    )
    context.heston_15, context.mse_15, _ = _calibrate(context.short, "heston", "lewis", 622)
    context.fitted_15 = context.short.copy()
    context.fitted_15["model"] = _prices(context.short, context.heston_15, "lewis")
    context.fitted_15["residual"] = context.fitted_15["model"] - context.fitted_15["price"]
    context.fitted_15["squared_error"] = context.fitted_15["residual"] ** 2
    display(context.fitted_15.round(4))
    feller = 2 * context.heston_15.kappa * context.heston_15.theta - context.heston_15.sigma**2
    display(
        pd.DataFrame(
            [
                asdict(context.heston_15)
                | {
                    "MSE": context.mse_15,
                    "RMSE": np.sqrt(context.mse_15),
                    "Feller diagnostic": feller,
                }
            ]
        ).round(6)
    )


def show_heston_lewis_fit() -> None:
    """Display the 15-day fitted prices and residuals."""
    context = _context()
    _, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=False)
    for axis, option_type, name in zip(axes, ["C", "P"], ["Calls", "Puts"], strict=True):
        data = context.fitted_15.query("type == @option_type")
        axis.plot(data.strike, data.price, "o", ms=7, label="Market")
        axis.plot(data.strike, data.model, "-", lw=2, label="Heston-Lewis")
        axis.set(title=f"15-day {name}", xlabel="Strike", ylabel="Option price (USD)")
        axis.legend()
    _save_figure("step1a_market_vs_model.png")

    plt.figure(figsize=(8, 4.2))
    sns.barplot(
        data=context.fitted_15,
        x="strike",
        y="residual",
        hue="type",
        palette="Set2",
    )
    plt.axhline(0, color="black", lw=1)
    plt.title("Step 1(a): pricing residuals")
    plt.xlabel("Strike")
    plt.ylabel("Model - market (USD)")
    _save_figure("step1a_residuals.png")


def show_heston_carr_madan_calibration() -> None:
    """Calibrate and display the 15-day Carr-Madan comparison."""
    context = _context()
    context.heston_cm_15, context.mse_cm_15, _ = _calibrate(
        context.short, "heston", "carr_madan", 623
    )
    comparison = pd.DataFrame(
        [
            {
                "method": "Lewis (2001)",
                **asdict(context.heston_15),
                "MSE": context.mse_15,
                "RMSE": np.sqrt(context.mse_15),
            },
            {
                "method": "Carr-Madan (1999)",
                **asdict(context.heston_cm_15),
                "MSE": context.mse_cm_15,
                "RMSE": np.sqrt(context.mse_cm_15),
            },
        ]
    )
    display(comparison.round(6))
    check_strikes = np.array([227.5, 232.5, 237.5])
    check_frame = pd.DataFrame(
        {
            "strike": check_strikes,
            "T": 15 / context.days_per_year,
            "type": "C",
        }
    )
    numerical_check = pd.DataFrame(
        {
            "strike": check_strikes,
            "Lewis": _prices(check_frame, context.heston_15, "lewis"),
            "Carr-Madan": _prices(check_frame, context.heston_15, "carr_madan"),
        }
    )
    numerical_check["difference"] = numerical_check["Lewis"] - numerical_check["Carr-Madan"]
    display(numerical_check.round(6))
    fit = context.short.copy()
    fit["model"] = _prices(context.short, context.heston_cm_15, "carr_madan")
    _, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for axis, option_type in zip(axes, ["C", "P"], strict=True):
        data = fit.query("type == @option_type")
        axis.plot(data.strike, data.price, "o", label="Market")
        axis.plot(data.strike, data.model, "-", lw=2, label="Heston-Carr-Madan")
        axis.set(
            title=f"15-day {option_type} prices",
            xlabel="Strike",
            ylabel="Option price (USD)",
        )
        axis.legend()
    _save_figure("step1b_carr_madan_fit.png")


def show_asian_call_valuation() -> None:
    """Price and display the 20-day arithmetic-average Asian call."""
    context = _context()
    context.asian_samples = simulate_heston_asian(context.heston_15)
    estimate = estimate_asian_call(context.asian_samples)
    context.asian_fair = estimate.fair_value
    context.asian_se = estimate.standard_error
    context.asian_client = estimate.client_price()
    display(
        pd.DataFrame(
            [
                {
                    "paths": estimate.paths,
                    "fair value": context.asian_fair,
                    "standard error": context.asian_se,
                    "95% low": estimate.confidence_low,
                    "95% high": estimate.confidence_high,
                    "client price incl. 4% fee": context.asian_client,
                }
            ]
        ).round(4)
    )
    plt.figure(figsize=(8, 4.2))
    plt.hist(context.asian_samples, bins=70, color="#4C78A8", alpha=0.85)
    plt.axvline(context.asian_fair, color="#E45756", lw=2, label="Fair value")
    plt.title("Step 1(c): discounted Asian-call simulation outcomes")
    plt.xlabel("Discounted payoff (USD)")
    plt.ylabel("Frequency")
    plt.legend()
    _save_figure("step1c_asian_distribution.png")


def show_bates_lewis_calibration() -> None:
    """Calibrate and display Bates-Lewis for the 60-day options."""
    context = _context()
    context.medium = (
        context.options.query("days == 60").sort_values(["type", "strike"]).reset_index(drop=True)
    )
    context.bates_60_lewis, context.mse_bates_lewis, _ = _calibrate(
        context.medium, "bates", "lewis", 625
    )
    context.fit_bates_60 = context.medium.copy()
    context.fit_bates_60["model"] = _prices(context.medium, context.bates_60_lewis, "lewis")
    context.fit_bates_60["residual"] = context.fit_bates_60["model"] - context.fit_bates_60["price"]
    display(
        pd.DataFrame(
            [
                asdict(context.bates_60_lewis)
                | {
                    "MSE": context.mse_bates_lewis,
                    "RMSE": np.sqrt(context.mse_bates_lewis),
                }
            ]
        ).round(6)
    )
    display(context.fit_bates_60.round(4))
    _, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for axis, option_type in zip(axes, ["C", "P"], strict=True):
        data = context.fit_bates_60.query("type == @option_type")
        axis.plot(data.strike, data.price, "o", label="Market")
        axis.plot(data.strike, data.model, "-", lw=2, label="Bates-Lewis")
        axis.set(
            title=f"60-day {option_type} prices",
            xlabel="Strike",
            ylabel="Option price (USD)",
        )
        axis.legend()
    _save_figure("step2a_market_vs_model.png")


def show_bates_carr_madan_calibration() -> None:
    """Calibrate and display Bates-Carr-Madan for the 60-day options."""
    context = _context()
    context.bates_60_cm, context.mse_bates_cm, _ = _calibrate(
        context.medium, "bates", "carr_madan", 626
    )
    comparison = pd.DataFrame(
        [
            {
                "method": "Lewis (2001)",
                **asdict(context.bates_60_lewis),
                "MSE": context.mse_bates_lewis,
                "RMSE": np.sqrt(context.mse_bates_lewis),
            },
            {
                "method": "Carr-Madan (1999)",
                **asdict(context.bates_60_cm),
                "MSE": context.mse_bates_cm,
                "RMSE": np.sqrt(context.mse_bates_cm),
            },
        ]
    )
    display(comparison.round(6))
    fit = context.medium.copy()
    fit["model"] = _prices(context.medium, context.bates_60_cm, "carr_madan")
    _, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for axis, option_type in zip(axes, ["C", "P"], strict=True):
        data = fit.query("type == @option_type")
        axis.plot(data.strike, data.price, "o", label="Market")
        axis.plot(data.strike, data.model, "-", lw=2, label="Bates-Carr-Madan")
        axis.set(
            title=f"60-day {option_type} prices",
            xlabel="Strike",
            ylabel="Option price (USD)",
        )
        axis.legend()
    _save_figure("step2b_carr_madan_fit.png")


def show_seventy_day_put_valuation() -> None:
    """Price and display the 70-day put requested by the client."""
    context = _context()
    put = pd.DataFrame(
        {
            "strike": [0.95 * context.spot],
            "T": [70 / context.days_per_year],
            "type": ["P"],
        }
    )
    context.fair_put70 = float(_prices(put, context.bates_60_lewis, "lewis")[0])
    display(
        pd.DataFrame(
            [
                {
                    "spot": context.spot,
                    "strike (95% spot)": 0.95 * context.spot,
                    "maturity (days)": 70,
                    "fair put value": context.fair_put70,
                    "client price incl. 4% fee": 1.04 * context.fair_put70,
                }
            ]
        ).round(4)
    )


def show_cir_calibration() -> None:
    """Calibrate CIR to the interpolated Euribor curve and display the fit."""
    context = _context()
    curve = EuriborCurve.supplied()
    context.tenor_days = curve.tenor_days
    context.euribor = curve.annual_rates
    weekly_days, market_rates = curve.weekly()
    maturities = weekly_days / 365
    context.r0 = context.euribor[0]
    context.cir_model, context.cir_mse = calibrate_cir(maturities, market_rates, context.r0)
    context.cir = np.array(
        [
            context.cir_model.mean_reversion,
            context.cir_model.long_run_rate,
            context.cir_model.volatility,
        ]
    )
    fitted_rates = cir_zero_rate(maturities, context.cir_model, context.r0)
    display(
        pd.DataFrame(
            [
                {
                    "a (mean reversion)": context.cir[0],
                    "b (long-run rate)": context.cir[1],
                    "eta (rate volatility)": context.cir[2],
                    "MSE": context.cir_mse,
                    "Feller diagnostic 2ab-eta^2": context.cir_model.feller_diagnostic,
                }
            ]
        ).round(8)
    )
    plt.figure(figsize=(9, 4.5))
    plt.plot(weekly_days, market_rates * 100, label="Cubic-spline Euribor curve", lw=2)
    plt.plot(weekly_days, fitted_rates * 100, "--", label="CIR fitted curve", lw=2)
    plt.scatter(
        context.tenor_days,
        context.euribor * 100,
        color="black",
        zorder=3,
        label="Supplied rates",
    )
    plt.xlabel("Maturity (days)")
    plt.ylabel("Annualized zero rate (%)")
    plt.title("Step 3(a): Euribor term structure and CIR fit")
    plt.legend()
    _save_figure("step3a_cir_curve.png")


def show_cir_scenarios() -> None:
    """Simulate and display the one-year CIR terminal-rate distribution."""
    context = _context()
    context.cir_terminal = simulate_cir_terminal(context.cir_model, context.r0)
    summary = pd.DataFrame(
        [
            {
                "paths": len(context.cir_terminal),
                "current 12M Euribor": context.euribor[-1],
                "expected 12M rate in one year": context.cir_terminal.mean(),
                "95% lower bound": np.quantile(context.cir_terminal, 0.025),
                "95% upper bound": np.quantile(context.cir_terminal, 0.975),
            }
        ]
    )
    display(summary.style.format("{:.4%}"))
    plt.figure(figsize=(8, 4.2))
    plt.hist(context.cir_terminal * 100, bins=80, color="#59A14F", alpha=0.85)
    plt.axvline(
        context.cir_terminal.mean() * 100,
        color="#E15759",
        lw=2,
        label="Expected terminal rate",
    )
    plt.xlabel("12-month Euribor in one year (%)")
    plt.ylabel("Frequency")
    plt.title("Step 3(b): 100,000 simulated terminal Euribor rates")
    plt.legend()
    _save_figure("step3b_cir_distribution.png")


def refresh_narrative_sidecar() -> None:
    """Refresh README.md from the results produced during this workflow."""
    context = _context()

    def metric(value) -> str:
        return f"{value:.4f}"

    readme_lines = [
        "# MScFE 622 - Stochastic Modeling: Group Work Project 1",
        "",
        "## Executive summary",
        "",
        f"This companion presents the assumptions, results, interpretations, and figures for all project questions. SM Energy spot is ${context.spot:.2f}; the annual risk-free rate is {context.rate:.2%}; one year is {context.days_per_year} trading days; and no dividend yield is assumed.",
        "",
        "## Step 1 - 15-day derivative",
        "",
        "### 1(a) Heston calibration using Lewis (2001)",
        "",
        f"The 15-day calls and puts were fitted jointly using the Heston specification and Lewis representation (Heston 327-343; Lewis), with unweighted USD price MSE as required by the project brief (WorldQuant University 1-2). The fitted parameters are kappa={metric(context.heston_15.kappa)}, theta={metric(context.heston_15.theta)}, sigma={metric(context.heston_15.sigma)}, rho={metric(context.heston_15.rho)}, and initial variance={metric(context.heston_15.v0)}. MSE is {metric(context.mse_15)} and RMSE is ${metric(np.sqrt(context.mse_15))}.",
        "",
        f"The input quotes have a put-call parity RMSE of ${metric(np.sqrt(np.mean(context.parity_gap**2)))}; an arbitrage-consistent fit cannot make all residuals zero. Short maturity also weakens identification of long-run variance and mean reversion.",
        "",
        "![15-day market and model prices](outputs/figures/step1a_market_vs_model.png)",
        "",
        "![15-day residuals](outputs/figures/step1a_residuals.png)",
        "",
        "### 1(b) Carr-Madan comparison",
        "",
        f"The damped transform follows Carr and Madan (61-73). Its MSE is {metric(context.mse_cm_15)}, compared with {metric(context.mse_15)} for Lewis. Fitted prices and numerical stability are the meaningful comparison.",
        "",
        "![15-day Carr-Madan fit](outputs/figures/step1b_carr_madan_fit.png)",
        "",
        "### 1(c) 20-day ATM Asian call",
        "",
        f"Today's spot is included in the arithmetic average, following the project instructions (WorldQuant University 1-2). Across {len(context.asian_samples):,} risk-neutral scenarios, fair value is ${metric(context.asian_fair)} (standard error ${metric(context.asian_se)}), with a 95% estimator interval of ${metric(context.asian_fair - 1.96 * context.asian_se)} to ${metric(context.asian_fair + 1.96 * context.asian_se)}. The 4% fee produces a client price of ${metric(context.asian_client)}.",
        "",
        "![Asian-call outcome distribution](outputs/figures/step1c_asian_distribution.png)",
        "",
        "## Step 2 - 60-day derivative",
        "",
        "### 2(a) Bates calibration using Lewis",
        "",
        f"The 60-day jump-diffusion calibration follows Bates (69-107) and has MSE {metric(context.mse_bates_lewis)} and RMSE ${metric(np.sqrt(context.mse_bates_lewis))}.",
        "",
        "![60-day market and model prices](outputs/figures/step2a_market_vs_model.png)",
        "",
        "### 2(b) Carr-Madan comparison",
        "",
        f"The matching Carr-Madan calibration follows Carr and Madan (61-73) and has MSE {metric(context.mse_bates_cm)}. The comparison is based on fitted-price errors and numerical stability.",
        "",
        "![60-day Carr-Madan fit](outputs/figures/step2b_carr_madan_fit.png)",
        "",
        "### 2(c) 70-day 95%-moneyness put",
        "",
        f"The project specifies a strike of 95% of spot (WorldQuant University 3), giving ${0.95 * context.spot:.2f}. Fair value is ${metric(context.fair_put70)} and the price after the 4% fee is ${metric(1.04 * context.fair_put70)}.",
        "",
        "## Step 3 - Euribor rates",
        "",
        "### 3(a) CIR calibration",
        "",
        f"The supplied curve contains 0.648% (1 week), 0.679% (1 month), 1.173% (3 months), 1.809% (6 months), and 2.556% (12 months). Weekly interpolation follows the project brief (WorldQuant University 3-4), and the rate model follows Cox, Ingersoll, and Ross (385-407). The fitted values are a={metric(context.cir[0])}, b={metric(context.cir[1])}, eta={metric(context.cir[2])}; MSE is {context.cir_mse:.8f}.",
        "",
        "![Euribor curve and CIR fit](outputs/figures/step3a_cir_curve.png)",
        "",
        "### 3(b) One-year rate scenarios",
        "",
        f"Following the required simulation size (WorldQuant University 4), 100,000 daily scenarios give an expected terminal 12-month Euribor of {context.cir_terminal.mean():.4%}; the 95% range is {np.quantile(context.cir_terminal, 0.025):.4%} to {np.quantile(context.cir_terminal, 0.975):.4%}. Higher expected rates increase discounting of future positive cash flows and generally lower present values, all else equal.",
        "",
        "![Terminal Euribor distribution](outputs/figures/step3b_cir_distribution.png)",
        "",
        "## Step 4 - Submission checklist",
        "",
        "The analytical material, figures, and references are organized for the required report and archive. The presentation export suppresses inputs while retaining explanations, tables, and figures.",
        "",
        "## Works Cited",
        "",
        '- Bates, David S. "Jumps and Stochastic Volatility: Exchange Rate Processes Implicit in Deutsche Mark Options." The Review of Financial Studies, vol. 9, no. 1, 1996, pp. 69-107. https://doi.org/10.1093/rfs/9.1.69.',
        "",
        '- Carr, Peter, and Dilip B. Madan. "Option Valuation Using the Fast Fourier Transform." The Journal of Computational Finance, vol. 2, no. 4, 1999, pp. 61-73.',
        "",
        '- Cox, John C., Jonathan E. Ingersoll, Jr., and Stephen A. Ross. "A Theory of the Term Structure of Interest Rates." Econometrica, vol. 53, no. 2, 1985, pp. 385-407. https://doi.org/10.2307/1911242.',
        "",
        '- Heston, Steven L. "A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options." The Review of Financial Studies, vol. 6, no. 2, 1993, pp. 327-343. https://doi.org/10.1093/rfs/6.2.327.',
        "",
        '- Lewis, Alan L. "A Simple Option Formula for General Jump-Diffusion and Other Exponential Levy Processes." SSRN, 2001, https://ssrn.com/abstract=282110.',
        "",
        "- WorldQuant University. MScFE 622 Stochastic Modeling: Group Work Project 1. 2022. Course handout.",
        "",
    ]
    (context.root / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    print("README.md refreshed with the current narrative and results.")


NOTEBOOK_FUNCTIONS = (
    show_data_quality_checks,
    show_pricing_method_configuration,
    show_heston_lewis_calibration,
    show_heston_lewis_fit,
    show_heston_carr_madan_calibration,
    show_asian_call_valuation,
    show_bates_lewis_calibration,
    show_bates_carr_madan_calibration,
    show_seventy_day_put_valuation,
    show_cir_calibration,
    show_cir_scenarios,
    refresh_narrative_sidecar,
)


def load_ipython_extension(ipython) -> None:
    """Expose the presentation calls when the runtime is loaded by Jupyter."""
    initialize_workflow()
    ipython.push({function.__name__: function for function in NOTEBOOK_FUNCTIONS})

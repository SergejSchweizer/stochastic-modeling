"""Presentation runtime for the call-only coursework notebook.

The notebook intentionally contains only calls into this module.  Keeping the
orchestration here makes the submitted notebook readable while retaining a
reproducible, testable project implementation outside the presentation layer.
"""

import warnings
from dataclasses import asdict
from html import escape
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import HTML, Markdown, display
from matplotlib.collections import LineCollection

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


def _save_figure(name: str, description: str) -> None:
    context = _context()
    context.plot_number += 1
    figure = plt.gcf()
    plt.tight_layout()
    figure.savefig(context.figure_dir / name, dpi=180, bbox_inches="tight")
    display(figure, metadata={"image/png": {"alt": description}})
    plt.close(figure)
    display(
        HTML(
            '<p class="plot-description" style="font-size: 8pt !important;">'
            f"<strong>Plot {context.plot_number}.</strong> {escape(description)}</p>"
        )
    )


def _prices(frame: pd.DataFrame, model, method: str = "lewis") -> np.ndarray:
    context = _context()
    return model_prices(frame, model, context.pricers[method])


def _calibrate(frame: pd.DataFrame, kind: str, method: str, seed: int):
    context = _context()
    result = CalibrationService(context.pricers[method]).calibrate(frame, kind, seed)
    return result.parameters, result.mse, result.residuals


def _show_discussion(title: str, *paragraphs: str) -> None:
    """Render a concise, result-aware interpretation beneath an output."""
    display(Markdown(f"#### {title}\n\n" + "\n\n".join(paragraphs)))


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
        plot_number=0,
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
    largest_residual = context.fitted_15.iloc[context.fitted_15["residual"].abs().argmax()]
    parity_floor = parity_rmse / 2
    _show_discussion(
        "Calibration discussion",
        f"**Fit quality.** The joint calibration has RMSE ${np.sqrt(context.mse_15):.4f}. "
        f"The quotes imply a minimum parity-consistent joint RMSE near ${parity_floor:.4f}, "
        "so most remaining error is explained by the internal call-put inconsistency rather "
        f"than Fourier integration. The largest residual is "
        f"${largest_residual.residual:+.4f} for the {largest_residual.type} option at strike "
        f"${largest_residual.strike:.2f}. In the fitted-price plot, call values decline and "
        "put values rise as strike increases, as expected. A positive residual means the "
        "model overprices the quote, whereas a negative residual means it underprices it; "
        "the residual chart therefore reveals systematic differences that the headline RMSE "
        "alone would conceal.",
        f"**Parameter interpretation.** Initial variance {context.heston_15.v0:.4f} implies "
        f"annualized spot volatility of {np.sqrt(context.heston_15.v0):.2%}. The strongly "
        f"negative correlation rho={context.heston_15.rho:.4f} produces the leverage effect, "
        f"while sigma={context.heston_15.sigma:.4f} permits rapid variance changes. The "
        f"long-run variance theta={context.heston_15.theta:.4f} corresponds to only "
        f"{np.sqrt(context.heston_15.theta):.2%} volatility and lies at its lower bound. "
        "Economically, the negative correlation makes adverse stock-price shocks coincide "
        "with rising variance, which raises the relative value of downside protection and "
        "helps generate the observed volatility skew.",
        f"**Identification and model risk.** The Feller diagnostic is {feller:.4f}, below "
        "zero, so fitted variance can approach the boundary. Fifteen-day options weakly "
        "identify kappa and theta; these values reproduce this cross-section but should not "
        "be interpreted as reliable long-horizon forecasts. The calibration should therefore "
        "be judged primarily by near-term fitted prices and residuals, not by attaching strong "
        "economic meaning to every parameter estimate.",
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
    _save_figure(
        "step1a_market_vs_model.png",
        "Market quotes are shown as points and Heston-Lewis fitted prices as curves; their "
        "distance shows the quality of the joint call-and-put calibration.",
    )

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
    _save_figure(
        "step1a_residuals.png",
        "Signed pricing errors by strike and option type reveal where the calibrated model "
        "overprices or underprices the observed 15-day contracts.",
    )


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
    maximum_pricing_difference = numerical_check["difference"].abs().max()
    _show_discussion(
        "Lewis versus Carr-Madan",
        f"**Numerical agreement.** The calibrations reach essentially identical losses: MSE "
        f"{context.mse_15:.6f} for Lewis and {context.mse_cm_15:.6f} for Carr-Madan. Holding "
        "the Lewis parameters fixed, the largest cross-method difference at the diagnostic "
        f"strikes is only ${maximum_pricing_difference:.6f}. Because both methods evaluate "
        "the same risk-neutral Heston distribution through different transforms, this close "
        "agreement is a useful implementation check: numerical integration choice is not "
        "driving the reported option values.",
        f"**Why parameters differ.** Despite equal prices, kappa changes from "
        f"{context.heston_15.kappa:.4f} to {context.heston_cm_15.kappa:.4f}. Short-dated "
        "options mainly identify current variance, skew, and short-run variance-of-variance; "
        "many kappa-theta combinations generate nearly the same 15-day distribution. This "
        "is evidence of an objective-function ridge, not a material disagreement between the "
        "two Fourier formulas. Consequently, fitted prices are stable even when individual "
        "parameters are not; risk management should emphasize price sensitivities and "
        "recalibration behavior rather than selecting a method because its kappa appears more "
        "plausible.",
    )
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
    _save_figure(
        "step1b_carr_madan_fit.png",
        "The Carr-Madan calibration is compared with the same 15-day market quotes to verify "
        "the alternative Fourier pricing route.",
    )


def show_asian_call_valuation() -> None:
    """Price and display the 20-day arithmetic-average Asian call."""
    context = _context()
    context.asian_samples, context.asian_paths = simulate_heston_asian(
        context.heston_15, return_paths=True
    )
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
    path_days = np.arange(context.asian_paths.shape[0])
    path_segments = np.stack(
        (
            np.broadcast_to(path_days, context.asian_paths.T.shape),
            context.asian_paths.T,
        ),
        axis=-1,
    )
    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.add_collection(
        LineCollection(
            path_segments,
            colors="#4C78A8",
            linewidths=0.08,
            alpha=0.01,
            rasterized=True,
        )
    )
    axis.set(
        xlim=(path_days[0], path_days[-1]),
        ylim=(context.asian_paths.min(), context.asian_paths.max()),
        xlabel="Trading day",
        ylabel="SM Energy price (USD)",
        title="Step 1(c): all 200,000 simulated Heston price paths",
    )
    _save_figure(
        "step1c_asian_paths.png",
        "All 200,000 simulated SM Energy paths are shown from today's spot through the "
        "20-day Asian-option horizon; darker regions indicate greater path concentration.",
    )
    plt.figure(figsize=(8, 4.2))
    plt.hist(context.asian_samples, bins=70, color="#4C78A8", alpha=0.85)
    plt.axvline(context.asian_fair, color="#E45756", lw=2, label="Fair value")
    plt.title("Step 1(c): discounted Asian-call simulation outcomes")
    plt.xlabel("Discounted payoff (USD)")
    plt.ylabel("Frequency")
    plt.legend()
    _save_figure(
        "step1c_asian_distribution.png",
        "The histogram shows simulated discounted Asian-call payoffs; the vertical line marks "
        "their mean, which is the estimated fair value.",
    )
    _show_discussion(
        "Valuation and client recommendation",
        f"**Simulation precision.** The fair value is ${context.asian_fair:.4f}; its standard "
        f"error is ${context.asian_se:.4f}, or {context.asian_se / context.asian_fair:.2%} of "
        f"the estimate. The 95% Monte Carlo interval of ${estimate.confidence_low:.4f} to "
        f"${estimate.confidence_high:.4f} is narrow enough that simulation noise is immaterial "
        "relative to calibration and model risk. The path plot shows every simulated daily "
        "trajectory, with the darkest region indicating the outcomes generated most often. "
        "The payoff histogram is right-skewed and "
        "contains many zero or small outcomes because the call pays only when the arithmetic "
        "average exceeds the strike. The vertical line is the probability-weighted mean of "
        "those outcomes, not the most likely realized payoff.",
        f"**Client quote.** Applying the required 4% fee gives ${context.asian_client:.4f}. "
        "In plain language, the quote is based on many plausible daily price paths inferred "
        "from traded options; averaging prices dampens the effect of any single day's move. "
        "The confidence interval measures numerical precision, not a guaranteed range for the "
        "eventual payoff. The 4% fee compensates the seller above model fair value but does not "
        "eliminate exposure to parameter uncertainty, discrete hedging, transaction costs, or "
        "a future volatility regime that differs from the calibration sample.",
    )


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
    calls = context.medium.query("type == 'C'").sort_values("strike").set_index("strike")["price"]
    puts = context.medium.query("type == 'P'").sort_values("strike").set_index("strike")["price"]
    parity_value = context.spot - calls.index.to_numpy() * np.exp(
        -context.rate * 60 / context.days_per_year
    )
    context.medium_parity_rmse = float(
        np.sqrt(np.mean((calls.to_numpy() - puts.to_numpy() - parity_value) ** 2))
    )
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
    _save_figure(
        "step2a_market_vs_model.png",
        "Observed 60-day option prices are compared with Bates-Lewis values, showing the fit "
        "after jump risk is added to stochastic volatility.",
    )
    bates_feller = (
        2 * context.bates_60_lewis.kappa * context.bates_60_lewis.theta
        - context.bates_60_lewis.sigma**2
    )
    jump_probability = 1 - np.exp(
        -context.bates_60_lewis.jump_intensity * 60 / context.days_per_year
    )
    _show_discussion(
        "Bates calibration discussion",
        f"**Fit quality.** The 60-day RMSE is ${np.sqrt(context.mse_bates_lewis):.4f}. "
        "The observed calls are not monotonically decreasing at the lowest strikes, and the "
        f"call-put pairs have parity RMSE ${context.medium_parity_rmse:.4f}; an "
        "arbitrage-consistent model therefore cannot match every quote exactly. The plotted "
        "Bates curves smooth across these noisy or inconsistent observations, so visible gaps "
        "should not automatically be read as integration failure. They show the compromise "
        "required when one parameter vector is fitted jointly to all calls and puts.",
        f"**Jump interpretation.** Intensity {context.bates_60_lewis.jump_intensity:.4f} "
        f"implies a {jump_probability:.2%} probability of at least one jump over 60 trading "
        f"days. Mean log-jump {context.bates_60_lewis.jump_mean:.4f} is positive, while jump "
        f"volatility {context.bates_60_lewis.jump_vol:.4f} is at its lower bound. The optimizer "
        "is using an almost deterministic jump component to absorb shape in an inconsistent "
        "price cross-section. Thus the fitted jump probability describes the model's chosen "
        "risk-neutral mechanism for matching prices; it is neither a historical forecast of "
        "real-world jump frequency nor direct evidence that upward jumps are expected.",
        f"**Stability warning.** rho={context.bates_60_lewis.rho:.3f} is at the upper bound, "
        f"theta and v0 are at their lower bounds, and the Feller diagnostic is "
        f"{bates_feller:.4f}. The fit is usable for this near-maturity exercise, but the "
        "individual parameters should not be extrapolated as economic forecasts. Several "
        "parameters pressing against bounds also indicate that the available strikes do not "
        "contain enough independent information to identify all eight Bates parameters "
        "reliably; small quote changes may therefore produce large parameter changes.",
    )


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
    _show_discussion(
        "Bates pricing-method comparison",
        f"**Price fit.** Lewis and Carr-Madan produce nearly the same loss: MSE "
        f"{context.mse_bates_lewis:.6f} versus {context.mse_bates_cm:.6f}. This supports the "
        "numerical implementation of both characteristic-function integrations. The nearly "
        "overlapping fitted curves imply that the practical valuation conclusion is robust to "
        "the Fourier representation, even though neither representation can remove defects in "
        "the market cross-section.",
        f"**Parameter stability.** Jump intensity and mean jump size are nearly unchanged, "
        f"but kappa moves from {context.bates_60_lewis.kappa:.4f} to "
        f"{context.bates_60_cm.kappa:.4f}. Because both solutions sit on several bounds and "
        "have almost equal MSE, the objective has a flat or multi-modal direction. Price "
        "agreement is more credible than equality of every parameter. This distinction matters "
        "for use: either method can support interpolation near the calibrated contracts, but "
        "parameter-dependent stress tests or long-maturity extrapolations require additional "
        "stability checks.",
    )
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
    _save_figure(
        "step2b_carr_madan_fit.png",
        "The Bates-Carr-Madan curves are plotted against the 60-day quotes as a numerical "
        "cross-check of the Lewis-based calibration.",
    )


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
    _show_discussion(
        "Pricing interpretation",
        f"The assignment's 95% moneyness convention gives K/S0=0.95 and strike "
        f"${0.95 * context.spot:.3f}. The selected Bates-Lewis calibration gives fair value "
        f"${context.fair_put70:.4f}; adding the 4% fee produces a client price of "
        f"${1.04 * context.fair_put70:.4f}. Because the strike is below spot, the put begins "
        "out of the money and its value is entirely time value: the premium reflects the "
        "risk-neutral probability and severity of SM Energy falling below the strike before "
        "expiry, including the downside-tail contribution generated by stochastic volatility "
        "and jumps.",
        "The maturity is ten trading days beyond the 60-day calibration set. This is a modest "
        "extrapolation, but the boundary-sensitive Bates parameters make model risk more "
        "important than numerical integration error. A production quote should be checked "
        "against fresh 60- to 70-day market data where available. The client price is therefore "
        "a model-based indication rather than a guaranteed hedge cost; liquidity, bid-ask "
        "spreads, recalibration risk, and the desk's hedging expenses may justify an additional "
        "commercial reserve beyond the stated fee.",
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
    _save_figure(
        "step3a_cir_curve.png",
        "Supplied Euribor tenors, the weekly interpolated curve, and the fitted CIR curve show "
        "how closely the rate model reproduces the observed term structure.",
    )
    half_life = np.log(2) / context.cir_model.mean_reversion
    curve_rmse_bps = np.sqrt(context.cir_mse) * 10_000
    _show_discussion(
        "CIR calibration discussion",
        f"**Parameter interpretation.** Mean reversion a={context.cir[0]:.4f} implies a "
        f"half-life of about {half_life:.2f} years. Long-run level b={context.cir[1]:.4%} "
        f"lies above the current one-year Euribor of {context.euribor[-1]:.4%}, creating "
        f"upward drift. Volatility eta={context.cir[2]:.4f} is large relative to the rate "
        "level and drives a wide, right-skewed forecast distribution. Mean reversion pulls "
        "rates toward b rather than allowing shocks to persist indefinitely, while the "
        "square-root volatility term makes absolute uncertainty smaller when rates are near "
        "zero and preserves non-negative simulated rates.",
        f"**Fit and limitation.** The curve RMSE is approximately {curve_rmse_bps:.2f} basis "
        f"points, so the fitted line tracks the weekly spline closely. However, the Feller "
        f"diagnostic is {context.cir_model.feller_diagnostic:.4f}, below zero, meaning the "
        "process can reach the zero boundary. The excellent in-sample fit does not remove "
        "long-horizon uncertainty inferred from only five supplied tenors. Moreover, the CIR "
        "curve is fitted to spline-interpolated observations, so the dense weekly line adds "
        "visual smoothness rather than new market information between the original tenor "
        "points.",
    )


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
    display(
        summary.style.format(
            {
                "paths": "{:,.0f}",
                "current 12M Euribor": "{:.4%}",
                "expected 12M rate in one year": "{:.4%}",
                "95% lower bound": "{:.4%}",
                "95% upper bound": "{:.4%}",
            }
        )
    )
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
    _save_figure(
        "step3b_cir_distribution.png",
        "The histogram summarizes 100,000 simulated one-year terminal rates; the vertical line "
        "marks the expected terminal Euribor rate.",
    )
    expected_change_bps = (context.cir_terminal.mean() - context.euribor[-1]) * 10_000
    lower, upper = np.quantile(context.cir_terminal, [0.025, 0.975])
    _show_discussion(
        "Rate-scenario discussion",
        f"**Distribution.** Expected one-year terminal rate "
        f"{context.cir_terminal.mean():.4%} is {expected_change_bps:.1f} basis points above "
        f"current 12-month Euribor {context.euribor[-1]:.4%}. The selected 95% interval is "
        f"{lower:.4%} to {upper:.4%}. Its zero lower endpoint and long upper tail follow from "
        "the non-negative square-root process combined with high fitted volatility. The mean "
        "sits to the right of the histogram's most concentrated region because relatively rare "
        "high-rate scenarios pull the arithmetic average upward; the interval is a model-based "
        "scenario range, not a confidence interval for the estimated mean.",
        "**Pricing implication.** If higher rates feed into the future discount curve, the "
        "present value of fixed positive cash flows falls and financing assumptions change. "
        "Option values move through both discounting and risk-neutral drift, so the direction "
        "is product-dependent rather than universally negative. The simulated short rate is "
        "used as a proxy for the requested 12-month Euribor; a production multi-curve model "
        "would represent that tenor and its spread explicitly. Decisions based on the tail "
        "scenarios should also account for parameter and curve-construction uncertainty, which "
        "is not included in the conditional Monte Carlo distribution shown here.",
    )


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
        "Heston represents equity returns with a randomly changing, mean-reverting variance process that can reproduce volatility clustering and implied-volatility skew. Lewis converts its characteristic function into a one-dimensional Fourier pricing integral.",
        "",
        f"The 15-day calls and puts were fitted jointly using the Heston specification and Lewis representation (Heston 327-343; Lewis), with unweighted USD price MSE as required by the project brief (WorldQuant University 1-2). The fitted parameters are kappa={metric(context.heston_15.kappa)}, theta={metric(context.heston_15.theta)}, sigma={metric(context.heston_15.sigma)}, rho={metric(context.heston_15.rho)}, and initial variance={metric(context.heston_15.v0)}. MSE is {metric(context.mse_15)} and RMSE is ${metric(np.sqrt(context.mse_15))}.",
        "",
        f"The input quotes have a put-call parity RMSE of ${metric(np.sqrt(np.mean(context.parity_gap**2)))}; an arbitrage-consistent fit cannot make all residuals zero. Short maturity also weakens identification of long-run variance and mean reversion.",
        "",
        f"Initial variance implies {np.sqrt(context.heston_15.v0):.2%} annualized volatility and rho={context.heston_15.rho:.4f} indicates a strong leverage effect. Theta is at its lower bound and the Feller diagnostic is {2 * context.heston_15.kappa * context.heston_15.theta - context.heston_15.sigma**2:.4f}; the parameter vector should therefore be treated as a short-horizon fit rather than a long-run forecast.",
        "",
        "![15-day market and model prices](outputs/figures/step1a_market_vs_model.png)",
        "",
        "![15-day residuals](outputs/figures/step1a_residuals.png)",
        "",
        "### 1(b) Carr-Madan comparison",
        "",
        "Carr-Madan damps the option-price function so it can be evaluated through an integrable Fourier transform, providing an independent pricing and calibration route for the same Heston dynamics.",
        "",
        f"The damped transform follows Carr and Madan (61-73). Its MSE is {metric(context.mse_cm_15)}, compared with {metric(context.mse_15)} for Lewis. Fitted prices and numerical stability are the meaningful comparison.",
        "",
        f"Kappa changes from {context.heston_15.kappa:.4f} under Lewis to {context.heston_cm_15.kappa:.4f} under Carr-Madan even though the losses are identical to four decimals. This is consistent with weak short-maturity identification: the two numerical formulas agree on prices, while the optimizer can move along a flat parameter direction.",
        "",
        "![15-day Carr-Madan fit](outputs/figures/step1b_carr_madan_fit.png)",
        "",
        "### 1(c) 20-day ATM Asian call",
        "",
        "The arithmetic-average Asian payoff reduces dependence on a single terminal stock price. Monte Carlo simulation is required because the payoff depends on the complete daily path.",
        "",
        f"Today's spot is included in the arithmetic average, following the project instructions (WorldQuant University 1-2). Across {len(context.asian_samples):,} risk-neutral scenarios, fair value is ${metric(context.asian_fair)} (standard error ${metric(context.asian_se)}), with a 95% estimator interval of ${metric(context.asian_fair - 1.96 * context.asian_se)} to ${metric(context.asian_fair + 1.96 * context.asian_se)}. The 4% fee produces a client price of ${metric(context.asian_client)}.",
        "",
        f"The standard error is {context.asian_se / context.asian_fair:.2%} of fair value, so simulation noise is small relative to calibration risk. The interval measures estimator precision and is not a guaranteed payoff range for the client.",
        "",
        "![All simulated Asian-call paths](outputs/figures/step1c_asian_paths.png)",
        "",
        "![Asian-call outcome distribution](outputs/figures/step1c_asian_distribution.png)",
        "",
        "## Step 2 - 60-day derivative",
        "",
        "### 2(a) Bates calibration using Lewis",
        "",
        "Bates extends Heston with random discontinuous price jumps, allowing the model to represent rare abrupt moves in addition to ordinary stochastic-volatility dynamics.",
        "",
        f"The 60-day jump-diffusion calibration follows Bates (69-107) and has MSE {metric(context.mse_bates_lewis)} and RMSE ${metric(np.sqrt(context.mse_bates_lewis))}.",
        "",
        f"The call-put pairs have parity RMSE ${context.medium_parity_rmse:.4f}. Rho is at its upper bound, theta and initial variance are at lower bounds, and jump volatility is at its lower bound. These boundary estimates and a Feller diagnostic of {2 * context.bates_60_lewis.kappa * context.bates_60_lewis.theta - context.bates_60_lewis.sigma**2:.4f} show that the eight parameters are weakly identified by this small, internally inconsistent cross-section.",
        "",
        "![60-day market and model prices](outputs/figures/step2a_market_vs_model.png)",
        "",
        "### 2(b) Carr-Madan comparison",
        "",
        "This specification retains the Bates jump and variance dynamics but uses the Carr-Madan transform as a numerical cross-check of Lewis pricing.",
        "",
        f"The matching Carr-Madan calibration follows Carr and Madan (61-73) and has MSE {metric(context.mse_bates_cm)}. The comparison is based on fitted-price errors and numerical stability.",
        "",
        f"Jump intensity and mean jump size remain stable across methods, but kappa changes from {context.bates_60_lewis.kappa:.4f} to {context.bates_60_cm.kappa:.4f}. Nearly equal loss with very different mean reversion indicates a flat or multi-modal objective; fitted prices are more reliable than parameter equality.",
        "",
        "![60-day Carr-Madan fit](outputs/figures/step2b_carr_madan_fit.png)",
        "",
        "### 2(c) 70-day 95%-moneyness put",
        "",
        f"The project specifies a strike of 95% of spot (WorldQuant University 3), giving ${0.95 * context.spot:.2f}. Fair value is ${metric(context.fair_put70)} and the price after the 4% fee is ${metric(1.04 * context.fair_put70)}.",
        "",
        "The 70-day maturity is a ten-trading-day extrapolation beyond the calibration sample. The extension is modest, but the boundary-sensitive Bates estimates make calibration risk more material than the difference between the two Fourier integration methods.",
        "",
        "## Step 3 - Euribor rates",
        "",
        "### 3(a) CIR calibration",
        "",
        "CIR models a non-negative, mean-reverting short rate and connects its parameters to the Euribor curve through closed-form zero-coupon bond prices.",
        "",
        f"The supplied curve contains 0.648% (1 week), 0.679% (1 month), 1.173% (3 months), 1.809% (6 months), and 2.556% (12 months). Weekly interpolation follows the project brief (WorldQuant University 3-4), and the rate model follows Cox, Ingersoll, and Ross (385-407). The fitted values are a={metric(context.cir[0])}, b={metric(context.cir[1])}, eta={metric(context.cir[2])}; MSE is {context.cir_mse:.8f}.",
        "",
        f"Mean reversion implies a half-life of {np.log(2) / context.cir[0]:.2f} years, while the long-run rate {context.cir[1]:.4%} lies above current 12-month Euribor. Curve RMSE is {np.sqrt(context.cir_mse) * 10_000:.2f} basis points, but the Feller diagnostic {context.cir_model.feller_diagnostic:.4f} is negative, so the process can reach zero and the broad dynamics remain uncertain.",
        "",
        "![Euribor curve and CIR fit](outputs/figures/step3a_cir_curve.png)",
        "",
        "### 3(b) One-year rate scenarios",
        "",
        "Daily CIR simulation turns the calibrated rate dynamics into a one-year distribution, supporting an expected-rate estimate and an explicit measure of rate uncertainty.",
        "",
        f"Following the required simulation size (WorldQuant University 4), 100,000 daily scenarios give an expected terminal 12-month Euribor of {context.cir_terminal.mean():.4%}; the 95% range is {np.quantile(context.cir_terminal, 0.025):.4%} to {np.quantile(context.cir_terminal, 0.975):.4%}. Higher expected rates increase discounting of future positive cash flows and generally lower present values, all else equal.",
        "",
        f"The expected rate is {(context.cir_terminal.mean() - context.euribor[-1]) * 10_000:.1f} basis points above today's 12-month quote. The distribution is strongly right-skewed: the lower endpoint is pinned at zero while the upper tail reaches high rates. Product effects are not universal because rates influence both discounting and risk-neutral drift; the CIR short rate is used here as a proxy for 12-month Euribor rather than a full multi-curve tenor model.",
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

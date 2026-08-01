# MScFE 622 - Stochastic Modeling: Group Work Project 1

## Executive summary

This companion presents the assumptions, results, interpretations, and figures for all project questions. SM Energy spot is $232.90; the annual risk-free rate is 1.50%; one year is 250 trading days; and no dividend yield is assumed.

## Step 1 - 15-day derivative

### 1(a) Heston calibration using Lewis (2001)

The 15-day calls and puts were fitted jointly using unweighted USD price MSE. The fitted parameters are kappa=2.2032, theta=0.0010, sigma=1.2733, rho=-0.9444, and initial variance=0.1084. MSE is 0.2458 and RMSE is $0.4958.

The input quotes have a put-call parity RMSE of $0.9068; an arbitrage-consistent fit cannot make all residuals zero. Short maturity also weakens identification of long-run variance and mean reversion.

![15-day market and model prices](outputs/figures/step1a_market_vs_model.png)

![15-day residuals](outputs/figures/step1a_residuals.png)

### 1(b) Carr-Madan comparison

The Carr-Madan MSE is 0.2458, compared with 0.2458 for Lewis. Fitted prices and numerical stability are the meaningful comparison.

![15-day Carr-Madan fit](outputs/figures/step1b_carr_madan_fit.png)

### 1(c) 20-day ATM Asian call

Today's spot is included in the arithmetic average. Across 200,000 risk-neutral scenarios, fair value is $4.8177 (standard error $0.0130), with a 95% estimator interval of $4.7922 to $4.8432. The 4% fee produces a client price of $5.0104.

![Asian-call outcome distribution](outputs/figures/step1c_asian_distribution.png)

## Step 2 - 60-day derivative

### 2(a) Bates calibration using Lewis

The 60-day Bates calibration has MSE 1.3324 and RMSE $1.1543.

![60-day market and model prices](outputs/figures/step2a_market_vs_model.png)

### 2(b) Carr-Madan comparison

The matching Carr-Madan MSE is 1.3335. The comparison is based on fitted-price errors and numerical stability.

![60-day Carr-Madan fit](outputs/figures/step2b_carr_madan_fit.png)

### 2(c) 70-day 95%-moneyness put

The strike is $221.25. Fair value is $8.8187 and the price after the 4% fee is $9.1715.

## Step 3 - Euribor rates

### 3(a) CIR calibration

The supplied curve contains 0.648% (1 week), 0.679% (1 month), 1.173% (3 months), 1.809% (6 months), and 2.556% (12 months). The fitted CIR values are a=0.7525, b=0.0742, eta=0.5638; MSE is 0.00000062.

![Euribor curve and CIR fit](outputs/figures/step3a_cir_curve.png)

### 3(b) One-year rate scenarios

Across 100,000 daily scenarios, the expected terminal 12-month Euribor is 4.2289%; the 95% range is 0.0000% to 25.0856%. Higher expected rates increase discounting of future positive cash flows and generally lower present values, all else equal.

![Terminal Euribor distribution](outputs/figures/step3b_cir_distribution.png)

## Step 4 - Submission checklist

The analytical material, figures, and references are organized for the required report and archive. The presentation export suppresses inputs while retaining explanations, tables, and figures.

## References

- Bates (1996); Carr and Madan (1999); Cox, Ingersoll, and Ross (1985); Heston (1993); Lewis (2001).

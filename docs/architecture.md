# Architecture

The project separates reusable quantitative logic from report orchestration.

```text
src/stochastic_modeling/
  config.py       shared market conventions
  models.py       immutable Heston, Bates, and CIR parameter objects
  data.py         quote normalization and parity diagnostics
  fourier.py      characteristic functions and pricing strategies
  calibration.py two-stage calibration application service
  simulation.py  risk-neutral Monte Carlo engines
  rates.py        Euribor interpolation and CIR calibration

scripts/
  generate_coursework_notebook.py  reproducible report builder
  quality.py                       local quality-gate entry point
  validate_artifacts.py            presentation integrity checks
```

## Design decisions

- **Strategy pattern:** `LewisPricer` and `CarrMadanPricer` implement the same pricing interface, allowing the calibration service to switch numerical methods without branching through model logic.
- **Application service:** `CalibrationService` owns the global-then-local optimization workflow while receiving its pricing strategy as a dependency.
- **Value objects:** frozen, slotted parameter dataclasses make calibrated states explicit and prevent accidental mutation.
- **Single source of truth:** the package owns formulas and simulations; the notebook invokes those APIs and focuses on questions, results, and interpretation.
- **Deterministic boundaries:** fixed seeds and a locked environment make calibration and Monte Carlo results reproducible.
- **Quality at the boundary:** raw data validation, artifact validation, branch-aware coverage, and protected-branch checks prevent invalid results from reaching `main`.

# Contributing

## Environment

Synchronize the pinned Python 3.14 environment:

```powershell
.venv\Scripts\uv.exe sync --frozen --all-groups
.venv\Scripts\python.exe -m playwright install chromium
```

## Local quality gate

Run the same gate used by GitHub Actions:

```powershell
.venv\Scripts\python.exe scripts\quality.py
```

The gate checks formatting, lint rules, the complete test suite, branch-aware coverage of at least 95%, artifact integrity, and live browser rendering of the numbered equations and plot descriptions.

## Artifact refresh

Regenerate and execute the deliverables before opening a pull request:

```powershell
.venv\Scripts\python.exe scripts\generate_coursework_notebook.py
.venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace notebooks\MScFE622_GWP1.ipynb --ExecutePreprocessor.timeout=1800
.venv\Scripts\jupyter.exe nbconvert --to html --config notebooks\nbconvert_hide_input.py --output MScFE622_GWP1 --output-dir outputs notebooks\MScFE622_GWP1.ipynb
.venv\Scripts\python.exe scripts\finalize_html_export.py
.venv\Scripts\python.exe scripts\generate_pdf_report.py
```

All changes reach `main` through a pull request. The protected branch accepts a merge only after the required `quality` check succeeds.

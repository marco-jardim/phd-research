# Agent Guide (phd-research)

This repository currently contains research notebooks (under `notebooks/`) and some docs (under `docs/`).
There is no detected project-level build system (no `pyproject.toml`, `requirements.txt`, `package.json`, `Makefile`, CI workflows, or test config found at repo root).

Because agentic tools need deterministic commands and conventions, this file defines a "minimal contract" that should work on a clean machine and also documents how to run individual notebook/scripts once they exist.

## Quick Start (Safe Defaults)

- Use Python 3.10+ (3.11 recommended).
- Work in a virtual environment.
- Keep executions reproducible: pin deps when you add them.

### Create / activate venv

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Windows cmd
.venv\Scripts\activate.bat
# macOS/Linux
source .venv/bin/activate
```

### Install dependencies (when present)

This repo does not currently ship dependency files, so install only what you need for the task.
If you add dependencies, also add a lock/pin file (see "Project Hygiene" below).

Common baseline for notebooks:

```bash
python -m pip install -U pip
python -m pip install jupyter ipykernel numpy pandas matplotlib
```

## Build / Lint / Test Commands

No build/lint/test tooling is configured yet. Prefer adding standard tools instead of bespoke scripts.
If you add any of the following files, also update this section.

### Suggested standard toolchain (Python)

- Formatter: `ruff format`
- Linter: `ruff check`
- Type checker: `pyright` (or `mypy` if you prefer)
- Tests: `pytest`

Install:

```bash
python -m pip install ruff pyright pytest
```

Run (whole repo):

```bash
ruff format .
ruff check .
pyright
pytest
```

Run a single test:

```bash
pytest path/to/test_file.py -k test_name
# or by node id
pytest path/to/test_file.py::TestClass::test_name
```

### Repository contents

Notebooks are present:
- `notebooks/01_analise_comparativa_tecnicas.ipynb`
- `notebooks/02_estrategia_maximo_recall.ipynb`
- `notebooks/03_estrategia_maxima_precisao.ipynb`

Docs currently include:
- `docs/RESUMOPASSOSANALISE.odt`

Data currently includes:
- `data/COMPARADORSEMIDENT.csv`

Notes:
- Treat `data/` as non-source artifacts; avoid editing in-place unless the task is explicitly data curation.
- Avoid committing large derived files; prefer keeping generated outputs outside git or adding patterns to `.gitignore`.

Run Jupyter:

```bash
python -m ipykernel install --user --name phd-research
jupyter lab
```

If you need a non-interactive notebook run (CI-friendly), prefer `papermill`:

```bash
python -m pip install papermill
papermill notebooks/01_analise_comparativa_tecnicas.ipynb /tmp/out.ipynb
```

## Code Style Guidelines

This repo has no explicit style config (`.editorconfig`, `ruff.toml`, `pyproject.toml`, etc.) at the moment.
Until configs are added, follow these conventions.

### Language + modules

- Prefer Python for any new scripts/utilities.
- Put reusable code in a package directory (e.g., `src/phd_research/`) rather than in notebooks.
- Keep notebooks for exploration and final figures; move logic into `.py` modules.

### Formatting

- Target: 88-char line length (Black/Ruff default).
- Use `ruff format` if present; otherwise follow Black-like formatting.
- Avoid manual alignment for tables/dicts that fight autoformatters.

### Imports

- Standard library imports first, then third-party, then local.
- One import per line when reasonable.
- Avoid wildcard imports.
- Example:

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from phd_research.metrics import score
```

### Types

- Use type hints for public functions and any non-trivial logic.
- Prefer standard collection types: `list[str]`, `dict[str, int]`.
- Prefer `Path` over raw strings for filesystem paths.
- Avoid `Any`; if unavoidable, contain it to the smallest scope.

### Naming

- Modules/files: `snake_case.py`.
- Packages: `snake_case`.
- Functions/variables: `snake_case`.
- Classes: `PascalCase`.
- Constants: `UPPER_SNAKE_CASE`.
- Notebook names are currently Portuguese; keep consistency within `notebooks/`.

### Error handling

- Fail fast with informative exceptions for invalid inputs.
- Do not swallow exceptions with bare `except:`.
- Catch only what you can handle; re-raise with context.
- For scripts/CLIs: return non-zero exit codes on failure.

### Logging

- Prefer the `logging` module over `print` for reusable code.
- Notebooks may use `print` for quick inspection, but keep outputs tidy.

### Data & I/O

- Make file paths explicit and configurable; avoid hard-coded absolute paths.
- Large artifacts: keep out of git; add to `.gitignore` if needed.
- For deterministic experiments: set random seeds and record them.

### Notebook hygiene

- Keep notebook cells idempotent and restartable.
- Avoid hidden state: re-run from top should work.
- Add a short "Purpose" markdown cell at the top.

## Repository Rules (Cursor / Copilot)

No Cursor rules were found (`.cursorrules` or `.cursor/rules/`).
No GitHub Copilot instructions were found (`.github/copilot-instructions.md`).

If you add any of these files later, copy the relevant constraints into this document.

## Project Hygiene (Recommended Additions)

Agentic work benefits from explicit tooling. Consider adding:

- `pyproject.toml` with:
  - `ruff` config (format + lint)
  - `pytest` config
  - `pyright` config (or `mypy`)
- `requirements.txt` (pinned) or `uv.lock` / `poetry.lock`
- `README.md` documenting environment + how to reproduce notebook outputs
- Optional: `pre-commit` hooks for formatting/linting

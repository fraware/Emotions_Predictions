# Contributing

## Environment

Use **Python 3.10, 3.11, or 3.12** (see `requires-python` in [`pyproject.toml`](pyproject.toml)). Create a virtual environment and install the project in editable mode with dev tools:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -e ".[dev]"
```

Runtime-only install (no pytest, ruff, pre-commit): `pip install -e .`

## Code style

- **Ruff** is the linter and formatter; configuration lives in [`pyproject.toml`](pyproject.toml).
- Before pushing, match CI:

```bash
ruff check src tests main.py
ruff format --check src tests main.py
```

- Optional: install hooks with `pre-commit install` (see [`.pre-commit-config.yaml`](.pre-commit-config.yaml)).

## Tests

Default CI runs fast tests only (excludes TensorFlow-heavy smoke):

```bash
pytest -m "not slow"
```

Optional full smoke test (small `fit` step):

```bash
pytest -m slow
```

## Pull requests

- Keep changes focused on a single concern when possible.
- Ensure `ruff check`, `ruff format --check`, and `pytest -m "not slow"` pass locally.
- If you change user-visible behavior, update [`README.md`](README.md) in the same change.

## Package build

To verify the distribution artifacts:

```bash
pip install build
python -m build
```

Wheels and sdist appear under `dist/`.

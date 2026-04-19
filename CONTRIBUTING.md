# Contributing to aa-animator

Thank you for your interest in contributing. This document covers everything you need to get started.

## Code of Conduct

This project follows the [Contributor Covenant v2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
By participating you agree to abide by its terms.

---

## Dev setup

```bash
git clone https://github.com/hinanohart/aa-animator
cd aa-animator
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

Requirements: Python 3.10–3.13, `ffmpeg` on PATH.

---

## Running tests

```bash
pytest tests/ -v
```

All 69 tests must pass before opening a PR. Do not modify test assertions to force a pass — fix the production code instead.

---

## Linting and formatting

```bash
ruff check src tests
ruff format src tests
mypy src/aa_animator_v2
```

The CI matrix runs all of these automatically.

---

## Personal information leak check

Before committing, verify no local paths or usernames have leaked:

```bash
rg -i "$(whoami)|/home/$(whoami)|/mnt/c/Users/$(whoami)" src/ tests/ scripts/ docs/ .github/
```

Zero matches required. If you find any, remove them and re-stage.

---

## PR workflow

1. Fork the repo and create a branch: `git checkout -b feat/my-thing`
2. Make your changes with the smallest viable diff.
3. Run `pytest tests/ -v` and all lint checks — all must pass.
4. Run the personal-info leak check above.
5. Open a PR against `main`. Fill in the PR template.
6. A maintainer will review within a few days.

---

## Commit message style

```
type: short summary (imperative, <= 72 chars)

Optional longer body explaining the why, not the what.
Wrap at 72 chars.
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`.

Examples:
```
feat: add --amp-deg CLI flag for parallax amplitude
fix: clamp EMA alpha to [0, 1] to prevent NaN frames
docs: add Limitations section to README
```

---

## Adding dependencies

- No new runtime dependencies without prior discussion in [Discussions → Ideas](https://github.com/hinanohart/aa-animator/discussions/categories/ideas).
- All licenses must be OSI-approved (no AGPL, no non-commercial).
- Update `NOTICE` with the new dependency's copyright header.
- Do not change `pyproject.toml` `[project.dependencies]` without maintainer approval.

---

## Reporting security issues

Please do **not** open a public issue. See [SECURITY.md](SECURITY.md).

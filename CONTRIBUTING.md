# Contributing to SpectraQC

Thanks for your interest in contributing to SpectraQC! This document covers the workflow, local setup, test commands, and coding conventions we use in this repository.

## Contribution workflow

1. **Search existing issues** to avoid duplicate work.
2. **Create a branch** from `main` (or the default branch) for your change.
3. **Make focused commits** with clear messages.
4. **Run tests** relevant to your change.
5. **Open a pull request** with a summary, rationale, and testing notes.

If you’re proposing a change to behavior, update or add documentation under `docs/` and include any necessary schema updates.

## Local setup

### Prerequisites

- Python 3.11+
- `ffmpeg` (optional, only required for loudness measurement and some decode paths)

### Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

## Test commands

Run the full unit test suite:

```bash
pytest
```

## Coding conventions

- **Python style:** follow PEP 8 and keep functions small and focused.
- **Type hints:** add type hints for new public functions and non-trivial data structures.
- **Determinism:** avoid non-deterministic behavior (e.g., random seeds without explicit control).
- **Documentation:** update README/docs for user-facing changes.
- **Error messages:** keep CLI errors actionable and consistent.

## Reporting security issues

If you discover a security issue, please avoid opening a public issue. Instead, email the maintainers with details and reproduction steps.

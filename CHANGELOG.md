# Changelog

All notable changes to this project are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `pyproject.toml` with project metadata, ruff and pytest configuration
- `.gitignore` now whitelists `CHANGELOG.md` and `CITATION.md`
- `.gitattributes` for cross-platform line ending consistency
- `src/data/efus_loader.py` — EFUS 2017 data loader prototype
- `src/research/desnz_lsoa_validation.py` — MSOA-level calibration vs DESNZ aggregates
- `src/research/targeting_comparison.py` — T* vs physics-only retrofit targeting analysis
- `src/config/paths.py`, `model.py`, `logging.py` — frozen `PathConfig` / `ModelConfig` dataclasses and `setup_logging`, extracted from `settings.py`
- `src/core/exceptions.py` — `DataValidationError`, `ConvergenceError`, `ProvenanceError` (each subclasses the built-in it replaces, so existing handlers still catch them)
- `[tool.pyright]` (basic, `src` excluding `src/research`); `pyright`, `pandas-stubs` and `ruff` added to `requirements-dev.txt`
- `tests/unit/test_core_exceptions.py`

### Changed
- Repo canonical location moved to OneDrive with junction at `~/national-energy-bayesian-network/`
- CLAUDE.md: venv instructions updated (create outside OneDrive at `~/.venvs/paper5/`)
- `src/config/settings.py` is now a thin import surface over `PathConfig` / `ModelConfig`; every module-level name is unchanged (verified identical in prod, `TEST_MODE` and `USE_FAKE_CITY`)
- Type annotations on every function signature outside `src/research`; pyright basic goes from 129 errors to 0 (type-only fixes — casts, `| None` on `None` defaults, unbound `RAW_DIR` in `epw_parser`'s `__main__`)
- Domain exceptions raised in `src/inference`, `src/utils/provenance.py` and `src/data`; `InlaGateFailedError` now derives from `ConvergenceError`
- `ruff check --fix` (safe fixes) and `ruff format` applied to `src` excluding `src/research`; 21 lint findings remain (E402 after `sys.path` setup, unused locals, whitespace)
- `LatinHypercube` deliberately keeps `seed=` — `rng=` produces a different stream and would change the LHS designs

## [1.0.0] — 2026-07-26

Initial release, accompanying the manuscript submitted to *Energy and Buildings*.

### Added
- BYM2 + restricted spatial regression model for 6,853 MSOAs
- Gaussian Process surrogate trained on EnergyPlus simulations (R² = 0.9937)
- Dual inference engines: R-INLA and PyMC NUTS sampler
- CLI entry point (`src/cli/run_inference.py`) with preflight input validation
- Pandera data contracts for population DataFrame
- SHA-256 provenance hashing for input data files
- Unit, integration, end-to-end, and property-based test suite
- CI pipeline with GitHub Actions
- Zenodo DOI archive

[Unreleased]: https://github.com/JulesBuckland/national-energy-bayesian-network/compare/v1.0...HEAD
[1.0.0]: https://github.com/JulesBuckland/national-energy-bayesian-network/releases/tag/v1.0

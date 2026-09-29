# Changelog

All notable changes to this project are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `pyproject.toml` with project metadata, ruff and pytest configuration
- `.gitignore` now whitelists `CHANGELOG.md` and `CITATION.md`

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

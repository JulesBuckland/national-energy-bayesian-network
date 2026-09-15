# Separating dwelling heat requirement from household energy rationing

[![Tests](../../actions/workflows/pytest.yml/badge.svg)](../../actions/workflows/pytest.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21629036.svg)](https://doi.org/10.5281/zenodo.21629036)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Licence: CC BY-NC 4.0](https://img.shields.io/badge/licence-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

A national-scale Bayesian model of England's housing stock that separates two
things routinely conflated in observed gas consumption:

- **the thermal energy a dwelling physically requires** — a function of its
  fabric, form and climate exposure, estimated by a Gaussian-process surrogate
  trained on EnergyPlus simulations; and
- **the energy its occupants actually use** — which, for households under
  financial pressure, can fall well below that requirement.

The gap between the two is a measure of *rationing* (self-disconnection,
under-heating) rather than of efficiency. Treating a low gas bill as evidence of
an efficient home is the inference error the model exists to avoid.

Estimation is intended to cover the English MSOA geography supplied through the
boundary input. The reproducible synthetic generator creates **100 households
per available MSOA**; when no boundary file is present its fallback is **6,800
dummy MSOAs (680,000 rows)**. Production MSOA and household counts therefore
come from the supplied inputs, not from fixed counts in this README. The spatial
term is a BYM2 field with penalised-complexity priors, fitted under restricted
spatial regression (RSR) so the spatial random effect is projected onto the
orthogonal complement of the income-deprivation covariate — without which the
spatial field absorbs the deprivation signal the model is trying to measure.

> **Status.** Supports a manuscript submitted to *Energy and Buildings* (July
> 2026). Results are not peer-reviewed yet.

---

## Architecture

```mermaid
flowchart TD
    classDef dataset fill:#333333,color:#ffffff,stroke:#FFCC33,stroke-width:2px
    classDef func fill:#FFCC33,color:#333333,stroke:#333333,stroke-width:2px
    classDef output fill:#660099,color:#ffffff,stroke:#333333,stroke-width:2px

    %% ── Datasets (external to both pipelines) ──
    Archetypes["32 building archetypes<br/>8 age bands × 4 built forms"]:::dataset
    Census["Census 2021<br/>housing + tenure tables"]:::dataset
    NEED["NEED 2024<br/>50k gas-consumption panel"]:::dataset
    EFUS["EFUS 2017<br/>2,632 monitored dwellings"]:::dataset
    Boundaries["MSOA boundary<br/>geometries (Dec 2021)"]:::dataset

    subgraph Surrogate ["Surrogate training (offline, one-time)"]
        direction TB
        LHS["Latin hypercube sampling<br/>over the archetype design space"]:::func
        EP["EnergyPlus simulation"]:::func
        GPTrain["Gaussian-process surrogate<br/>(6-D Matérn, R² ≥ 0.99)"]:::func
        LHS --> EP --> GPTrain
    end

    subgraph Pipeline ["National inference pipeline"]
        direction TB
        IPF["Iterative proportional fitting<br/>100 synthetic households per available MSOA"]:::func
        Apply["Apply GP surrogate<br/>per-household thermal demand"]:::func
        Fit["Bayesian spatial model<br/>BYM2 + RSR + PC priors<br/>R-INLA (wrapper default) · PyMC NUTS (alternative)"]:::func
        Output["Per-MSOA posterior estimates<br/>T* + credible intervals"]:::output
        Validation["External validation<br/>(separate from main fit)"]:::func
        IPF --> Apply --> Fit --> Output
    end

    Archetypes --> LHS
    Census --> IPF
    NEED --> IPF
    EFUS -.->|optional external validation<br/>not read by main fit| Validation
    Boundaries --> Fit
    GPTrain -.->|trained surrogate| Apply
```

The Windows wrapper selects R-INLA by default; the direct Python inference CLI
runs the PyMC NUTS path. Runtime is hardware-, input-, and mode-dependent, so
the figures above are not reproducibility guarantees. The two implementations
are parallel fits rather than numerically identical cross-checks: R-INLA uses
PC priors, whereas PyMC uses Beta/HalfNormal priors. Their comparison should be
reported as an implementation/prior sensitivity check, not as validation of the
same posterior.

---

## Installation

Requires **Python 3.12**. R and INLA are optional — see
[Optional: R-INLA](#optional-r-inla).

```bash
git clone https://github.com/JulesBuckland/national-energy-bayesian-network.git
cd national-energy-bayesian-network
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

`requirements.txt` holds runtime dependencies with version floors;
`requirements-dev.txt` adds the test tooling; `requirements-lock.txt` pins the
full transitive tree used to produce the published results, for exact
reproduction.

### Verify the installation

```bash
pytest -ra
```

Most unit tests need no external data. Tests that require licensed fixtures are
marked and skipped only when `tests/fixtures/raw/` is absent; if that directory
exists but contains invalid or incomplete files, those tests can fail rather
than skip. R-INLA component tests require a separate R installation with INLA
and are not part of a normal Python-only run. A green `pytest -ra` run therefore
covers the available unit/fixture paths, but is not evidence that the restricted
production inputs or a full national fit have been exercised.

---

## Quick start on synthetic data

The pipeline's real inputs are large and partly licence-restricted, so the
repository ships a generator that produces synthetic inputs of the right shape
from a fixed seed:

```bash
python -m src.data.generate_synthetic_data
```

That writes 100 rows per available MSOA to `data/processed/`. Without a boundary
file it uses 6,800 dummy MSOA codes and therefore writes 680,000 rows. This is a
smoke-test input, not a substitute for the NEED/Census-based population
synthesis used for the substantive analysis.

To run inference on this generated input you additionally need the MSOA boundary
geometries and a trained surrogate (`data/processed/gp_emulator.pkl`). The
analytical CSV fallback is intended for population-synthesis output containing
`property_type` and `property_age`; the lightweight generator does not create
those columns, so it cannot by itself exercise that fallback path.

```bash
python -m src.cli.run_inference
```

For a fast approximate run on real data, set `PILOT_MODE=1`: it uses a
deliberately small draw count and writes to distinctly-suffixed files, stamped
with run metadata, so pilot output cannot be mistaken for a final result.

---

## Data

No restricted production input data is committed. Locally generated physics
artifacts may exist in an ignored checkout, while the tracked golden files are
test outputs rather than licensed raw inputs. Sizes below describe the intended
national inputs, not necessarily what is present in a fresh clone.

| Input | Licence | How to obtain |
|---|---|---|
| MSOA boundaries (Dec 2021) | Open Government Licence v3 | [ONS Open Geography Portal](https://geoportal.statistics.gov.uk/) |
| Census 2021 housing and tenure (TS044, TS054) | Open Government Licence v3 | [ONS / Nomis](https://www.nomisweb.co.uk/) |
| NEED gas-consumption panel (50,000-property sample) | DESNZ end-user licence | [gov.uk NEED collection](https://www.gov.uk/government/collections/national-energy-efficiency-data-need-framework) |
| EFUS 2017 (SN 9434, ~2,632 monitored dwellings) | UK Data Service end-user licence | Licensed external validation input; not used in main-fit training and not reported as if it were. [UK Data Service](https://doi.org/10.5255/UKDA-SN-9434-1) |
| EnergyPlus LHS simulation results | Generated locally (~6.8 GB of runs) | `src/inference/lhs_sampler.py` then `src/physics/energyplus_batch.py` |

Place raw inputs under `data/raw/` following the paths in
`src/config/settings.py`, which is the central source for the main data
locations. It is not yet the source of every physics setting: in particular,
`src/physics/energyplus_batch.py` and `src/physics/energyplus_client.py` still
contain local EnergyPlus executable and/or weather/template/simulation paths
that must be configured for the machine running the simulations.

The test fixtures under `tests/fixtures/raw/` are derived from the NEED sample
and are therefore **not redistributable**. Regenerate them with
`python tests/generate_fixtures.py` once `data/raw/` is populated. Fixture tests
skip only when that directory is absent; they do not certify that an existing
fixture directory contains valid or complete inputs.

---

## Reproducing the published results

```bash
python -m src.inference.lhs_sampler          # LHS design over the archetype space
python -m src.physics.energyplus_batch       # EnergyPlus runs (long; needs EnergyPlus)
python -m src.inference.gp_emulator          # train and validate the GP surrogate
python -m src.data.population                # IPF synthetic population
python -m src.inference.inla.run_inla        # primary national fit (R-INLA)
```

`run_national_pipeline.ps1` wraps the population → surrogate → inference stages
on Windows and defaults to R-INLA; pass `-nuts` to select PyMC (or `-inla`
explicitly). The direct `src.cli.run_inference` entry point currently invokes
PyMC. Validation and figure scripts live in `src/research/`:

| Script | Purpose |
|---|---|
| `regenerate_main_figures.py` | Manuscript figures, drawn from data at run time |
| `regenerate_fig1_architecture.py` | Architecture figure |
| `make_graphical_abstract.py` | Graphical abstract |
| `ukhls_convergent_validation.py` | External convergent check against UKHLS household gas expenditure (not part of the main fit) |
| `spatial_holdout_test.py` | LAD-grouped spatial holdout predictive check (no LAD crosses train/test) |
| `prior_sensitivity.py` | Prior-sensitivity sweep |
| `nuts_validation.py` | PyMC NUTS cross-check of the INLA posterior |
| `scaling_benchmark.py` | Runtime scaling benchmark |

Figures are always regenerated from data rather than checked in, so a stale
hardcoded constant cannot silently outlive the number it came from. Validation
reports must likewise be generated from the current output files: the spatial
holdout is a grouped LAD split, while UKHLS and EFUS are external checks and
must not be described as observations used in the main Bayesian fit. EFUS is
currently retained as a licensed input for a future, explicitly documented
meter-level validation module; its raw meter files are not silently folded into
model training.

---

## Optional: R-INLA

The primary engine calls R through a subprocess. Install R, then:

```r
install.packages("INLA",
  repos = c(getOption("repos"), INLA = "https://inla.r-inla-download.org/R/stable"),
  dep = TRUE)
```

INLA is not on CRAN, so a default R install will not have it. Without it,
the R-INLA runner and its R integration checks are unavailable; the PyMC NUTS
path in `src/inference/model_unified.py` remains the usable Python engine. The
R component checks under `tests/` are `.R` scripts and require an R-aware test
runner or explicit `Rscript` invocation; they are not evidence supplied by a
normal `pytest` run.

---

## Repository layout

```
src/
  cli/          entry point with input preflight checks
  config/       settings.py — every path, constant and mode flag
  data/         IPF population synthesis, archetypes, synthetic-input generator
  physics/      EnergyPlus batch driver and client
  inference/    GP surrogate, LHS sampler, ICAR scaling, NUTS model
    inla/       primary R-INLA engine (BYM2 + PC priors + RSR)
  research/     figure regeneration and validation scripts
  utils/        data contracts, EPW parsing, result cleaning
tests/
  unit/         no external data required
  integration/  real statistical cores, no mocked computation
  fixtures/     golden baselines (inputs are not redistributable)
```

Three environment flags change behaviour, all read in
`src/config/settings.py`: `TEST_MODE` (tiny fixture data), `PILOT_MODE` (real
data, small draw count) and `USE_FAKE_CITY` (synthetic single-city inputs).

---

## Citation

If you use this code, please cite the archived release:

```bibtex
@software{buckland_thermal_rationing,
  author  = {Buckland, Jules},
  title   = {Separating dwelling heat requirement from household energy
             rationing at national scale},
  year    = {2026},
  doi     = {10.5281/zenodo.21629036},
  url     = {https://github.com/JulesBuckland/national-energy-bayesian-network}
}
```

The DOI above is the concept DOI and always resolves to the latest version;
`10.5281/zenodo.21629037` pins the v1.0 submission release.

---

## Licence

Released under the [Creative Commons Attribution-NonCommercial 4.0
International licence](https://creativecommons.org/licenses/by-nc/4.0/).
This licence applies to the repository's code and documentation; it does not
relicense NEED, EFUS, Census, boundary, EnergyPlus, or other third-party inputs,
and it does not grant automatic redistribution rights for derived outputs built
from restricted sources. Check each source's terms before sharing data,
fixtures, trained models, or result files.

Note: the archived Zenodo release (DOI: 10.5281/zenodo.21629036) predates this
change and remains under CC BY 4.0, which cannot be revised retroactively;
this licence governs the current and future state of this repository.

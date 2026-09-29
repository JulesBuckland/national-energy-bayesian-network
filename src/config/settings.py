"""Backwards-compatible import surface for project configuration.

The values live in ``paths.PathConfig`` and ``model.ModelConfig``; this module
selects them from environment variables and re-exports the original
module-level names so existing ``from src.config.settings import X`` keeps
working.
"""
import os

from src.config.logging import setup_logging
from src.config.model import ModelConfig
from src.config.paths import BASE_DIR, PathConfig

SCRIPT_DIR = BASE_DIR / "src" / "config"

TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"

USE_FAKE_CITY = os.environ.get("USE_FAKE_CITY", "0") == "1"

# A fast, low-draw approximation run — NOT the same as TEST_MODE (which
# points at tiny fixture data). PILOT_MODE runs on real/production data with
# a deliberately small draw count, so its output must never be mistaken for
# a final result: model_unified.py routes pilot output to distinctly-suffixed
# files and stamps every saved trace with its run metadata (see
# src/inference/model_unified.py's _run_metadata()).
PILOT_MODE = os.environ.get("PILOT_MODE", "0") == "1"

_paths = PathConfig(test_mode=TEST_MODE, use_fake_city=USE_FAKE_CITY)
_model = ModelConfig.for_mode(test_mode=TEST_MODE)

# --- DIRECTORY STRUCTURE ---
RAW_DIR = _paths.raw_dir
PROCESSED_DIR = _paths.processed_dir
LOGS_DIR = _paths.logs_dir
REGIONAL_TRACES_DIR = _paths.regional_traces_dir
LAD_TRACES_DIR = _paths.lad_traces_dir
for _d in (PROCESSED_DIR, LOGS_DIR, REGIONAL_TRACES_DIR, LAD_TRACES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- DATA PATHS ---
NEED_MICRODATA_PATH = _paths.need_microdata_path
CENSUS_CONSTRAINTS_PATH = _paths.census_constraints_path
IMD_PATH = _paths.imd_path
PHYSICS_ARCHETYPES_PATH = _paths.physics_archetypes_path
CENSUS_HOUSING_NATIONAL = _paths.census_housing_national
CENSUS_TENURE_NATIONAL = _paths.census_tenure_national
MSOA_CONFOUNDERS_NATIONAL = _paths.msoa_confounders_national
LOOKUP_PATH = _paths.lookup_path
LAD_LOOKUP_PATH = _paths.lad_lookup_path
MSOA_REGION_LOOKUP = _paths.msoa_region_lookup
BOUNDARIES_PATH = _paths.boundaries_path

# --- FILE OUTPUT NAMES ---
SYNTHETIC_POP_FILE = "national_synthetic_population_eti.parquet"
HEATING_DEFICIT_FILE = "msoa_heating_deficit_results.csv"
BAYESIAN_TRACE_FILE = "eti_bayesian_icar_trace.nc"
ETI_RESULTS_FILE = "empirical_thermal_index_results.csv"

# --- MODEL CONSTANTS ---
N_MSOAS = 6840  # legacy fallback only; production coverage is input-derived
RANDOM_SEED = _model.RANDOM_SEED
N_HH_SAMPLES_PER_MSOA = _model.N_HH_SAMPLES_PER_MSOA
COLD_SNAP_DURATION_HOURS = _model.COLD_SNAP_DURATION_HOURS
ELECTRIC_BASELOAD_KWH = _model.ELECTRIC_BASELOAD_KWH
GAS_PRESENCE_THRESHOLD_KWH = _model.GAS_PRESENCE_THRESHOLD_KWH
GP_ACCEPTANCE_R2 = _model.GP_ACCEPTANCE_R2
BASE_TEMP_HDD = _model.BASE_TEMP_HDD

# --- STATISTICAL SETTINGS ---
MCMC_SAMPLES = _model.MCMC_SAMPLES
MCMC_TUNE = _model.MCMC_TUNE
MCMC_CORES = _model.MCMC_CORES
MCMC_CHAINS = _model.MCMC_CHAINS
MCMC_MAX_RHAT = _model.MCMC_MAX_RHAT
MCMC_MAX_DIVERGENCES = _model.MCMC_MAX_DIVERGENCES

# --- GEOGRAPHIC SCOPE ---
# National focus: England. Re-exported as list/dict, the types callers used.
REGIONS = list(_model.REGIONS)
REGIONAL_CENTERS = dict(_model.REGIONAL_CENTERS)

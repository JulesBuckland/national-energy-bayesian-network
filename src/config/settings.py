from pathlib import Path
import numpy as np

# --- DIRECTORY STRUCTURE ---
import os
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent

TEST_MODE = os.environ.get("TEST_MODE", "0") == "1"

USE_FAKE_CITY = os.environ.get("USE_FAKE_CITY", "0") == "1"

# A fast, low-draw approximation run — NOT the same as TEST_MODE (which
# points at tiny fixture data). PILOT_MODE runs on real/production data with
# a deliberately small draw count, so its output must never be mistaken for
# a final result: model_unified.py routes pilot output to distinctly-suffixed
# files and stamps every saved trace with its run metadata (see
# src/inference/model_unified.py's _run_metadata()).
PILOT_MODE = os.environ.get("PILOT_MODE", "0") == "1"

if TEST_MODE:
    RAW_DIR = BASE_DIR / "tests" / "fixtures" / "raw"
    PROCESSED_DIR = BASE_DIR / "tests" / "fixtures" / "processed"
elif USE_FAKE_CITY:
    RAW_DIR = BASE_DIR / "data" / "raw" / "fake"
    PROCESSED_DIR = BASE_DIR / "data" / "processed" / "fake"
else:
    RAW_DIR = BASE_DIR / "data" / "raw"
    PROCESSED_DIR = BASE_DIR / "data" / "processed"

LOGS_DIR = BASE_DIR / "logs"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# --- LOGGING CONFIGURATION ---
import logging
import datetime

def setup_logging(name):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"run_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(name)

# --- MODEL CONSTANTS ---
RANDOM_SEED = 42
# For national scale, we limit synthetic samples per MSOA for compute efficiency
N_HH_SAMPLES_PER_MSOA = 100 
N_MSOAS = 6840

# --- DATA PATHS (Projected) ---
NEED_MICRODATA_PATH = RAW_DIR / "energy" / "need_2024_official_50k.csv"
CENSUS_CONSTRAINTS_PATH = RAW_DIR / "census" / "census_2021_msoa_housing.csv"
IMD_PATH = PROCESSED_DIR / "msoa_imd_official.csv"
PHYSICS_ARCHETYPES_PATH = RAW_DIR / "physics" / "physics_archetypes_baseline.csv"

# --- GEOGRAPHIC SCOPE ---
# National focus: England
REGIONS = ['North West', 'North East', 'Yorkshire and The Humber', 'East Midlands', 
           'West Midlands', 'East of England', 'London', 'South East', 'South West']

# Representative cities for regional weather pulling
REGIONAL_CENTERS = {
    'London': (51.5074, -0.1278),
    'Manchester': (53.4808, -2.2426),
    'Birmingham': (52.4862, -1.8904),
    'Leeds': (53.8008, -1.5491),
    'Newcastle': (54.9783, -1.6178),
    'Bristol': (51.4545, -2.5879),
    'Norwich': (52.6309, 1.2974),
    'Southampton': (50.9097, -1.4044),
    'Nottingham': (52.9548, -1.1581)
}



COLD_SNAP_DURATION_HOURS = 336 # 14 days
ELECTRIC_BASELOAD_KWH = 2000
GAS_PRESENCE_THRESHOLD_KWH = 500

# GP emulator held-out R^2 acceptance threshold. Lower under TEST_MODE because
# the tiny synthetic fixture (a few hundred rows) cannot reliably reach the
# production bar that a full multi-thousand-point EnergyPlus LHS sample can -
# this is a deliberately-chosen, still-enforced bar, not a bypass of it.
GP_ACCEPTANCE_R2 = 0.99

# --- PHYSICAL ASSUMPTIONS ---
# Base temperature for heating degree day calculations
BASE_TEMP_HDD = 15.5 

# --- STATISTICAL SETTINGS ---
# MCMC settings for Bayesian ICAR model
MCMC_SAMPLES = 2000 # Increased for final publication run
MCMC_TUNE = 3000 # Extra warmup for mass-matrix adaptation in the rho funnel
MCMC_CORES = 4 
MCMC_CHAINS = 4

if TEST_MODE:
    MCMC_SAMPLES = 10
    MCMC_TUNE = 10
    MCMC_CORES = 1
    MCMC_CHAINS = 1
    GP_ACCEPTANCE_R2 = 0.95

# Convergence gate applied after every NUTS run, before results are treated as
# final. A small nonzero divergence tolerance (rather than a hard zero) avoids
# an overly brittle gate on an otherwise well-fit model; r_hat is only
# meaningful with >=2 chains, so it is checked in model_unified.py only when
# MCMC_CHAINS >= 2 (true for production and PILOT_MODE, not TEST_MODE's
# single-chain fixture runs).
MCMC_MAX_RHAT = 1.01
MCMC_MAX_DIVERGENCES = 10

# --- FILE PATHS (National) ---
CENSUS_HOUSING_NATIONAL = RAW_DIR / "census" / "ts044_extracted" / "census2021-ts044-msoa.csv"
if not CENSUS_HOUSING_NATIONAL.exists():
    CENSUS_HOUSING_NATIONAL = RAW_DIR / "census" / "census2021-ts044-msoa.csv"
CENSUS_TENURE_NATIONAL = RAW_DIR / "census" / "TS054-2021-4-filtered-2026-02-27T03_51_51Z.csv"
MSOA_CONFOUNDERS_NATIONAL = PROCESSED_DIR / "msoa_confounders_national.csv"
LOOKUP_PATH = RAW_DIR / "spatial" / "lookup.csv"
LAD_LOOKUP_PATH = PROCESSED_DIR / "msoa_lad_lookup.csv"
MSOA_REGION_LOOKUP = PROCESSED_DIR / "msoa_region_lookup.csv"
BOUNDARIES_PATH = RAW_DIR / "spatial" / "msoa dec 2021 boundaries.gpkg"

if USE_FAKE_CITY:
    CENSUS_HOUSING_NATIONAL = RAW_DIR / "fake_census_housing.csv"
    CENSUS_TENURE_NATIONAL = RAW_DIR / "fake_census_tenure.csv"
    MSOA_CONFOUNDERS_NATIONAL = PROCESSED_DIR / "fake_msoa_confounders.csv"
    LOOKUP_PATH = RAW_DIR / "fake_lookup.csv"
    MSOA_REGION_LOOKUP = PROCESSED_DIR / "fake_msoa_region_lookup.csv"
    BOUNDARIES_PATH = RAW_DIR / "fake_msoa_boundaries.gpkg"
    NEED_MICRODATA_PATH = RAW_DIR / "fake_need_seed.csv"

REGIONAL_TRACES_DIR = PROCESSED_DIR / "regional_traces"
LAD_TRACES_DIR = PROCESSED_DIR / "lad_traces"
REGIONAL_TRACES_DIR.mkdir(parents=True, exist_ok=True)
LAD_TRACES_DIR.mkdir(parents=True, exist_ok=True)

# --- FILE OUTPUT NAMES ---
SYNTHETIC_POP_FILE = "national_synthetic_population_eti.parquet"
HEATING_DEFICIT_FILE = "msoa_heating_deficit_results.csv"
BAYESIAN_TRACE_FILE = "eti_bayesian_icar_trace.nc"
ETI_RESULTS_FILE = "empirical_thermal_index_results.csv"

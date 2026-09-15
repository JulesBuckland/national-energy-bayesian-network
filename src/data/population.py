import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import os
from src.utils.provenance import assert_unique_keys, assert_keys_subset
from src.config.settings import (
    NEED_MICRODATA_PATH, RAW_DIR,
    PROCESSED_DIR, SYNTHETIC_POP_FILE, RANDOM_SEED,
    N_HH_SAMPLES_PER_MSOA, MSOA_CONFOUNDERS_NATIONAL, setup_logging,
    REGIONS, REGIONAL_CENTERS,
    ELECTRIC_BASELOAD_KWH, GAS_PRESENCE_THRESHOLD_KWH
)

logger = setup_logging("NationalSynthesisV8")

# ---------------------------------------------------------------------------
# Physical parameters per age band — nominal values with per-household noise
# (same specs as src/inference/lhs_sampler.py)
# ---------------------------------------------------------------------------
CONSTRUCTION_SPECS = {
    "Pre-1900":  {"wall_u": 2.1,  "ach": 1.5},
    "1900-1929": {"wall_u": 2.1,  "ach": 1.5},
    "1930-1949": {"wall_u": 1.7,  "ach": 1.2},
    "1950-1966": {"wall_u": 1.5,  "ach": 1.0},
    "1967-1982": {"wall_u": 1.0,  "ach": 0.8},
    "1983-1995": {"wall_u": 0.6,  "ach": 0.6},
    "1996-2006": {"wall_u": 0.45, "ach": 0.5},
    "2007+":     {"wall_u": 0.3,  "ach": 0.5},
}
FORM_CODE = {"Flat": 0, "Bungalow": 1, "Maisonette": 2, "House": 3}
WWR_NOMINAL = {"Flat": 0.20, "Bungalow": 0.15, "Maisonette": 0.15, "House": 0.15}
# Mean floor area per archetype used when FLOOR_AREA_BAND is missing
ARCHETYPE_AREA_MEAN = {
    "Pre-1900":  {"House": 106.3, "Flat": 59.6,  "Bungalow": 101.5, "Maisonette": 95.8},
    "1900-1929": {"House": 91.6,  "Flat": 52.8,  "Bungalow": 97.2,  "Maisonette": 75.6},
    "1930-1949": {"House": 89.3,  "Flat": 56.6,  "Bungalow": 71.8,  "Maisonette": 64.2},
    "1950-1966": {"House": 86.5,  "Flat": 55.1,  "Bungalow": 70.9,  "Maisonette": 65.8},
    "1967-1982": {"House": 90.5,  "Flat": 51.3,  "Bungalow": 73.5,  "Maisonette": 64.2},
    "1983-1995": {"House": 90.8,  "Flat": 49.4,  "Bungalow": 66.0,  "Maisonette": 61.1},
    "1996-2006": {"House": 97.8,  "Flat": 60.5,  "Bungalow": 70.8,  "Maisonette": 78.8},
    "2007+":     {"House": 102.6, "Flat": 58.5,  "Bungalow": 78.4,  "Maisonette": 81.9},
}
# Within-archetype coefficient of variation for area (used to add per-HH noise)
_AREA_CV = 0.15
_PHYS_CV = 0.10   # CV for wall_u, ach, wwr noise
PT_TYPES = ['House', 'Flat', 'Bungalow']
TENURE_TYPES = ['Owned', 'Social', 'Private']
# Maps official ONS region names to the representative city used for HDD lookup
REGION_TO_HDD_KEY = {
    'North West': 'Manchester',
    'North East': 'Newcastle',
    'Yorkshire and The Humber': 'Leeds',
    'East Midlands': 'Nottingham',
    'West Midlands': 'Birmingham',
    'East of England': 'Norwich',
    'London': 'London',
    'South East': 'Southampton',
    'South West': 'Bristol'
}


def clean_seed_age_band(age_band):
    mapping = {
        1: "Pre-1900", 2: "1900-1929", 3: "1930-1949", 4: "1950-1966",
        5: "1967-1982", 6: "1983-1995", 7: "1996-2006", 8: "2007+"
    }
    return mapping.get(age_band, "1950-1966")


def clean_seed_type(prop_type):
    mapping = {
        'Detached': 'House',
        'Semi-detached': 'House',
        'Mid terrace': 'House',
        'End terrace': 'House',
        'Bungalow': 'Bungalow',
        'Flat': 'Flat'
    }
    return mapping.get(prop_type, 'House')


def load_and_clean_seed_data() -> pd.DataFrame:
    """Load NEED microdata, classify heating fuel, drop physically-impossible
    readings, and expand across tenure categories.

    Raises:
        FileNotFoundError: if the seed microdata file is missing.
    """
    if not NEED_MICRODATA_PATH.exists():
        raise FileNotFoundError(f"Seed data not found at {NEED_MICRODATA_PATH}")

    seed_data = pd.read_csv(NEED_MICRODATA_PATH)
    seed_data = seed_data.rename(columns={
        'PROP_TYPE': 'property_type_raw',
        'PROP_AGE_BAND': 'property_age_band',
        'Gcons2022': 'empirical_gas_kwh',
        'Econs2022': 'empirical_elec_kwh'
    })

    seed_data = seed_data.dropna(subset=['empirical_gas_kwh', 'empirical_elec_kwh'])
    assert not seed_data['empirical_gas_kwh'].isna().any(), "FATAL: Missing empirical gas data!"
    assert not seed_data['empirical_elec_kwh'].isna().any(), "FATAL: Missing empirical elec data!"

    # Total Thermal Load Modeling: Handle electric heating in flats without gas
    # We estimate weather-dependent electricity by subtracting a baseload.
    seed_data = seed_data.assign(
        elec_heating_estimated=np.maximum(0, seed_data['empirical_elec_kwh'] - ELECTRIC_BASELOAD_KWH)
    )

    # If gas is virtually non-existent, they are primarily using electric heating.
    # Otherwise, gas is their primary thermal source.
    seed_data = seed_data.assign(
        empirical_thermal_kwh=np.where(
            seed_data['empirical_gas_kwh'] < GAS_PRESENCE_THRESHOLD_KWH,
            seed_data['empirical_gas_kwh'] + seed_data['elec_heating_estimated'],
            seed_data['empirical_gas_kwh']
        )
    )

    seed_data = seed_data[seed_data['empirical_thermal_kwh'] > 0]
    seed_data = seed_data.assign(
        property_type=seed_data['property_type_raw'].apply(clean_seed_type),
        property_age=seed_data['property_age_band'].apply(clean_seed_age_band)
    )

    # Filter out physically impossible meter readings
    # Calculate an approximate T* using archetype mean area to drop insane NEED outliers
    seed_data = seed_data.assign(
        approx_area=seed_data.apply(
            lambda row: ARCHETYPE_AREA_MEAN.get(row['property_age'], {}).get(row['property_type'], 80.0), axis=1
        )
    )
    seed_data = seed_data.assign(
        t_star_approx=seed_data['empirical_thermal_kwh'] / seed_data['approx_area']
    )

    # Typical UK T* is 50-300. Anything > 400 is certainly a commercial property/data error.
    # Anything < 15 is an empty home or meter fault.
    valid_mask = (seed_data['t_star_approx'] >= 15.0) & (seed_data['t_star_approx'] <= 400.0)
    dropped = (~valid_mask).sum()
    logger.info(f"Dropping {dropped} households with physically implausible T* (probable meter errors)")
    seed_data = seed_data[valid_mask]

    # -------------------------------------------------------------
    # DATA LINEAGE TRACKING
    # -------------------------------------------------------------
    from src.utils.tracker import log_distribution
    log_distribution(seed_data, 'empirical_thermal_kwh', '01_pop_synthesis_seed_loaded', logger)

    seed_data_expanded = [
        seed_data.assign(tenure=tenure)
        for tenure in ['Owned', 'Social', 'Private']
    ]
    return pd.concat(seed_data_expanded).reset_index(drop=True)


def build_seed_lookup_arrays(seed_q: pd.DataFrame) -> dict:
    """Pre-extract numpy arrays and boolean masks from the seed once, avoiding
    a full DataFrame.copy() and repeated groupby calls inside the MSOA loop."""
    seed_pt = seed_q['property_type'].values
    seed_tenure = seed_q['tenure'].values
    return {
        "seed_pt": seed_pt,
        "seed_tenure": seed_tenure,
        "seed_imd": seed_q['IMD_BAND_ENG'].values,
        "seed_age": seed_q['property_age'].values,
        "seed_thermal": seed_q['empirical_thermal_kwh'].values,
        "seed_gas": seed_q['empirical_gas_kwh'].values,
        "seed_area_band": seed_q['FLOOR_AREA_BAND'].values,
        "pt_masks": {t: seed_pt == t for t in PT_TYPES},
        "tenure_masks": {t: seed_tenure == t for t in TENURE_TYPES},
    }


def load_census_marginals() -> tuple:
    """Load and index the TS044 (property type), TS054 (tenure), and income
    deprivation confounder files used as IPF marginals.

    Returns:
        (type_marginals, tenure_marginals, confounders_idx) — each indexed by
        MSOA/geography code for O(1) per-MSOA lookup in the synthesis loop.

    Raises:
        FileNotFoundError: if any of the three source files is missing.
    """
    from src.config.settings import CENSUS_HOUSING_NATIONAL, CENSUS_TENURE_NATIONAL

    ts044_path = CENSUS_HOUSING_NATIONAL
    if not ts044_path.exists():
        raise FileNotFoundError(f"TS044 marginals not found at {ts044_path}")
    # Use the complete, deduplicated bulk extract.  The older extracted CSV
    # contains three duplicated geography codes caused by six named MSOAs being
    # represented twice; accepting that file would make ``.loc`` return a
    # DataFrame and silently choose the first row below.
    type_marginals = pd.read_csv(ts044_path)
    assert_unique_keys(type_marginals, ["geography code"], label="Census TS044")
    type_marginals = type_marginals.set_index('geography code')


    ts054_path = CENSUS_TENURE_NATIONAL
    if not ts054_path.exists():
        raise FileNotFoundError(f"TS054 marginals not found at {ts054_path}")
    tenure_raw = pd.read_csv(ts054_path)
    assert_keys_subset(
        type_marginals.reset_index().rename(columns={"geography code": "msoa_cd"}),
        tenure_raw.rename(columns={"Middle layer Super Output Areas Code": "msoa_cd"}),
        "msoa_cd",
        required_label="Census TS044",
        available_label="Census TS054",
    )
    def map_tenure(code):
        if code in [0, 1]: return 'Owned'
        if code in [3, 4]: return 'Social'
        return 'Private'
    tenure_raw = tenure_raw.assign(
        tenure_cat=tenure_raw['Tenure of household (9 categories) Code'].apply(map_tenure)
    )
    tenure_marginals = (
        tenure_raw
        .groupby(['Middle layer Super Output Areas Code', 'tenure_cat'])['Observation']
        .sum()
        .unstack(fill_value=0)
    )   # already indexed by MSOA code

    if not MSOA_CONFOUNDERS_NATIONAL.exists():
        raise FileNotFoundError(f"Confounders not found at {MSOA_CONFOUNDERS_NATIONAL}")
    confounders = pd.read_csv(MSOA_CONFOUNDERS_NATIONAL)
    assert_unique_keys(confounders, ["msoa_cd"], label="MSOA confounders")
    assert_keys_subset(
        confounders,
        type_marginals.reset_index().rename(columns={"geography code": "msoa_cd"}),
        "msoa_cd",
        required_label="MSOA confounders",
        available_label="Census TS044",
    )
    if len(confounders) < 10:
        confounders = confounders.assign(decile=5)
    else:
        confounders = confounders.assign(
            decile=pd.qcut(
                confounders['income_dep_score'], 10, labels=range(1, 11), duplicates="drop"
            ).astype(int)
        )

    confounders = confounders.drop_duplicates(subset=['msoa_cd'])
    confounders_idx = confounders.set_index('msoa_cd')   # O(1) lookup

    return type_marginals, tenure_marginals, confounders_idx


def build_hdd_lookup() -> tuple:
    """Build the region -> city -> Heating Degree Day lookup used to assign
    each MSOA a weather-driven HDD value.

    Returns:
        (region_lookup, regional_hdd) — region_lookup maps MSOA code to
        official region name; regional_hdd maps REGION_TO_HDD_KEY city names
        to their HDD value (see module-level REGION_TO_HDD_KEY for the
        region-name -> representative-city mapping).

    Raises:
        FileNotFoundError: if the MSOA-to-region lookup file is missing.
    """
    from src.config.settings import MSOA_REGION_LOOKUP
    from src.utils.epw_parser import get_regional_hdd_map

    if not MSOA_REGION_LOOKUP.exists():
        raise FileNotFoundError(f"Region lookup not found at {MSOA_REGION_LOOKUP}")
    region_lookup = pd.read_csv(MSOA_REGION_LOOKUP).set_index("msoa21cd")["region"].to_dict()

    physics_dir = RAW_DIR / "physics"
    cities = list(REGIONAL_CENTERS.keys())
    regional_hdd = get_regional_hdd_map(physics_dir, cities)

    return region_lookup, regional_hdd


def filter_msoas_for_test_mode(msoa_codes: np.ndarray, region_lookup: dict) -> np.ndarray:
    """Apply E2E_TARGET_LAD / E2E_TARGET_REGION env-var subsetting, used only
    by the test/pilot pipeline to restrict synthesis to a small area. Returns
    msoa_codes unchanged if neither env var is set."""
    target_lad = os.environ.get("E2E_TARGET_LAD")
    if target_lad:
        logger.info(f"*** E2E TEST MODE: Filtering MSOAs for LAD '{target_lad}' ***")
        from src.config.settings import LOOKUP_PATH
        if not LOOKUP_PATH.exists():
            raise FileNotFoundError(f"Cannot find LAD lookup for E2E filter at {LOOKUP_PATH}")
        lad_lookup = pd.read_csv(LOOKUP_PATH, usecols=['msoa21cd', 'ladnm']).drop_duplicates()

        target_msoas = lad_lookup[lad_lookup['ladnm'].str.contains(target_lad, case=False, na=False)]['msoa21cd'].unique()

        msoa_codes = np.intersect1d(msoa_codes, target_msoas)
        assert len(msoa_codes) > 0, f"FATAL: E2E test failed. No MSOAs found for LAD '{target_lad}'!"
        logger.info(f"*** Filtered down to {len(msoa_codes)} MSOAs for {target_lad} ***")
        return msoa_codes

    target_region = os.environ.get("E2E_TARGET_REGION")
    if target_region:
        logger.info(f"*** REGIONAL PILOT: Filtering MSOAs for region '{target_region}' ***")
        region_msoas = [m for m, r in region_lookup.items() if target_region.lower() in r.lower()]
        msoa_codes = np.intersect1d(msoa_codes, region_msoas)
        assert len(msoa_codes) > 0, f"FATAL: No MSOAs found for region '{target_region}'!"
        logger.info(f"*** Filtered down to {len(msoa_codes)} MSOAs for {target_region} ***")
        return msoa_codes

    return msoa_codes


def run_national_synthesis():
    logger.info("--- STAGE 1: NATIONAL POPULATION SYNTHESIS V8 (Full 6,840 MSOAs) ---")

    seed_q = load_and_clean_seed_data()
    arrays = build_seed_lookup_arrays(seed_q)
    seed_pt = arrays["seed_pt"]
    seed_tenure = arrays["seed_tenure"]
    seed_imd = arrays["seed_imd"]
    seed_age = arrays["seed_age"]
    seed_thermal = arrays["seed_thermal"]
    seed_gas = arrays["seed_gas"]
    seed_area_band = arrays["seed_area_band"]
    pt_masks = arrays["pt_masks"]
    tenure_masks = arrays["tenure_masks"]

    type_marginals, tenure_marginals, confounders_idx = load_census_marginals()
    region_lookup, regional_hdd = build_hdd_lookup()

    msoa_codes = np.sort(confounders_idx.index.unique())
    msoa_codes = filter_msoas_for_test_mode(msoa_codes, region_lookup)

    all_results = []

    for msoa_idx, msoa_code in enumerate(tqdm(msoa_codes, desc="Synthesising MSOAs")):

        # Fast O(1) lookups (indexed DataFrames)
        if msoa_code not in type_marginals.index:
            continue
        m_type = type_marginals.loc[msoa_code]

        if msoa_code not in tenure_marginals.index:
            continue
        m_tenure = tenure_marginals.loc[msoa_code]

        if msoa_code not in confounders_idx.index:
            continue
        decile = confounders_idx.loc[msoa_code, 'decile']

        msoa_region = region_lookup.get(msoa_code, 'North West')
        hdd_key = REGION_TO_HDD_KEY.get(msoa_region, 'Manchester')
        msoa_hdd = regional_hdd.get(hdd_key, 2500)

        # Target marginals (same logic as before)
        target_type = {
            'House': (
                m_type.get('Accommodation type: Detached', 0) +
                m_type.get('Accommodation type: Semi-detached', 0) +
                m_type.get('Accommodation type: Terraced', 0)
            ),
            'Flat': (
                m_type.get('Accommodation type: In a purpose-built block of flats or tenement', 0) +
                m_type.get('Accommodation type: Part of a converted or shared house, including bedsits', 0)
            ),
            'Bungalow': m_type.get('Accommodation type: A caravan or other mobile or temporary structure', 0)
        }
        target_tenure = m_tenure.to_dict()

        # ------------------------------------------------------------------
        # IPF using numpy arrays — avoids seed_q.copy() (expensive for
        # large seed DataFrames) and avoids groupby inside the loop.
        # Pre-computed boolean masks (pt_masks, tenure_masks) are reused
        # across all MSOAs; only the weights array is reallocated.
        # ------------------------------------------------------------------
        weights = np.where(seed_imd == decile, 1.0, 0.05)

        for _ in range(5):
            # Property-type adjustment
            for t in PT_TYPES:
                target_val = target_type.get(t, 0)
                mask = pt_masks[t]
                curr = weights[mask].sum()
                if curr > 0 and target_val > 0:
                    weights = np.where(mask, weights * (target_val / curr), weights)

            # Tenure adjustment
            for t in TENURE_TYPES:
                target_val = target_tenure.get(t, 0)
                mask = tenure_masks[t]
                curr = weights[mask].sum()
                if curr > 0 and target_val > 0:
                    weights = np.where(mask, weights * (target_val / curr), weights)

        # 5 % weight cap (V15 Integrity Plan)
        total_w = weights.sum()
        if total_w > 0:
            cap = total_w * 0.05
            if (weights > cap).any():
                raise ValueError(f"FATAL: IPF weights severely skewed for MSOA {msoa_code}! Max weight exceeds 5% cap.")
            total_w = weights.sum()

        # Sampling — diversity-enforcing: try sampling without replacement to maximize unique HH
        rng = np.random.RandomState(RANDOM_SEED + msoa_idx)
        if total_w == 0:
            chosen = rng.choice(len(seed_q), size=N_HH_SAMPLES_PER_MSOA, replace=True)
        else:
            probs  = weights / total_w
            # Identify indices with non-zero probability
            nz_idx = np.where(probs > 0)[0]

            if len(nz_idx) >= N_HH_SAMPLES_PER_MSOA:
                # If we have enough candidates, sample without replacement for 100% diversity
                chosen = rng.choice(len(seed_q), size=N_HH_SAMPLES_PER_MSOA, replace=False, p=probs)
            else:
                # Fallback to with-replacement if the filtered pool is too small
                chosen = rng.choice(len(seed_q), size=N_HH_SAMPLES_PER_MSOA, replace=True, p=probs)

        # --------------------------------------------------------------------
        # Per-household physical parameters for GP emulator
        # Draw from truncated normal around archetype-nominal values (CV=10%)
        # so that within-archetype variance is captured by the emulator.
        # --------------------------------------------------------------------
        n_hh  = len(chosen)
        rng2  = np.random.RandomState(RANDOM_SEED + msoa_idx + 1)

        pt_chosen  = seed_pt[chosen]
        age_chosen = seed_age[chosen]

        # Floor area: look up archetype mean, then add Gaussian noise
        area_arr = np.array([
            ARCHETYPE_AREA_MEAN.get(age_chosen[i], {}).get(pt_chosen[i], 80.0)
            for i in range(n_hh)
        ], dtype=float)
        floor_area_arr = area_arr * (1.0 + rng2.normal(0, _AREA_CV, n_hh))
        floor_area_arr = np.clip(floor_area_arr, 20.0, None)

        # Wall U-value: age-band nominal + Gaussian noise
        wall_u_nom = np.array([
            CONSTRUCTION_SPECS.get(age_chosen[i], {"wall_u": 1.0})["wall_u"]
            for i in range(n_hh)
        ], dtype=float)
        wall_u_arr = wall_u_nom * (1.0 + rng2.normal(0, _PHYS_CV, n_hh))
        wall_u_arr = np.clip(wall_u_arr, 0.10, 3.0)

        # ACH: age-band nominal + Gaussian noise
        ach_nom = np.array([
            CONSTRUCTION_SPECS.get(age_chosen[i], {"ach": 1.0})["ach"]
            for i in range(n_hh)
        ], dtype=float)
        ach_arr = ach_nom * (1.0 + rng2.normal(0, _PHYS_CV, n_hh))
        ach_arr = np.clip(ach_arr, 0.15, 3.0)

        # WWR: form nominal + Gaussian noise
        wwr_nom = np.array([WWR_NOMINAL.get(pt_chosen[i], 0.15) for i in range(n_hh)], dtype=float)
        wwr_arr = wwr_nom * (1.0 + rng2.normal(0, _PHYS_CV, n_hh))
        wwr_arr = np.clip(wwr_arr, 0.05, 0.40)

        # Form code: deterministic lookup
        form_code_arr = np.array([FORM_CODE.get(pt_chosen[i], 3) for i in range(n_hh)], dtype=int)

        all_results.append(pd.DataFrame({
            'msoa21cd':             msoa_code,
            'property_type':        pt_chosen,
            'property_age':         age_chosen,
            'property_area_band':   seed_area_band[chosen],
            'tenure':               seed_tenure[chosen],
            'empirical_thermal_kwh': seed_thermal[chosen],
            'empirical_gas_kwh':    seed_gas[chosen],
            'IMD_BAND_ENG':         seed_imd[chosen],
            # GP emulator feature columns
            'floor_area':           floor_area_arr,
            'wall_u':               wall_u_arr,
            'ach':                  ach_arr,
            'wwr':                  wwr_arr,
            'form_code':            form_code_arr,
            'hdd':                  np.full(n_hh, msoa_hdd, dtype=float),
        }))

    if not all_results:
        raise ValueError("FATAL: No results generated — all MSOAs were skipped during synthesis.")

    final_synthetic_pop = pd.concat(all_results, ignore_index=True)

    # Guard against silent MSOA drops
    expected_msoas = len(msoa_codes)
    actual_msoas = final_synthetic_pop['msoa21cd'].nunique()
    logger.info(f"Adversarial Check: Synthesized {actual_msoas} out of {expected_msoas} expected MSOAs.")
    if actual_msoas < expected_msoas * 0.99: # Allowing max 1% missing due to empty marginals
        raise ValueError(f"FATAL: Silent MSOA drop detected! Dropped {expected_msoas - actual_msoas} MSOAs during synthesis!")

    # Apply Pandera Data Contract to enforce absolute statistical bounds
    from src.utils.data_contracts import population_schema
    logger.info("Validating synthetic population against absolute data contract...")
    final_synthetic_pop = population_schema.validate(final_synthetic_pop)

    # -------------------------------------------------------------
    # DATA LINEAGE TRACKING
    # -------------------------------------------------------------
    from src.utils.tracker import log_distribution
    log_distribution(final_synthetic_pop, 'empirical_thermal_kwh', '01_pop_synthesis_final_output', logger)

    # Parquet Export Validations
    print(f"Exporting {len(final_synthetic_pop)} rows to Parquet...")

    # E2E ROUTING & ASSERTIONS
    target_lad = os.environ.get("E2E_TARGET_LAD")
    target_region = os.environ.get("E2E_TARGET_REGION")
    if target_lad or target_region:
        # Guarantee GP Emulator features are intact
        assert not final_synthetic_pop[['floor_area', 'wall_u', 'ach']].isna().any().any(), "FATAL: GP Emulator inputs contain NaNs!"
        # Route to E2E output directory to avoid overwriting national data
        e2e_dir = PROCESSED_DIR / "tests" / "e2e_outputs"
        e2e_dir.mkdir(parents=True, exist_ok=True)
        output_path = e2e_dir / SYNTHETIC_POP_FILE
    else:
        output_path = PROCESSED_DIR / SYNTHETIC_POP_FILE

    final_synthetic_pop.to_parquet(output_path, index=False)
    logger.info(f"Synthetic Population saved to {output_path}")
    logger.info(f"Total Households: {len(final_synthetic_pop)}")

if __name__ == "__main__":
    run_national_synthesis()

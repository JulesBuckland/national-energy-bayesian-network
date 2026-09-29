"""
End-to-End Reproducibility Module.
Generates synthetic populations and confounders so the pipeline can be run out-of-the-box.
"""

from pathlib import Path  # noqa: F401  (tests patch generate_synthetic_data.Path)

import geopandas as gpd
import numpy as np
import pandas as pd

from src.config.settings import (
    BOUNDARIES_PATH,
    MSOA_CONFOUNDERS_NATIONAL,
    PROCESSED_DIR,
    setup_logging,
)

logger = setup_logging("SyntheticDataGenerator")


def generate_data() -> None:
    logger.info("Initializing Synthetic Data Generation...")

    # Ensure processed directory exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Check for boundaries file to get valid MSOAs
    if not BOUNDARIES_PATH.exists():
        logger.warning(f"Cannot find boundary file at {BOUNDARIES_PATH}.")
        logger.warning("Generating 6,800 dummy MSOA codes instead.")
        msoa_codes = [f"E02{str(i).zfill(6)}" for i in range(1, 6801)]
    else:
        logger.info(f"Loading valid MSOA codes from {BOUNDARIES_PATH}")
        gdf = gpd.read_file(BOUNDARIES_PATH)
        msoa_codes = gdf["MSOA21CD"].unique()

    np.random.seed(42)

    # 1. Generate MSOA Confounders
    logger.info("Generating MSOA Confounders...")
    confounders = pd.DataFrame(
        {"msoa_cd": msoa_codes, "income_dep_score": np.random.normal(0.15, 0.05, len(msoa_codes))}
    )
    confounders.to_csv(MSOA_CONFOUNDERS_NATIONAL, index=False)
    logger.info(f"Saved {len(confounders)} confounder records to {MSOA_CONFOUNDERS_NATIONAL}")

    # 2. Generate Synthetic Population
    # 100 households per MSOA for fast end-to-end reproducibility
    n_households = 100
    total_hh = len(msoa_codes) * n_households
    logger.info(f"Generating Synthetic Population ({total_hh} households)...")

    msoa_col = np.repeat(msoa_codes, n_households)

    population = pd.DataFrame(
        {
            "msoa21cd": msoa_col,
            "floor_area": np.random.normal(90, 20, total_hh).clip(min=30),
            "wall_u": np.random.normal(1.2, 0.4, total_hh).clip(min=0.1),
            "ach": np.random.normal(0.6, 0.2, total_hh).clip(min=0.1),
            "wwr": np.random.normal(0.2, 0.05, total_hh).clip(min=0.05, max=0.8),
            "form_code": np.random.randint(1, 5, total_hh),
            "hdd": np.random.normal(2200, 300, total_hh),
            "empirical_thermal_kwh": np.random.lognormal(mean=9.2, sigma=0.4, size=total_hh).clip(
                min=100
            ),
        }
    )

    output_path = PROCESSED_DIR / "national_synthetic_population_eti.parquet"
    population.to_parquet(output_path, index=False)
    logger.info(f"Saved synthetic population to {output_path}")
    logger.info("Out-of-the-box data generation complete! The Bayesian pipeline can now be run.")


if __name__ == "__main__":
    generate_data()

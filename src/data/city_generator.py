import argparse
from pathlib import Path
from typing import cast

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from sklearn.gaussian_process.kernels import Matern

# --- CONFIGURATION DEFAULTS ---
BETA_TH_TRUE = -0.25
BETA_INC_TRUE = -0.15
SIGMA_SPATIAL_TRUE = 0.5
SIGMA_ERR_TRUE = 0.2

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent
FAKE_RAW_DIR = BASE_DIR / "data" / "raw" / "fake"
FAKE_PROCESSED_DIR = BASE_DIR / "data" / "processed" / "fake"

FAKE_RAW_DIR.mkdir(parents=True, exist_ok=True)
FAKE_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def generate_spatial_grid(n_x: int, n_y: int) -> tuple[list[str], np.ndarray]:
    print(f"Generating {n_x}x{n_y} spatial grid...")
    polygons = []
    msoa_codes = []
    centroids = []

    idx = 0
    for i in range(n_x):
        for j in range(n_y):
            # 1x1 degree fake polygons
            poly = Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)])
            polygons.append(poly)
            msoa_codes.append(f"fake_{idx}")
            centroids.append([i + 0.5, j + 0.5])
            idx += 1

    gdf = gpd.GeoDataFrame({"MSOA21CD": msoa_codes, "geometry": polygons})

    # Save boundaries
    boundaries_path = FAKE_RAW_DIR / "fake_msoa_boundaries.gpkg"
    if boundaries_path.exists():
        boundaries_path.unlink()
    gdf.to_file(boundaries_path, driver="GPKG")

    return msoa_codes, np.array(centroids)


def generate_confounders_and_spatial_field(
    msoa_codes: list[str], centroids: np.ndarray, n_msoas: int
) -> pd.DataFrame:
    print("Generating continuous Matern spatial field and confounders...")
    # 1. Generate True Spatial Field (phi) using Matern GP (to avoid inverse crime)
    kernel = 1.0 * Matern(length_scale=3.0, nu=1.5)
    cov_matrix = kernel(centroids)
    # Add tiny jitter for numerical stability
    cov_matrix += np.eye(n_msoas) * 1e-6

    # Draw true spatial effects
    phi_true = np.random.multivariate_normal(np.zeros(n_msoas), cov_matrix)
    # Zero-mean center the field
    phi_true = phi_true - phi_true.mean()
    # Scale by true spatial sigma
    omega_true = SIGMA_SPATIAL_TRUE * phi_true

    # 2. Generate Income Deprivation Score (IMD)
    # We deliberately add high IID variance so it's not perfectly collinear with space
    # spatial gradient
    base_imd = centroids[:, 0] * 0.1 - centroids[:, 1] * 0.1
    base_imd = (base_imd - base_imd.mean()) / base_imd.std()

    # Add strong local variance
    iid_noise = np.random.normal(0, 1.0, n_msoas)
    income_dep_score = base_imd + iid_noise
    income_dep_score = (income_dep_score - income_dep_score.mean()) / income_dep_score.std()

    confounders = pd.DataFrame(
        {"msoa_cd": msoa_codes, "income_dep_score": income_dep_score, "true_omega": omega_true}
    )

    confounders.to_csv(FAKE_PROCESSED_DIR / "fake_msoa_confounders.csv", index=False)

    # Also generate LAD and Region lookups to satisfy pipeline
    lad_lookup = pd.DataFrame({"msoa21cd": msoa_codes, "ladnm": "FakeLAD"})
    lad_lookup.to_csv(FAKE_RAW_DIR / "fake_lookup.csv", index=False)

    region_lookup = pd.DataFrame({"msoa21cd": msoa_codes, "region": "London"})
    region_lookup.to_csv(FAKE_PROCESSED_DIR / "fake_msoa_region_lookup.csv", index=False)

    return confounders


def generate_census_marginals(msoa_codes: list[str], n_msoas: int) -> None:
    print("Generating fake census marginals...")

    # TS044
    ts044 = pd.DataFrame({"geography code": msoa_codes}).assign(
        **{
            "Accommodation type: Detached": np.random.randint(50, 150, n_msoas),
            "Accommodation type: Semi-detached": np.random.randint(50, 150, n_msoas),
            "Accommodation type: Terraced": np.random.randint(50, 150, n_msoas),
            "Accommodation type: In a purpose-built block of flats or tenement": np.random.randint(
                10, 50, n_msoas
            ),
            "Accommodation type: Part of a converted or shared house, including bedsits": np.random.randint(
                5, 20, n_msoas
            ),
            "Accommodation type: A caravan or other mobile or temporary structure": np.random.randint(
                0, 5, n_msoas
            ),
        }
    )
    ts044.to_csv(FAKE_RAW_DIR / "fake_census_housing.csv", index=False)

    # TS054
    records = [
        [msoa, code, count]
        for msoa in msoa_codes
        for code, count in [
            (1, np.random.randint(100, 300)),
            (3, np.random.randint(50, 150)),
            (5, np.random.randint(50, 150)),
        ]
    ]

    ts054 = pd.DataFrame(
        records,
        columns=[
            "Middle layer Super Output Areas Code",
            "Tenure of household (9 categories) Code",
            "Observation",
        ],
    )
    ts054.to_csv(FAKE_RAW_DIR / "fake_census_tenure.csv", index=False)


def generate_seed_microdata(confounders: pd.DataFrame, n_households: int) -> None:
    print(f"Generating fake NEED seed microdata ({n_households} households)...")

    archetypes_df = pd.read_csv(
        BASE_DIR / "data" / "raw" / "physics" / "physics_archetypes_baseline.csv"
    )
    archetypes = archetypes_df[["property_type", "property_age", "theoretical_gas_kwh"]].copy()

    # Sample households
    sampled_indices = np.random.choice(archetypes.index, size=n_households, replace=True)
    hh_df = archetypes.loc[sampled_indices].reset_index(drop=True)

    # Assign properties
    hh_df = hh_df.assign(
        IMD_BAND_ENG=np.random.randint(1, 11, n_households),
        FLOOR_AREA_BAND=np.random.randint(1, 5, n_households),
        Econs2022=2000.0,
    )

    # Assign MSOAs randomly
    hh_msoas = np.random.choice(confounders["msoa_cd"].to_numpy(), size=n_households)

    # Calculate Gcons2022
    def _calc_obs(i: int) -> float:
        msoa = hh_msoas[i]
        t_theory = cast(float, hh_df.loc[i, "theoretical_gas_kwh"])
        c_row = confounders[confounders["msoa_cd"] == msoa].iloc[0]
        z_inc = c_row["income_dep_score"]
        omega = c_row["true_omega"]
        err = np.random.normal(0, SIGMA_ERR_TRUE)
        log_y = np.log(t_theory) + BETA_TH_TRUE + (BETA_INC_TRUE * z_inc) + omega + err
        return np.exp(log_y)

    y_obs = [_calc_obs(i) for i in range(n_households)]

    # Map back age band
    age_map = {
        "Pre-1900": 1,
        "1900-1929": 2,
        "1930-1949": 3,
        "1950-1966": 4,
        "1967-1982": 5,
        "1983-1995": 6,
        "1996-2006": 7,
        "2007+": 8,
    }

    hh_df = hh_df.assign(
        Gcons2022=y_obs,
        PROP_TYPE=hh_df["property_type"],
        PROP_AGE_BAND=hh_df["property_age"].map(age_map),
    )

    hh_df.to_csv(FAKE_RAW_DIR / "fake_need_seed.csv", index=False)
    print("Seed microdata generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Fake City for UBEM Verification")
    parser.add_argument("--msoas-x", type=int, default=20, help="Number of MSOAs in X direction")
    parser.add_argument("--msoas-y", type=int, default=25, help="Number of MSOAs in Y direction")
    parser.add_argument("--households", type=int, default=5000, help="Number of seed households")
    args = parser.parse_args()

    n_msoas = args.msoas_x * args.msoas_y
    np.random.seed(42)  # Deterministic generation!
    msoa_codes, centroids = generate_spatial_grid(args.msoas_x, args.msoas_y)
    generate_census_marginals(msoa_codes, n_msoas)
    confounders = generate_confounders_and_spatial_field(msoa_codes, centroids, n_msoas)
    generate_seed_microdata(confounders, args.households)

    print(f"\n--- GROUND TRUTH FOR {n_msoas} MSOAs ---")
    print(f"Beta_th: {BETA_TH_TRUE}")
    print(f"Beta_inc: {BETA_INC_TRUE}")
    print(f"Sigma_spatial: {SIGMA_SPATIAL_TRUE}")
    print(f"Sigma_err: {SIGMA_ERR_TRUE}")
    print("Run pipeline with USE_FAKE_CITY=1")

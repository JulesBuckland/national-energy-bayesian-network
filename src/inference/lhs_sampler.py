"""
src/inference/lhs_sampler.py
============================
Generates Latin Hypercube Sampling (LHS) experimental designs for EnergyPlus
emulator training. Produces one CSV per archetype (32 files × 300 rows).

Outputs: data/raw/physics/lhs_designs/lhs_{archetype_slug}.csv
Columns: floor_area, wall_u, ach, wwr, form_code, age_band, property_type,
         archetype, wall_u_nominal, ach_nominal, wwr_nominal, area_nominal
"""

import numpy as np
import pandas as pd
import logging
from scipy.stats.qmc import LatinHypercube, scale
from pathlib import Path
from src.config.settings import REGIONAL_CENTERS, RAW_DIR
import argparse

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("LHSSampler")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PHYSICS_DIR = RAW_DIR / "physics"
OUTPUT_DIR = PHYSICS_DIR / "lhs_designs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_CSV = PHYSICS_DIR / "physics_archetypes_baseline.csv"

N_SAMPLES = 300   # LHS points per archetype
VARIATION = 0.30  # ± 30% around nominal values
from src.config.settings import RANDOM_SEED

# ---------------------------------------------------------------------------
# Construction specs per age band (from 01b/01c — UK Building Regs history)
# ---------------------------------------------------------------------------
CONSTRUCTION_SPECS = {
    "Pre-1900":  {"wall": 2.1,  "roof": 2.0,  "floor": 1.2, "window": 4.8, "ach": 1.5},
    "1900-1929": {"wall": 2.1,  "roof": 1.5,  "floor": 1.2, "window": 4.8, "ach": 1.5},
    "1930-1949": {"wall": 1.7,  "roof": 1.0,  "floor": 1.0, "window": 4.8, "ach": 1.2},
    "1950-1966": {"wall": 1.5,  "roof": 0.7,  "floor": 0.8, "window": 4.8, "ach": 1.0},
    "1967-1982": {"wall": 1.0,  "roof": 0.4,  "floor": 0.6, "window": 3.0, "ach": 0.8},
    "1983-1995": {"wall": 0.6,  "roof": 0.3,  "floor": 0.4, "window": 2.5, "ach": 0.6},
    "1996-2006": {"wall": 0.45, "roof": 0.2,  "floor": 0.3, "window": 2.0, "ach": 0.5},
    "2007+":     {"wall": 0.3,  "roof": 0.15, "floor": 0.2, "window": 1.6, "ach": 0.5},
}

FORM_SPECS = {
    "House":      {"wwr": 0.15, "form_code": 3, "exposed_walls": 4, "floors": 2},
    "Flat":       {"wwr": 0.20, "form_code": 0, "exposed_walls": 1, "floors": 1},
    "Bungalow":   {"wwr": 0.15, "form_code": 1, "exposed_walls": 4, "floors": 1},
    "Maisonette": {"wwr": 0.15, "form_code": 2, "exposed_walls": 2, "floors": 2},
}

# Physical constraints — never let parameters go outside plausible limits
U_WALL_MIN, U_WALL_MAX = 0.10, 3.0   # W/m²K (passivhaus to uninsulated solid brick)
ACH_MIN, ACH_MAX       = 0.15, 3.0   # air changes/hour
WWR_MIN, WWR_MAX       = 0.05, 0.40  # window-to-wall ratio


def generate_archetype_lhs(archetype: str, property_type: str, age_band: str,
                            area_nominal: float) -> pd.DataFrame:
    """
    Generate N_SAMPLES LHS points for one archetype.
    
    Inputs varied:
        floor_area  — ± VARIATION around area_nominal
        wall_u      — ± VARIATION around age-band nominal (clamped to physical range)
        ach         — ± VARIATION around age-band nominal (clamped to physical range)
        wwr         — ± VARIATION around form nominal (clamped to physical range)
    """
    specs = CONSTRUCTION_SPECS[age_band]
    form  = FORM_SPECS[property_type]

    wall_u_nom = specs["wall"]
    ach_nom    = specs["ach"]
    wwr_nom    = form["wwr"]

    # Bounds
    lo = np.array([
        area_nominal * (1 - VARIATION),
        max(U_WALL_MIN, wall_u_nom * (1 - VARIATION)),
        max(ACH_MIN,    ach_nom    * (1 - VARIATION)),
        max(WWR_MIN,    wwr_nom    * (1 - VARIATION)),
    ])
    hi = np.array([
        area_nominal * (1 + VARIATION),
        min(U_WALL_MAX, wall_u_nom * (1 + VARIATION)),
        min(ACH_MAX,    ach_nom    * (1 + VARIATION)),
        min(WWR_MAX,    wwr_nom    * (1 + VARIATION)),
    ])

    sampler = LatinHypercube(d=4, seed=RANDOM_SEED)
    raw = sampler.random(n=N_SAMPLES)
    X = scale(raw, l_bounds=lo, u_bounds=hi)

    # Calculate HDDs dynamically from EPW files
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.utils.epw_parser import get_regional_hdd_map
    cities = list(REGIONAL_CENTERS.keys())
    physics_dir = RAW_DIR / "physics"
    regional_hdd = get_regional_hdd_map(physics_dir, cities)

    df = pd.DataFrame(X, columns=["floor_area", "wall_u", "ach", "wwr"]).assign(
        form_code=form["form_code"],
        exposed_walls=form["exposed_walls"],
        floors=form["floors"],
        age_band=age_band,
        property_type=property_type,
        archetype=archetype,
        area_nominal=area_nominal,
        wall_u_nominal=wall_u_nom,
        ach_nominal=ach_nom,
        wwr_nominal=wwr_nom,
        sample_id=np.arange(N_SAMPLES),
        # Use the sample_id to deterministically but evenly distribute cities
        city=lambda d: [cities[i % len(cities)] for i in d["sample_id"]],
        hdd=lambda d: d["city"].apply(lambda c: regional_hdd[c]),
    )

    return df


def verify_discrepancy(output_dir: Path):
    """Check space-filling quality: centred L2 discrepancy should be < 0.02."""
    from scipy.stats.qmc import discrepancy
    files = sorted(output_dir.glob("lhs_*.csv"))
    if not files:
        logger.warning("No LHS files found — run without --verify-discrepancy first.")
        return
    all_pass = True
    for f in files:
        df = pd.read_csv(f)
        X = df[["floor_area", "wall_u", "ach", "wwr"]].values
        # Normalise to [0,1]
        X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        d = discrepancy(X_norm)
        status = "PASS" if d < 0.02 else "FAIL"
        if status == "FAIL":
            all_pass = False
        logger.info(f"{status} | {f.stem} | discrepancy={d:.5f}")
    logger.info("All discrepancy checks passed." if all_pass else
                "WARNING: Some designs exceed discrepancy threshold.")


def main(verify: bool = False):
    if verify:
        verify_discrepancy(OUTPUT_DIR)
        return

    if not BASELINE_CSV.exists():
        logger.error(f"Baseline CSV not found at {BASELINE_CSV}")
        return

    baseline = pd.read_csv(BASELINE_CSV)
    # Normalise column names defensively
    baseline.columns = [c.strip() for c in baseline.columns]

    # Identify the archetype, type, age, area columns
    type_col = next(c for c in baseline.columns if "type" in c.lower() and "property" in c.lower())
    age_col  = next(c for c in baseline.columns if "age"  in c.lower() and "property" in c.lower())
    area_col = next(c for c in baseline.columns if "area" in c.lower() and "mean" in c.lower())
    arch_col = next((c for c in baseline.columns if c.lower() == "archetype"), baseline.columns[0])

    logger.info(f"Loaded {len(baseline)} archetypes from {BASELINE_CSV}")

    all_frames = []
    for _, row in baseline.iterrows():
        ptype = str(row[type_col]).strip()
        age   = str(row[age_col]).strip()
        area  = float(row[area_col])
        arch  = str(row[arch_col]).strip()

        if ptype not in FORM_SPECS:
            logger.warning(f"Skipping unknown property type '{ptype}' for {arch}")
            continue
        if age not in CONSTRUCTION_SPECS:
            logger.warning(f"Skipping unknown age band '{age}' for {arch}")
            continue

        df = generate_archetype_lhs(arch, ptype, age, area)

        slug = arch.replace(" ", "_").replace("/", "-").replace("+", "plus")
        out_path = OUTPUT_DIR / f"lhs_{slug}.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"  Wrote {len(df)} samples → {out_path.name}")
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    combined_path = PHYSICS_DIR / "lhs_design_combined.csv"
    combined.to_csv(combined_path, index=False)
    logger.info(f"\nDone. {len(combined)} total LHS points → {combined_path}")
    logger.info(f"Individual designs → {OUTPUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-discrepancy", action="store_true",
                        help="Check space-filling quality of existing designs.")
    args = parser.parse_args()
    main(verify=args.verify_discrepancy)

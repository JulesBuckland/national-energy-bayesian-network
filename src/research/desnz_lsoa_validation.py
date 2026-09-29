"""DESNZ LSOA gas consumption aggregated to MSOA, compared against the
model's synthetic-population-derived y_mean.

Tests whether the NEED-seed-based synthetic population reproduces
official DESNZ aggregate gas consumption at MSOA level — a calibration
check on population synthesis, not the GP emulator.

Data: DESNZ sub-national gas consumption statistics (OGL v3.0)
Path: data/raw/energy/desnz_lsoa_gas_2010_2024.xlsx
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.config.settings import PROCESSED_DIR, setup_logging

logger = setup_logging("DESNZValidation")

DESNZ_PATH = Path("data/raw/energy/desnz_lsoa_gas_2010_2024.xlsx")
RESULTS_PATH = PROCESSED_DIR / "msoa_unified_results_inla.csv"


def load_desnz_lsoa(year: int = 2022, path: Path = DESNZ_PATH) -> pd.DataFrame:
    """Load one year of DESNZ LSOA gas data from the Excel workbook.

    Args:
        year: sheet name (2010–2024).
        path: path to the workbook.

    Returns:
        DataFrame with columns: la_code, msoa_code, lsoa_code,
        n_meters, total_kwh, mean_kwh, median_kwh.
    """
    if not path.exists():
        raise FileNotFoundError(f"DESNZ workbook not found: {path}")

    df = pd.read_excel(path, sheet_name=str(year), header=4)
    col_map = {}
    for col in df.columns:
        cl = str(col).replace("\n", " ").lower().strip()
        if "local authority code" in cl:
            col_map[col] = "la_code"
        elif "msoa code" in cl:
            col_map[col] = "msoa_code"
        elif "lsoa code" in cl:
            col_map[col] = "lsoa_code"
        elif "number" in cl and "meter" in cl and "non" not in cl:
            col_map[col] = "n_meters"
        elif "total" in cl and "consumption" in cl:
            col_map[col] = "total_kwh"
        elif "mean" in cl and "consumption" in cl:
            col_map[col] = "mean_kwh"
        elif "median" in cl and "consumption" in cl:
            col_map[col] = "median_kwh"

    df = df.rename(columns=col_map)

    required = ["msoa_code", "lsoa_code", "total_kwh", "n_meters"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns after rename: {missing}. Available: {list(df.columns)}")

    df = df[df["msoa_code"].astype(str).str.startswith("E")]
    for col in ["total_kwh", "n_meters"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["total_kwh", "n_meters"])

    logger.info(f"Loaded {len(df)} England LSOAs for {year}.")
    return df


def aggregate_to_msoa(lsoa_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate LSOA gas data to MSOA level.

    Returns DataFrame with columns: msoa_code, total_kwh, n_meters,
    mean_kwh_per_meter.
    """
    msoa = lsoa_df.groupby("msoa_code").agg(
        total_kwh=("total_kwh", "sum"),
        n_meters=("n_meters", "sum"),
    ).reset_index()

    msoa["mean_kwh_per_meter"] = msoa["total_kwh"] / msoa["n_meters"]
    logger.info(f"Aggregated to {len(msoa)} MSOAs.")
    return msoa


def compare_with_model(
    desnz_msoa: pd.DataFrame,
    results_path: Path = RESULTS_PATH,
) -> pd.DataFrame:
    """Merge DESNZ MSOA aggregates with model results and compute residuals.

    Returns merged DataFrame with columns including y_mean (model),
    desnz_mean_kwh (DESNZ), and residual.
    """
    if not results_path.exists():
        raise FileNotFoundError(f"Model results not found: {results_path}")

    model = pd.read_csv(results_path)
    logger.info(f"Loaded {len(model)} MSOAs from model results.")

    merged = model.merge(
        desnz_msoa[["msoa_code", "mean_kwh_per_meter", "n_meters", "total_kwh"]],
        left_on="msoa21cd",
        right_on="msoa_code",
        how="inner",
    )
    merged = merged.rename(columns={"mean_kwh_per_meter": "desnz_mean_kwh"})
    merged["residual"] = merged["y_mean"] - merged["desnz_mean_kwh"]
    merged["pct_error"] = merged["residual"] / merged["desnz_mean_kwh"] * 100

    logger.info(f"Matched {len(merged)} MSOAs between model and DESNZ.")
    return merged


def run_desnz_validation(year: int = 2022) -> dict:
    """Run the full DESNZ validation pipeline.

    Returns dict with correlation stats, residual summary, and the merged
    DataFrame.
    """
    lsoa = load_desnz_lsoa(year)
    msoa = aggregate_to_msoa(lsoa)
    merged = compare_with_model(msoa)

    r_pearson, p_pearson = stats.pearsonr(merged["y_mean"], merged["desnz_mean_kwh"])
    r_spearman, p_spearman = stats.spearmanr(merged["y_mean"], merged["desnz_mean_kwh"])

    mae = np.mean(np.abs(merged["residual"]))
    mape = np.mean(np.abs(merged["pct_error"]))
    rmse = np.sqrt(np.mean(merged["residual"] ** 2))

    results = {
        "year": year,
        "n_msoas": len(merged),
        "pearson_r": r_pearson,
        "pearson_p": p_pearson,
        "spearman_r": r_spearman,
        "spearman_p": p_spearman,
        "mae_kwh": mae,
        "mape_pct": mape,
        "rmse_kwh": rmse,
        "mean_residual_kwh": merged["residual"].mean(),
        "merged_df": merged,
    }

    logger.info(
        f"\nDESNZ vs Model ({year}):\n"
        f"  MSOAs matched: {len(merged)}\n"
        f"  Pearson r = {r_pearson:.4f} (p = {p_pearson:.2e})\n"
        f"  Spearman rho = {r_spearman:.4f} (p = {p_spearman:.2e})\n"
        f"  MAE = {mae:,.0f} kWh\n"
        f"  MAPE = {mape:.1f}%\n"
        f"  RMSE = {rmse:,.0f} kWh\n"
        f"  Mean residual = {merged['residual'].mean():,.0f} kWh"
    )

    out_dir = PROCESSED_DIR / "desnz_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / f"desnz_model_comparison_{year}.csv", index=False)
    logger.info(f"Saved comparison to {out_dir / f'desnz_model_comparison_{year}.csv'}")

    return results


if __name__ == "__main__":
    results = run_desnz_validation(2022)

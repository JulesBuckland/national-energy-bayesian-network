"""Prototype loader for EFUS 2017 (Energy Follow-Up Survey) data.

Loads the interview responses and half-hourly gas meter readings, then
computes annual gas consumption per dwelling for comparison against the
GP emulator's predictions. This is exploratory — not wired into the
main pipeline.

Data: UK Data Service SN 9434
Path: data/raw/efus_2017/ukda_9434_csv_r/csv/
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.settings import setup_logging

logger = setup_logging("EFUSLoader")

EFUS_DIR = Path("data/raw/efus_2017/ukda_9434_csv_r/csv")

DWELLING_TYPE_MAP = {
    1: "Detached",
    2: "Semi-detached",
    3: "Terraced",
    4: "Bungalow",
    5: "Converted flat",
    6: "Purpose-built flat",
}

FLOOR_AREA_MIDPOINTS = {1: 40, 2: 60, 3: 80, 4: 100, 5: 130, 6: 180}

DWELLING_AGE_MAP = {
    1: "pre-1919",
    2: "1919-1944",
    3: "1945-1964",
    4: "1965-1980",
    5: "1981-1990",
    6: "1991-2002",
    7: "2003+",
}

EPC_BAND_MAP = {1: "A-B", 2: "C", 3: "D", 4: "E-G"}


def load_interview_responses(efus_dir: Path = EFUS_DIR) -> pd.DataFrame:
    """Load and decode the EFUS interview responses CSV.

    Returns a DataFrame with one row per dwelling (CaseID), with decoded
    categorical columns and a midpoint floor_area_m2 estimate.
    """
    path = efus_dir / "selected_interview_responses_caseid.csv"
    if not path.exists():
        raise FileNotFoundError(f"EFUS interview file not found: {path}")

    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} EFUS interview responses.")

    df["dwelling_type"] = df["dwtype_efus"].map(DWELLING_TYPE_MAP)
    df["floor_area_m2"] = df["floor6x_efus"].map(FLOOR_AREA_MIDPOINTS)
    df["dwelling_age"] = df["dwage_efus"].map(DWELLING_AGE_MAP)
    df["epc_band"] = df["EPceeb12e_efus"].map(EPC_BAND_MAP)
    df["gas_heated"] = df["MainFuelUsedNM_efus"] == 1
    df["cavity_wall"] = df["WallType2x_efus"] == 1
    df["walls_insulated"] = df["InsulatedWalls_efus"] == 1

    return df


def load_gas_meter_data(efus_dir: Path = EFUS_DIR) -> pd.DataFrame:
    """Load all half-hourly gas meter CSVs and concatenate.

    Returns a DataFrame with columns: ReadingDate (datetime), MeterReading,
    CaseID.
    """
    gas_dir = efus_dir / "gasdata_csv"
    if not gas_dir.exists():
        raise FileNotFoundError(f"EFUS gas data directory not found: {gas_dir}")

    files = sorted(glob.glob(str(gas_dir / "gas_*.csv")))
    if not files:
        raise FileNotFoundError(f"No gas_*.csv files found in {gas_dir}")

    frames = []
    for f in files:
        chunk = pd.read_csv(f, parse_dates=["ReadingDate"])
        frames.append(chunk)
        logger.info(f"Loaded {len(chunk):,} rows from {Path(f).name}")

    df = pd.concat(frames, ignore_index=True)
    logger.info(f"Total gas meter readings: {len(df):,} across {df['CaseID'].nunique()} dwellings.")
    return df


def compute_annual_gas_kwh(meter_df: pd.DataFrame) -> pd.DataFrame:
    """Compute annual gas consumption (kWh) per dwelling from half-hourly
    meter readings.

    The meter readings are cumulative within each half-hour slot; we sum
    them per CaseID across the full monitoring period, then annualise based
    on the number of days monitored.

    Returns DataFrame with columns: CaseID, annual_gas_kwh, days_monitored.
    """
    meter_df = meter_df.copy()
    meter_df["date"] = meter_df["ReadingDate"].dt.date

    per_dwelling = (
        meter_df.groupby("CaseID")
        .agg(
            total_kwh=("MeterReading", "sum"),
            date_min=("date", "min"),
            date_max=("date", "max"),
        )
        .reset_index()
    )

    per_dwelling["days_monitored"] = (
        pd.to_datetime(per_dwelling["date_max"]) - pd.to_datetime(per_dwelling["date_min"])
    ).dt.days + 1

    # Readings are half-hourly kWh. Simple annualisation assumes uniform
    # consumption across the year, which overstates summer-monitored
    # dwellings' heating and understates winter-monitored ones. A
    # degree-day-weighted annualisation would be more accurate but
    # requires matching each dwelling to a weather station.
    per_dwelling["annual_gas_kwh"] = np.where(
        per_dwelling["days_monitored"] > 30,
        per_dwelling["total_kwh"] * 365.25 / per_dwelling["days_monitored"],
        np.nan,
    )

    valid = per_dwelling["annual_gas_kwh"].notna().sum()
    logger.info(
        f"Annualised gas for {valid}/{len(per_dwelling)} dwellings "
        f"(excluded {len(per_dwelling) - valid} with <30 days monitoring)."
    )
    return per_dwelling[["CaseID", "annual_gas_kwh", "days_monitored"]]


def build_efus_validation_dataset(efus_dir: Path = EFUS_DIR) -> pd.DataFrame:
    """Merge interview responses with annualised gas consumption.

    Returns a DataFrame ready for comparison against GP emulator predictions,
    with columns including floor_area_m2, dwelling_type, gas_heated,
    cavity_wall, walls_insulated, and annual_gas_kwh.
    """
    interviews = load_interview_responses(efus_dir)
    meter = load_gas_meter_data(efus_dir)
    annual = compute_annual_gas_kwh(meter)

    merged = interviews.merge(annual, on="CaseID", how="inner")
    merged = merged[merged["gas_heated"] & merged["annual_gas_kwh"].notna()]

    logger.info(
        f"EFUS validation dataset: {len(merged)} gas-heated dwellings with metered consumption."
    )
    return merged


if __name__ == "__main__":
    df = build_efus_validation_dataset()
    print(f"\nEFUS validation dataset: {len(df)} dwellings")
    print(f"Mean annual gas: {df['annual_gas_kwh'].mean():,.0f} kWh")
    print(f"Median annual gas: {df['annual_gas_kwh'].median():,.0f} kWh")
    print(f"\nDwelling types:\n{df['dwelling_type'].value_counts()}")
    print(f"\nEPC bands:\n{df['epc_band'].value_counts()}")

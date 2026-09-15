"""
epw_parser.py
==============
Dynamically parses CIBSE TRY 2016 / Exeter Prometheus .epw files to mathematically
calculate the Heating Degree Days (HDD) for the 9 English regions.

Base temperature for CIBSE standard: 15.5°C
Formula: HDD = sum(max(0, 15.5 - T_mean_daily)) over 365 days.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger("EPWParser")

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
try:
    from src.config.settings import BASE_TEMP_HDD, RAW_DIR, REGIONAL_CENTERS
except ImportError:
    BASE_TEMP_HDD = 15.5

def extract_daily_means(temperatures: np.ndarray) -> np.ndarray:
    num_days = len(temperatures) // 24
    trimmed = temperatures[:num_days * 24]
    return trimmed.reshape(num_days, 24).mean(axis=1)

def calculate_hdd(daily_means: np.ndarray, base_temp: float) -> float:
    return float(np.sum(np.maximum(0, base_temp - daily_means)))

def calculate_hdd_from_epw(epw_path: Path) -> float:
    """
    Parses an EPW file, extracts the 8760 hourly dry bulb temperatures,
    calculates daily means, and sums the Heating Degree Days.
    """
    if not epw_path.exists():
        raise FileNotFoundError(f"Missing EPW file: {epw_path}. Cannot calculate valid HDD without real weather data.")

    try:
        df = pd.read_csv(
            epw_path,
            skiprows=8,
            header=None,
            usecols=[6],
            names=["dry_bulb_temp"],
        )
        temperatures = pd.to_numeric(df["dry_bulb_temp"], errors="coerce").to_numpy()
        if len(temperatures) < 24 or not np.isfinite(temperatures).all():
            raise ValueError(
                "EPW must contain at least 24 finite hourly dry-bulb temperatures"
            )
        daily_means = extract_daily_means(temperatures)
        return calculate_hdd(daily_means, BASE_TEMP_HDD)
    except Exception as e:
        raise RuntimeError(f"Failed to parse EPW {epw_path}: {e}")

def get_regional_hdd_map(physics_dir: Path, cities: list[str]) -> dict[str, float]:
    """
    Given a list of cities and the physics directory containing their EPW files,
    returns a dictionary mapping each city to its calculated HDD.
    """
    hdd_map = {}
    for city in cities:
        epw_file = physics_dir / f"{city}_2030_ColdSnap.epw"
        hdd_map[city] = calculate_hdd_from_epw(epw_file)
        
    return hdd_map

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    PHYSICS_DIR = RAW_DIR / "physics"
    cities = list(REGIONAL_CENTERS.keys())
    hdd_map = get_regional_hdd_map(PHYSICS_DIR, cities)
    for city, hdd in hdd_map.items():
        logger.info(f"{city}: {hdd:.1f} HDD")

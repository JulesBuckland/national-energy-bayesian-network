import os
import sys

import numpy as np
import pandas as pd

# Add root to path
sys.path.append(os.getcwd())
from src.config.settings import PROCESSED_DIR, RAW_DIR


def check_ipf_convergence() -> None:
    print("--- CALCULATING IPF CONVERGENCE DIAGNOSTICS ---")

    # Target Marginals
    ts044_path = RAW_DIR / "census" / "ts044_extracted" / "census2021-ts044-msoa.csv"
    if not ts044_path.exists():
        ts044_path = RAW_DIR / "census" / "census2021-ts044-msoa.csv"

    ts044 = pd.read_csv(ts044_path)

    # Synthetic Population
    syn_path = PROCESSED_DIR / "national_synthetic_population_eti.csv"
    syn = pd.read_csv(syn_path)

    # Aggregated Synthetic Counts
    # Mapping TS044 types to our property_type (House, Flat, Bungalow)
    syn_counts = syn.groupby(["msoa21cd", "property_type"]).size().unstack(fill_value=0)

    # MSOA-level check
    msoas = syn["msoa21cd"].unique()
    mapes = []

    for msoa in msoas:
        if msoa not in ts044["geography code"].values:
            continue

        target = ts044[ts044["geography code"] == msoa].iloc[0]
        # target_house = Detached + Semi + Terraced
        t_house = (
            target["Accommodation type: Detached"]
            + target["Accommodation type: Semi-detached"]
            + target["Accommodation type: Terraced"]
        )
        t_flat = (
            target["Accommodation type: In a purpose-built block of flats or tenement"]
            + target["Accommodation type: Part of a converted or shared house, including bedsits"]
        )
        t_bungalow = target["Accommodation type: A caravan or other mobile or temporary structure"]

        # We sampled 100 per MSOA, so we need to normalize targets to 100
        total_target = t_house + t_flat + t_bungalow
        if total_target == 0:
            continue

        t_house_n = (t_house / total_target) * 100
        t_flat_n = (t_flat / total_target) * 100
        t_bung_n = (t_bungalow / total_target) * 100

        s_house = syn_counts.loc[msoa, "House"] if "House" in syn_counts.columns else 0
        s_flat = syn_counts.loc[msoa, "Flat"] if "Flat" in syn_counts.columns else 0
        s_bung = syn_counts.loc[msoa, "Bungalow"] if "Bungalow" in syn_counts.columns else 0

        # Abs Diff
        diff = np.abs(t_house_n - s_house) + np.abs(t_flat_n - s_flat) + np.abs(t_bung_n - s_bung)
        mapes.append(diff / 3)

    avg_mape = np.mean(mapes)
    max_mape = np.max(mapes)

    print(f"IPF Mean Validation Error: {avg_mape:.4f}%")
    print(f"IPF Max Validation Error: {max_mape:.4f}%")

    with open(PROCESSED_DIR / "ipf_convergence_report.txt", "w") as f:
        f.write("IPF Synthesis Validation Report\n")
        f.write("==============================\n")
        f.write(f"Mean MSOA-level Marginal Error: {avg_mape:.6f}%\n")
        f.write(f"Max MSOA-level Marginal Error: {max_mape:.6f}%\n")
        f.write(f"Total MSOAs Validated: {len(mapes)}\n")


if __name__ == "__main__":
    check_ipf_convergence()

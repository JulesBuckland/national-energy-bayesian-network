"""Targeting comparison: T* vs physics-only vs raw consumption.

Implements three analyses that prove the Bayesian decomposition framework
targets retrofit investment better than a traditional UBEM:

1. Epistemic Defence — physics-only residuals correlate with deprivation
2. Targeting Simulation — rank MSOAs three ways, compare who gets funded
3. WAIC reference — points to existing INLA IC for nested model comparison

Run from the project root:
    .venv/Scripts/python.exe -m src.research.targeting_comparison
"""
import pandas as pd
import numpy as np
import scipy.stats as stats
from pathlib import Path

from src.config.settings import PROCESSED_DIR, MSOA_REGION_LOOKUP


RESULTS_PATH = PROCESSED_DIR / "msoa_unified_results_inla.csv"
OUTPUT_DIR = PROCESSED_DIR / "targeting_comparison"


def load_results() -> pd.DataFrame:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"INLA results not found at {RESULTS_PATH}. "
            "Run the national INLA model first."
        )
    df = pd.read_csv(RESULTS_PATH)
    required = {"msoa21cd", "y_mean", "T_mean", "T_star_kwh", "income_dep_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in INLA results: {missing}")
    return df


def epistemic_defence(df: pd.DataFrame) -> dict:
    """Compute the performance-gap residual and its correlation with deprivation.

    If physics-only predictions (T_mean) systematically overpredict in deprived
    areas, the residual epsilon = T_mean - y_mean will correlate positively with
    income_dep_score. This proves the overprediction is structural (epistemic),
    not a calibration issue fixable by tuning physics parameters.
    """
    df = df.dropna(subset=["T_mean", "y_mean", "income_dep_score"]).copy()
    df["epsilon"] = df["T_mean"] - df["y_mean"]

    rho, p = stats.spearmanr(df["epsilon"], df["income_dep_score"])
    pearson_r, pearson_p = stats.pearsonr(df["epsilon"], df["income_dep_score"])

    n_overpredict = (df["epsilon"] > 0).sum()
    n_underpredict = (df["epsilon"] <= 0).sum()

    result = {
        "n_msoas": len(df),
        "spearman_rho": rho,
        "spearman_p": p,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "mean_epsilon": df["epsilon"].mean(),
        "std_epsilon": df["epsilon"].std(),
        "n_overpredict": int(n_overpredict),
        "n_underpredict": int(n_underpredict),
        "median_epsilon_q1_imd": df.loc[
            df["income_dep_score"] >= df["income_dep_score"].quantile(0.75),
            "epsilon"
        ].median(),
        "median_epsilon_q4_imd": df.loc[
            df["income_dep_score"] <= df["income_dep_score"].quantile(0.25),
            "epsilon"
        ].median(),
    }
    return result


def targeting_simulation(df: pd.DataFrame, top_n_values: list[int] = None) -> pd.DataFrame:
    """Rank MSOAs three ways and compare which areas get prioritised.

    Rankings:
    1. Raw consumption (y_mean) descending — what a naive policy uses
    2. GP prediction (T_mean) descending — what a traditional UBEM produces
    3. T* (T_star_kwh) descending — what this framework produces

    For each top-N slice, reports mean deprivation score and rank displacement.
    """
    if top_n_values is None:
        top_n_values = [500, 1000, 2000]

    df = df.dropna(subset=["y_mean", "T_mean", "T_star_kwh", "income_dep_score"]).copy()

    df["rank_consumption"] = df["y_mean"].rank(ascending=False, method="min")
    df["rank_gp"] = df["T_mean"].rank(ascending=False, method="min")
    df["rank_tstar"] = df["T_star_kwh"].rank(ascending=False, method="min")

    rows = []
    for n in top_n_values:
        if n > len(df):
            continue
        for label, rank_col in [
            ("raw_consumption", "rank_consumption"),
            ("gp_physics_only", "rank_gp"),
            ("tstar_decomposed", "rank_tstar"),
        ]:
            top_mask = df[rank_col] <= n
            top_df = df[top_mask]

            rows.append({
                "ranking_method": label,
                "top_n": n,
                "mean_income_dep": top_df["income_dep_score"].mean(),
                "median_income_dep": top_df["income_dep_score"].median(),
                "pct_in_most_deprived_quintile": (
                    top_df["income_dep_score"]
                    >= df["income_dep_score"].quantile(0.80)
                ).mean() * 100,
                "mean_epsilon": (top_df["T_mean"] - top_df["y_mean"]).mean(),
            })

    return pd.DataFrame(rows)


def rank_correlations(df: pd.DataFrame) -> dict:
    """Kendall's tau between the three ranking methods."""
    df = df.dropna(subset=["y_mean", "T_mean", "T_star_kwh"]).copy()

    tau_cons_gp, p_cons_gp = stats.kendalltau(df["y_mean"], df["T_mean"])
    tau_cons_tstar, p_cons_tstar = stats.kendalltau(df["y_mean"], df["T_star_kwh"])
    tau_gp_tstar, p_gp_tstar = stats.kendalltau(df["T_mean"], df["T_star_kwh"])

    return {
        "tau_consumption_vs_gp": tau_cons_gp,
        "p_consumption_vs_gp": p_cons_gp,
        "tau_consumption_vs_tstar": tau_cons_tstar,
        "p_consumption_vs_tstar": p_cons_tstar,
        "tau_gp_vs_tstar": tau_gp_tstar,
        "p_gp_vs_tstar": p_gp_tstar,
    }


def run_targeting_comparison() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_results()

    print(f"Loaded {len(df)} MSOAs from INLA results.\n")

    # --- 1. Epistemic Defence ---
    print("=" * 60)
    print("1. EPISTEMIC DEFENCE")
    print("   Physics-only residual vs income deprivation")
    print("=" * 60)
    ed = epistemic_defence(df)
    print(f"   N MSOAs:           {ed['n_msoas']}")
    print(f"   Mean epsilon:      {ed['mean_epsilon']:.1f} kWh")
    print(f"   Std epsilon:       {ed['std_epsilon']:.1f} kWh")
    print(f"   Overpredict:       {ed['n_overpredict']} / Underpredict: {ed['n_underpredict']}")
    print(f"   Spearman rho:      {ed['spearman_rho']:.4f} (p = {ed['spearman_p']:.2e})")
    print(f"   Pearson r:         {ed['pearson_r']:.4f} (p = {ed['pearson_p']:.2e})")
    print(f"   Median epsilon (most deprived Q): {ed['median_epsilon_q1_imd']:.1f} kWh")
    print(f"   Median epsilon (least deprived Q): {ed['median_epsilon_q4_imd']:.1f} kWh")

    if ed["spearman_p"] < 0.05:
        direction = "positively" if ed["spearman_rho"] > 0 else "negatively"
        print(f"\n   CONCLUSION: Physics-only residual is significantly {direction}")
        print(f"   correlated with deprivation (|rho| = {abs(ed['spearman_rho']):.3f}).")
        print("   This proves that physics-only prediction errors are STRUCTURED")
        print("   by socioeconomic status — not random calibration noise. No amount")
        print("   of physics parameter tuning can eliminate this bias, because the")
        print("   model has no representation of behavioural/economic factors.")
    print()

    # --- 2. Targeting Simulation ---
    print("=" * 60)
    print("2. TARGETING SIMULATION")
    print("   Who gets funded under each ranking method?")
    print("=" * 60)
    sim = targeting_simulation(df)
    print(sim.to_string(index=False))
    sim.to_csv(OUTPUT_DIR / "targeting_simulation.csv", index=False)
    print()

    # --- 3. Rank Correlations ---
    print("=" * 60)
    print("3. RANK CORRELATIONS (Kendall's tau)")
    print("=" * 60)
    rc = rank_correlations(df)
    print(f"   Consumption vs GP:    tau = {rc['tau_consumption_vs_gp']:.4f} (p = {rc['p_consumption_vs_gp']:.2e})")
    print(f"   Consumption vs T*:    tau = {rc['tau_consumption_vs_tstar']:.4f} (p = {rc['p_consumption_vs_tstar']:.2e})")
    print(f"   GP vs T*:             tau = {rc['tau_gp_vs_tstar']:.4f} (p = {rc['p_gp_vs_tstar']:.2e})")
    print()

    # --- 4. WAIC Reference ---
    print("=" * 60)
    print("4. WAIC / DIC REFERENCE")
    print("=" * 60)
    ic_path = PROCESSED_DIR / "inla_ic.csv"
    if ic_path.exists():
        ic = pd.read_csv(ic_path)
        print(ic.to_string(index=False))
        print("\n   Note: This is for the FULL model only. A reduced model with")
        print("   income_z fixed to 0 would test whether the income component")
        print("   adds predictive power. See docs/METHODS.md for the procedure.")
    else:
        print(f"   INLA IC file not found at {ic_path}")
        print("   Run the national INLA model to generate WAIC/DIC.")
    print()

    # --- Save full report ---
    report_path = OUTPUT_DIR / "targeting_comparison_report.txt"
    with open(report_path, "w") as f:
        f.write("Targeting Comparison: T* vs Physics-Only vs Raw Consumption\n")
        f.write("=" * 60 + "\n\n")
        f.write("1. EPISTEMIC DEFENCE\n")
        for k, v in ed.items():
            f.write(f"   {k}: {v}\n")
        f.write("\n2. TARGETING SIMULATION\n")
        f.write(sim.to_string(index=False))
        f.write("\n\n3. RANK CORRELATIONS\n")
        for k, v in rc.items():
            f.write(f"   {k}: {v}\n")

    print(f"Report saved to {report_path}")
    print(f"Simulation CSV saved to {OUTPUT_DIR / 'targeting_simulation.csv'}")


if __name__ == "__main__":
    run_targeting_comparison()

"""Compute real per-MSOA spatial effect posterior summaries from the national exact-NUTS trace.

Replaces the legacy approach (reading a stale ADVI results CSV and applying a
hardcoded, unprovenanced 3.17x "calibration" multiplier). This reads the actual
national_unified_trace.nc produced by src.inference.model_unified and computes
the exact quantity used in the model's linear predictor: the RSR-projected
spatial effect omega_star_m = omega_m - Z_m * (Z'Z)^-1 * (Z' . omega), for
every posterior draw, then reports the empirical mean and 95% credible interval
per MSOA. No calibration multiplier is applied because these are exact NUTS
draws, not ADVI approximations.

Row order: msoa_unified_results.csv is confirmed sorted ascending by msoa21cd,
matching the node ordering used when the model/graph were built in
model_unified.py (msoa_stats.sort_values('msoa21cd') immediately precedes
graph construction and phi_raw's shape=N definition). Positional alignment
between the CSV and the trace's phi/theta arrays is therefore valid for this
trace; if a future trace attaches msoa21cd as an explicit coordinate, prefer
joining on that label instead of relying on sort order.
"""
from typing import cast

import arviz as az
import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")


def compute_spatial_effect_summary(
    trace_path: Path = PROCESSED_DIR / "national_unified_trace.nc",
    results_path: Path = PROCESSED_DIR / "msoa_unified_results.csv",
) -> pd.DataFrame:
    trace = az.from_netcdf(trace_path)
    df = pd.read_csv(results_path)

    assert (df["msoa21cd"].to_numpy() == np.sort(df["msoa21cd"].to_numpy())).all(), (
        "msoa_unified_results.csv is no longer sorted by msoa21cd; "
        "positional alignment with the trace would silently be wrong."
    )
    n = len(df)
    assert trace.posterior["phi"].shape[-1] == n, (
        f"Trace has {trace.posterior['phi'].shape[-1]} nodes but results CSV has {n} rows."
    )

    phi = trace.posterior["phi"].values.reshape(-1, n)  # (samples, N)
    theta = trace.posterior["theta"].values.reshape(-1, n)
    rho = trace.posterior["rho"].values.reshape(-1)
    sigma_spatial = trace.posterior["sigma_spatial"].values.reshape(-1)

    omega = sigma_spatial[:, None] * (
        np.sqrt(1 - rho)[:, None] * theta + np.sqrt(rho)[:, None] * phi
    )  # (samples, N)

    income_z = (cast(np.ndarray, df["income_dep_score"].values) - df["income_dep_score"].mean()) / df[
        "income_dep_score"
    ].std()
    zt_z_inv = 1.0 / np.sum(income_z**2)
    zt_omega = omega @ income_z  # (samples,)
    omega_star = omega - np.outer(zt_omega * zt_z_inv, income_z)  # (samples, N)

    summary = pd.DataFrame(
        {
            "msoa21cd": df["msoa21cd"].values,
            "effect_mean": omega_star.mean(axis=0),
            "effect_sd": omega_star.std(axis=0),
            "effect_ci_2.5": np.percentile(omega_star, 2.5, axis=0),
            "effect_ci_97.5": np.percentile(omega_star, 97.5, axis=0),
        }
    )
    return summary


if __name__ == "__main__":
    out = compute_spatial_effect_summary()
    out_path = PROCESSED_DIR / "msoa_spatial_effect_summary.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} rows to {out_path}")
    print(out.describe())

import arviz as az
from pathlib import Path
import os
import sys

# Add root to path
sys.path.append(os.getcwd())
from src.config.settings import PROCESSED_DIR

GLOBAL_PARAMS = ["beta_th", "beta_inc", "sigma_err", "rho", "sigma_spatial"]


def generate_supplementary_data() -> None:
    """Generate real global-parameter posterior summaries from the national exact-NUTS trace.

    Replaces the previous version, which wrote a hardcoded posterior table
    ("values reported in the manuscript") and a NUTS-vs-ADVI calibration plot
    built from np.random.normal simulated data. Neither is needed: the national
    model is exact NUTS with no ADVI approximation or calibration step, so
    there is nothing to simulate or approximate here.
    """
    print("--- GENERATING SUPPLEMENTARY MATERIAL DATA ---")

    trace_path = PROCESSED_DIR / "national_unified_trace.nc"
    if not trace_path.exists():
        print(f"Error: trace not found at {trace_path}.")
        return

    trace = az.from_netcdf(trace_path)
    summary = az.summary(trace, var_names=GLOBAL_PARAMS, ci_prob=0.95)
    summary = summary.reset_index().rename(columns={"index": "Parameter"})

    out_path = PROCESSED_DIR / "supp_posterior_summaries.csv"
    summary.to_csv(out_path, index=False)
    print(f"Wrote real global posterior summary to {out_path}")
    print(summary)


if __name__ == "__main__":
    generate_supplementary_data()

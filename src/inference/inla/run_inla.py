"""Orchestrator for the R-INLA national/pilot inference run: the INLA
analogue of src.inference.model_unified.run_national_unified_model().

Data-preparation pipeline (population load -> GP emulator -> MSOA
aggregation -> Queen contiguity graph -> T_var/income_z/theory_log/y_obs)
is intentionally a parallel implementation here, NOT a shared/refactored
call into model_unified.py, even though the logic is equivalent. This is a
deliberate choice: model_unified.py is the already-verified, production
NUTS path that Table 3's manuscript numbers depend on, and keeping it
completely untouched removes any risk of the INLA migration destabilizing
it. The cost is ~100 lines of parallel (not literally duplicated-and-drifted)
logic between the two paths; both are covered by their own tests, and any
future change to one's data handling should be mirrored in the other by a
human reviewing both, not by shared code that could silently couple them.

Pipeline:
    1. prepare_national_msoa_dataset_for_inla() -- pure-ish data prep,
       returns the same node1/node2/T_var/income_z/theory_log/y_obs arrays
       build_unified_model() uses, plus msoa_stats (with msoa21cd).
    2. export_inla_inputs() -- writes the CSVs fit_inla.R reads.
    3. Rscript fit_inla.R (subprocess) -- fits the model, applies its own
       quality gate, writes results (or a DIAGNOSTIC-suffixed rejection).
    4. load_inla_results() -- reads results back, raising InlaGateFailedError
       if the gate rejected the fit (mirroring model_unified.py's
       RuntimeError on a failed NUTS convergence gate, just crossing the
       subprocess boundary instead of raising from within the same process).
    5. Computes T* (decoupled energy metric) and writes
       msoa_unified_results_inla{suffix}.csv -- a distinctly-named output,
       never overwriting the NUTS path's msoa_unified_results.csv, so both
       engines' results coexist for the cross-validation write-up.
"""

import os
import subprocess
from pathlib import Path
from typing import Any, cast

import geopandas as gpd
import joblib
import libpysal
import numpy as np
import pandas as pd

from src.config.settings import (
    BOUNDARIES_PATH,
    MSOA_CONFOUNDERS_NATIONAL,
    PILOT_MODE,
    PROCESSED_DIR,
    setup_logging,
)
from src.core.exceptions import DataValidationError
from src.inference.inla.data_export import export_inla_inputs
from src.inference.inla.read_results import (
    InlaGateFailedError,  # noqa: F401  (re-exported for callers of run_inla)
    build_spatial_effect_summary_from_inla,
    load_inla_results,
)
from src.inference.model_unified import GP_FEATURES, GP_MODEL_PATH, _use_csv_baseline, log_memory

logger = setup_logging("BayesianUnifiedNationalINLA")

FIT_INLA_R_SCRIPT = Path(__file__).parent / "fit_inla.R"


def _resolve_rscript_path() -> str:
    """Finds Rscript.exe: checks R_HOME env var first, then this session's
    known install location, then falls back to relying on PATH."""
    r_home = os.environ.get("R_HOME")
    if r_home:
        candidate = Path(r_home) / "bin" / "Rscript.exe"
        if candidate.exists():
            return str(candidate)
    known_install = Path(r"C:\Program Files\R\R-4.5.2\bin\Rscript.exe")
    if known_install.exists():
        return str(known_install)
    return "Rscript"


def prepare_national_msoa_dataset_for_inla(lad_codes: list | None = None) -> dict:
    """Loads/aggregates the national synthetic population into the arrays
    fit_inla.R needs. Mirrors model_unified.py's run_national_unified_model()
    data-prep section (see module docstring for why this is a parallel, not
    shared, implementation) -- INLA-specific differences: no
    icar_scaling_factor (INLA's scale.model=TRUE recomputes this natively)
    and no zt_z_inv_scalar (RSR is applied via fit_inla.R's extraconstr
    instead, see tests/unit/test_inference_inla_rsr.R).

    Args:
        lad_codes: if given, filters the REAL national household-level
            parquet down to just these LAD codes (via msoa_lad_lookup.csv)
            before running the identical GP-emulator/aggregation/graph
            pipeline used for the full national run -- e.g. for the
            Greater Manchester pilot subset (tasks/inla_migration_plan.md
            #3 step 7). This filters real production data to a smaller
            real subset; it does not use synthetic/fixture data.

    Returns:
        dict with keys: msoa_stats (DataFrame with msoa21cd), node1, node2
        (0-indexed edge lists), T_var, income_z, theory_log, y_obs, N, n_edges.
    """
    logger.info("--- STAGE 3 (INLA): UNIFIED NATIONAL BAYESIAN MODEL ---")
    log_memory("Initialization")

    data_path = PROCESSED_DIR / "national_synthetic_population_eti.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data at {data_path}")

    df = pd.read_parquet(data_path)

    if lad_codes:
        from src.config.settings import LAD_LOOKUP_PATH

        lookup = pd.read_csv(LAD_LOOKUP_PATH)
        subset_msoas = lookup[lookup["ladcd"].isin(lad_codes)]["msoa21cd"]
        before = df["msoa21cd"].nunique()
        df = df[df["msoa21cd"].isin(subset_msoas)]
        logger.info(
            f"*** SUBSET MODE: filtered {before} MSOAs down to "
            f"{df['msoa21cd'].nunique()} MSOAs for LAD codes {lad_codes} ***"
        )
        if df.empty:
            raise DataValidationError(
                f"LAD filter {lad_codes} matched zero households -- check the codes."
            )

    if GP_MODEL_PATH.exists():
        logger.info(f"Loading GP emulator from {GP_MODEL_PATH}...")
        payload = joblib.load(GP_MODEL_PATH)
        gp_model = payload["gp"]
        gp_scaler = payload["scaler"]

        missing_cols = [c for c in GP_FEATURES if c not in df.columns]
        if missing_cols:
            logger.warning(
                f"GP feature columns missing: {missing_cols}. Falling back to CSV baseline."
            )
            df = _use_csv_baseline(df)
        else:
            logger.info(f"Running GP predictions for {len(df):,} households...")
            X_hh_s = gp_scaler.transform(df[GP_FEATURES].values.astype(float))
            BATCH_SIZE = 20000
            T_preds, T_stds = [], []
            for i in range(0, len(X_hh_s), BATCH_SIZE):
                pred, std = gp_model.predict(X_hh_s[i : i + BATCH_SIZE], return_std=True)
                T_preds.append(pred)
                T_stds.append(std)
            T_pred = np.maximum(0.0, np.concatenate(T_preds))
            T_std = np.concatenate(T_stds)
            df = df.assign(theoretical_gas_kwh=T_pred * 277.778, T_std_kwh=T_std * 277.778)
            assert (df["theoretical_gas_kwh"] >= 0).all(), (
                "FATAL: Negative theoretical gas prediction detected!"
            )
            log_memory("Post-GP Prediction")
    else:
        logger.warning(f"GP emulator not found at {GP_MODEL_PATH}. Falling back to CSV baseline.")
        df = _use_csv_baseline(df)

    confounders = pd.read_csv(MSOA_CONFOUNDERS_NATIONAL).set_index("msoa_cd")

    if "T_std_kwh" in df.columns:
        df = df.assign(log_T_var=(df["T_std_kwh"] / df["theoretical_gas_kwh"].clip(lower=1)) ** 2)
        msoa_stats = (
            df.groupby("msoa21cd")
            .agg(
                y_mean=("empirical_thermal_kwh", "mean"),
                T_mean=("theoretical_gas_kwh", "mean"),
                T_var=("log_T_var", "mean"),
            )
            .reset_index()
        )
    else:
        msoa_stats = (
            df.groupby("msoa21cd")
            .agg(
                y_mean=("empirical_thermal_kwh", "mean"),
                T_mean=("theoretical_gas_kwh", "mean"),
                T_var=("empirical_thermal_kwh", lambda x: np.var(np.log(x + 1e-6))),
            )
            .reset_index()
        )

    msoa_stats = msoa_stats.merge(
        confounders.reset_index(), left_on="msoa21cd", right_on="msoa_cd", how="inner"
    )

    initial_len = len(msoa_stats)
    msoa_stats = msoa_stats.dropna(subset=["y_mean", "T_mean", "income_dep_score"])
    assert len(msoa_stats) / initial_len > 0.99, "CRITICAL: Spatial merge dropped >1% of data!"
    logger.info(f"Aggregated {len(df)} households into {len(msoa_stats)} MSOAs.")

    gdf = gpd.read_file(BOUNDARIES_PATH)
    gdf = (
        gdf[gdf["MSOA21CD"].isin(msoa_stats["msoa21cd"])]
        .sort_values("MSOA21CD")
        .reset_index(drop=True)
    )
    msoa_stats = msoa_stats.sort_values("msoa21cd").reset_index(drop=True)
    assert len(gdf) == len(msoa_stats), (
        f"FATAL: Dimension mismatch! GDF has {len(gdf)} but stats has {len(msoa_stats)}"
    )
    if len(gdf) == 0:
        raise DataValidationError(
            "FATAL: GeoDataFrame is empty after filtering! Check spatial boundary data."
        )

    # Any: libpysal's stubs type this as WSP, which lacks neighbors/id2i.
    w: Any = libpysal.weights.Queen.from_dataframe(
        gdf, ids=gdf["MSOA21CD"].tolist(), silence_warnings=True
    )
    node1, node2 = [], []
    for i, neighbors in w.neighbors.items():
        for j in neighbors:
            if w.id2i[i] < w.id2i[j]:
                node1.append(w.id2i[i])
                node2.append(w.id2i[j])
    node1 = np.array(node1)
    node2 = np.array(node2)
    logger.info(f"Built spatial graph: {len(msoa_stats)} nodes, {len(node1)} edges.")

    y_obs = np.log(msoa_stats["y_mean"].values)
    theory_log = np.log(msoa_stats["T_mean"].values)
    T_var = msoa_stats["T_var"].values

    std_val = msoa_stats["income_dep_score"].std()
    if np.isnan(std_val) or std_val == 0:
        income_z = np.zeros(len(msoa_stats))
    else:
        income_z = (
            cast(np.ndarray, msoa_stats["income_dep_score"].values)
            - msoa_stats["income_dep_score"].mean()
        ) / std_val

    return {
        "msoa_stats": msoa_stats,
        "node1": node1,
        "node2": node2,
        "T_var": T_var,
        "income_z": income_z,
        "theory_log": theory_log,
        "y_obs": y_obs,
        "N": len(msoa_stats),
        "n_edges": len(node1),
    }


def run_national_inla_model(
    check_laplace_agreement: bool = True,
    lad_codes: list | None = None,
    output_suffix: str | None = None,
) -> dict:
    """Effectful orchestrator: prepares data, exports it, invokes fit_inla.R
    as a subprocess, reads results back, computes T*, and writes
    msoa_unified_results_inla{suffix}.csv. Mirrors
    model_unified.py's run_national_unified_model() end-to-end, for the
    INLA engine.

    Args:
        lad_codes: passed through to prepare_national_msoa_dataset_for_inla
            -- filters to a real MSOA subset (e.g. the Greater Manchester
            pilot) instead of the full national dataset.
        output_suffix: overrides the default output-file suffix. Defaults
            to "_pilot" when PILOT_MODE is set, "_gm_pilot" when lad_codes
            is given (and PILOT_MODE is not), else "" (final national run)
            -- always something other than "" whenever this is not
            genuinely the full national run, so a subset/pilot result can
            never be mistaken for (or silently overwrite) the final one.

    Raises:
        InlaGateFailedError: if fit_inla.R's quality gate rejected the fit
            (mirrors the NUTS path's RuntimeError on a failed convergence gate).
        RuntimeError: if the Rscript subprocess itself fails for a reason
            other than the quality gate (e.g. a crash before the gate check).
    """
    data = prepare_national_msoa_dataset_for_inla(lad_codes=lad_codes)
    is_pilot = PILOT_MODE
    if output_suffix is not None:
        suffix = output_suffix
    elif is_pilot:
        suffix = "_pilot"
    elif lad_codes:
        suffix = "_gm_pilot"
    else:
        suffix = ""

    io_dir = PROCESSED_DIR / f"inla_io{suffix}"
    export_paths = export_inla_inputs(
        data["node1"],
        data["node2"],
        data["T_var"],
        data["income_z"],
        data["theory_log"],
        data["y_obs"],
        io_dir / "in",
    )

    output_dir = io_dir / "out"
    rscript = _resolve_rscript_path()
    cmd = [
        rscript,
        str(FIT_INLA_R_SCRIPT),
        "--nodes",
        str(export_paths["nodes_path"]),
        "--edges",
        str(export_paths["edges_path"]),
        "--output_dir",
        str(output_dir),
        "--mode",
        "pilot" if is_pilot else "final",
        "--check_laplace_agreement",
        "TRUE" if check_laplace_agreement else "FALSE",
    ]
    logger.info(f"Invoking: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    logger.info(result.stdout)
    if result.returncode != 0 and "GATE" not in result.stdout:
        logger.error(result.stderr)
        raise RuntimeError(
            f"fit_inla.R failed (exit {result.returncode}), not via the quality gate:\n{result.stderr}"
        )
    if result.stderr:
        logger.warning(result.stderr)

    # load_inla_results raises InlaGateFailedError itself if the gate rejected
    # the fit (detected via which metadata file is present) -- let it propagate.
    inla_results = load_inla_results(output_dir, n=data["N"])

    beta_inc_mean = inla_results["fixed_effects"].loc["income_z", "mean"]
    T_star = np.exp(data["y_obs"] - beta_inc_mean * data["income_z"])
    msoa_stats = data["msoa_stats"].assign(T_star_kwh=T_star)
    results_path = PROCESSED_DIR / f"msoa_unified_results_inla{suffix}.csv"
    msoa_stats.to_csv(results_path, index=False)
    logger.info(f"Saved INLA-decoupled T* results to {results_path}")

    spatial_effect_summary = build_spatial_effect_summary_from_inla(
        inla_results["random_effects"]["b"], msoa_stats["msoa21cd"].values
    )
    spatial_effect_path = PROCESSED_DIR / f"msoa_spatial_effect_summary_inla{suffix}.csv"
    spatial_effect_summary.to_csv(spatial_effect_path, index=False)
    logger.info(f"Saved INLA spatial effect summary to {spatial_effect_path}")

    return {
        "inla_results": inla_results,
        "msoa_stats": msoa_stats,
        "spatial_effect_summary": spatial_effect_summary,
        "results_path": results_path,
    }


if __name__ == "__main__":
    run_national_inla_model()

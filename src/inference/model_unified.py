import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import geopandas as gpd
import libpysal
import os
import joblib
import arviz as az
import math

try:
    import psutil
except ImportError:
    psutil = None

from src.config.settings import (
    PROCESSED_DIR, RAW_DIR,
    MCMC_SAMPLES, MCMC_TUNE, MCMC_CORES, MCMC_CHAINS,
    MCMC_MAX_RHAT, MCMC_MAX_DIVERGENCES, PILOT_MODE,
    RANDOM_SEED, setup_logging
)

GP_MODEL_PATH = PROCESSED_DIR / "gp_emulator.pkl"
GP_FEATURES = ["floor_area", "wall_u", "ach", "wwr", "form_code", "hdd"]

logger = setup_logging("BayesianUnifiedNational")


def _run_metadata(mode: str, draws: int, tune: int, chains: int) -> dict:
    import subprocess
    import datetime
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=RAW_DIR.parent.parent, text=True
        ).strip()
    except Exception:
        git_commit = "unknown"
    return {
        "mode": mode,
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "git_commit": git_commit,
        "generated_at_utc": datetime.datetime.utcnow().isoformat(),
    }


def log_memory(stage_name: str) -> None:
    if psutil is None:
        logger.info(f"[RAM USAGE - {stage_name}]: (psutil not installed, cannot track RAM)")
        return
    try:
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
    except Exception as exc:
        logger.warning(f"[RAM USAGE - {stage_name}]: unavailable ({exc})")
        return
    logger.info(f"[RAM USAGE - {stage_name}]: {mem_mb:.2f} MB")


def summarize_divergent_draws(trace, param_names: list) -> dict:
    """Diagnostic: split scalar posteriors by divergent/non-divergent draws."""
    diverging = trace.sample_stats["diverging"].values.astype(bool)
    result = {}
    for name in param_names:
        values = trace.posterior[name].values
        div_vals = values[diverging]
        nondiv_vals = values[~diverging]
        result[name] = {
            "diverging": {
                "n": int(div_vals.size),
                "mean": float(np.mean(div_vals)) if div_vals.size else None,
                "std": float(np.std(div_vals)) if div_vals.size else None,
                "min": float(np.min(div_vals)) if div_vals.size else None,
                "max": float(np.max(div_vals)) if div_vals.size else None,
            },
            "non_diverging": {
                "n": int(nondiv_vals.size),
                "mean": float(np.mean(nondiv_vals)) if nondiv_vals.size else None,
                "std": float(np.std(nondiv_vals)) if nondiv_vals.size else None,
                "min": float(np.min(nondiv_vals)) if nondiv_vals.size else None,
                "max": float(np.max(nondiv_vals)) if nondiv_vals.size else None,
            },
        }
    return result


def _use_csv_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback: merge theoretical_gas_kwh from the analytical CSV baseline."""
    from src.config.settings import RAW_DIR
    archetypes_path = RAW_DIR / "physics" / "physics_archetypes_baseline.csv"
    archetypes = pd.read_csv(archetypes_path)
    archetypes = archetypes[["property_type", "property_age", "theoretical_gas_kwh"]].drop_duplicates()
    merged = df.merge(archetypes, on=["property_type", "property_age"], how="left")
    if merged["theoretical_gas_kwh"].isna().any():
        raise ValueError("Failed to map CSV baseline to some archetypes.")
    return df.assign(theoretical_gas_kwh=merged["theoretical_gas_kwh"].values)


# ---------------------------------------------------------------------------
# Pipeline sub-functions extracted from the former monolithic
# run_national_unified_model(). Each handles one stage; the orchestrator
# at the bottom calls them in sequence.
# ---------------------------------------------------------------------------

def _load_population() -> pd.DataFrame:
    """Load the synthetic household population parquet."""
    target_lad = os.environ.get("E2E_TARGET_LAD")
    target_region = os.environ.get("E2E_TARGET_REGION")
    if target_lad or target_region:
        subset_name = target_lad or target_region
        logger.info(f"*** SUBSET MODE: Reading data for {subset_name} ***")
        data_path = PROCESSED_DIR / "tests" / "e2e_outputs" / "national_synthetic_population_eti.parquet"
    else:
        data_path = PROCESSED_DIR / "national_synthetic_population_eti.parquet"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Synthetic population not found at {data_path}.\n"
            "Build it with 'python -m src.data.population' (needs the licensed "
            "NEED seed in data/raw/), or generate a synthetic stand-in with "
            "'python -m src.data.generate_synthetic_data'. See README.md."
        )
    return pd.read_parquet(data_path)


def _apply_gp_emulator(df: pd.DataFrame) -> pd.DataFrame:
    """Run GP emulator predictions or fall back to CSV baseline."""
    if not GP_MODEL_PATH.exists():
        logger.warning(
            f"GP emulator not found at {GP_MODEL_PATH}. "
            "Falling back to analytical CSV baseline (power-law HLC scaling)."
        )
        return _use_csv_baseline(df)

    logger.info(f"Loading GP emulator from {GP_MODEL_PATH}...")
    payload = joblib.load(GP_MODEL_PATH)
    gp_model = payload["gp"]
    gp_scaler = payload["scaler"]

    missing_cols = [c for c in GP_FEATURES if c not in df.columns]
    if missing_cols:
        logger.warning(
            f"GP feature columns missing from parquet: {missing_cols}. "
            "Falling back to CSV archetype baseline."
        )
        return _use_csv_baseline(df)

    logger.info(f"Running GP predictions for {len(df):,} households...")
    X_hh = df[GP_FEATURES].values.astype(float)
    X_hh_s = gp_scaler.transform(X_hh)

    BATCH_SIZE = 20000
    T_preds, T_stds = [], []
    for i in range(0, len(X_hh_s), BATCH_SIZE):
        batch_X = X_hh_s[i:i + BATCH_SIZE]
        pred, std = gp_model.predict(batch_X, return_std=True)
        T_preds.append(pred)
        T_stds.append(std)

    T_pred = np.maximum(0.0, np.concatenate(T_preds))
    T_std = np.concatenate(T_stds)

    df = df.assign(
        theoretical_gas_kwh=T_pred * 277.778,
        T_std_kwh=T_std * 277.778
    )

    if not (df["theoretical_gas_kwh"] >= 0).all():
        raise ValueError("Negative theoretical gas prediction detected after GP emulation.")

    log_memory("Post-GP Prediction")
    return df


def _aggregate_to_msoa(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate household-level data to MSOA means and merge confounders."""
    from src.config.settings import MSOA_CONFOUNDERS_NATIONAL
    from src.utils.tracker import log_distribution

    confounders = pd.read_csv(MSOA_CONFOUNDERS_NATIONAL).set_index('msoa_cd')

    log_distribution(df, 'theoretical_gas_kwh', '02a_bayesian_input', logger)
    log_distribution(df, 'empirical_thermal_kwh', '02a_bayesian_input', logger)

    if "T_std_kwh" in df.columns:
        df = df.assign(log_T_var=(df['T_std_kwh'] / df['theoretical_gas_kwh'].clip(lower=1)) ** 2)
        msoa_stats = df.groupby('msoa21cd').agg(
            y_mean=('empirical_thermal_kwh', 'mean'),
            T_mean=('theoretical_gas_kwh', 'mean'),
            T_var=('log_T_var', 'mean')
        ).reset_index()
        logger.info("Using GP predictive variance for Jensen's correction.")
    else:
        msoa_stats = df.groupby('msoa21cd').agg(
            y_mean=('empirical_thermal_kwh', 'mean'),
            T_mean=('theoretical_gas_kwh', 'mean'),
            T_var=('empirical_thermal_kwh', lambda x: np.var(np.log(x + 1e-6)))
        ).reset_index()
        logger.info("Using empirical within-MSOA variance for Jensen's correction (GP fallback).")

    msoa_stats = msoa_stats.merge(
        confounders.reset_index(), left_on='msoa21cd', right_on='msoa_cd', how='inner'
    )

    initial_len = len(msoa_stats)
    msoa_stats = msoa_stats.dropna(subset=['y_mean', 'T_mean', 'income_dep_score'])
    if len(msoa_stats) / initial_len <= 0.99:
        raise ValueError(
            f"Spatial merge dropped {initial_len - len(msoa_stats)} of {initial_len} "
            f"MSOAs (>{1:.0%} loss). Check confounder/boundary alignment."
        )

    logger.info(f"Aggregated {len(df)} households into {len(msoa_stats)} MSOAs.")
    log_memory("Post-Aggregation Data Load")
    return msoa_stats


def _build_spatial_graph(msoa_stats: pd.DataFrame) -> tuple:
    """Build Queen contiguity graph and compute ICAR scaling factor.

    Returns:
        (msoa_stats_sorted, gdf_sorted, node1, node2, icar_scaling_factor)
    """
    from src.config.settings import BOUNDARIES_PATH
    from src.inference.icar_scaling import compute_icar_scaling_factor

    gdf = gpd.read_file(BOUNDARIES_PATH)
    gdf = gdf[gdf['MSOA21CD'].isin(msoa_stats['msoa21cd'])].sort_values('MSOA21CD').reset_index(drop=True)
    msoa_stats = msoa_stats.sort_values('msoa21cd').reset_index(drop=True)

    if len(gdf) != len(msoa_stats):
        raise ValueError(f"Dimension mismatch: GDF has {len(gdf)} but stats has {len(msoa_stats)}")
    if len(gdf) == 0:
        raise ValueError("GeoDataFrame is empty after filtering. Check spatial boundary data.")

    w = libpysal.weights.Queen.from_dataframe(gdf, ids=gdf['MSOA21CD'].tolist(), silence_warnings=True)

    node1, node2 = [], []
    for i, neighbors in w.neighbors.items():
        for j in neighbors:
            if w.id2i[i] < w.id2i[j]:
                node1.append(w.id2i[i])
                node2.append(w.id2i[j])

    node1 = np.array(node1)
    node2 = np.array(node2)

    logger.info(f"Built spatial graph: {len(msoa_stats)} nodes, {len(node1)} edges.")
    log_memory("Sparse Graph Contiguity Built")

    if node1.ndim != 1 or node2.ndim != 1:
        raise ValueError("Graph arrays must be 1D vectors.")
    if node1.shape != node2.shape:
        raise ValueError("node1 and node2 graph arrays have mismatched shapes.")
    if not (np.all((node1 >= 0) & (node1 < len(msoa_stats)))):
        raise ValueError("node1 contains out-of-bounds indices.")
    if not (np.all((node2 >= 0) & (node2 < len(msoa_stats)))):
        raise ValueError("node2 contains out-of-bounds indices.")

    icar_scaling_factor = compute_icar_scaling_factor(node1, node2, len(msoa_stats))
    logger.info(f"ICAR BYM2 scaling factor: {icar_scaling_factor:.4f}")

    return msoa_stats, gdf, node1, node2, icar_scaling_factor


def _prepare_tensors(msoa_stats: pd.DataFrame) -> tuple:
    """Prepare log-transformed observations, theory, income_z, and RSR scalar.

    Returns:
        (y_obs, theory_log, T_var, income_z, zt_z_inv_scalar)
    """
    y_obs = np.log(msoa_stats['y_mean'].values)
    theory_log = np.log(msoa_stats['T_mean'].values)
    T_var = msoa_stats['T_var'].values

    std_val = msoa_stats['income_dep_score'].std()
    if np.isnan(std_val) or std_val == 0:
        income_z = np.zeros(len(msoa_stats))
    else:
        income_z = (msoa_stats['income_dep_score'].values - msoa_stats['income_dep_score'].mean()) / std_val

    Z = income_z.reshape(-1, 1)
    zt_z_inv_scalar = np.linalg.pinv(Z.T @ Z)[0, 0]

    log_memory("RSR Orthogonal Projection Created")
    return y_obs, theory_log, T_var, income_z, zt_z_inv_scalar


def _check_convergence(trace, chains: int) -> az.InferenceData:
    """Gate on divergences and r_hat. Raises RuntimeError if either fails."""
    n_divergences = int(trace.sample_stats["diverging"].sum())
    logger.info(f"Convergence check: {n_divergences} divergences (max allowed {MCMC_MAX_DIVERGENCES}).")

    if n_divergences > MCMC_MAX_DIVERGENCES:
        diag_summary = summarize_divergent_draws(
            trace, ["rho", "sigma_spatial", "sigma_err", "beta_th", "beta_inc"]
        )
        logger.warning(f"DIAGNOSTIC: divergent vs non-divergent draw stats: {diag_summary}")
        diag_path = PROCESSED_DIR / "national_unified_trace_DIAGNOSTIC_FAILED_GATE.nc"
        if diag_path.exists():
            diag_path.unlink(missing_ok=True)
        trace.to_netcdf(diag_path)
        logger.warning(f"DIAGNOSTIC trace saved to {diag_path}")
        raise RuntimeError(
            f"{n_divergences} divergences exceeds the allowed maximum of "
            f"{MCMC_MAX_DIVERGENCES}. Refusing to save results as converged."
        )

    summary = az.summary(trace, round_to="none")
    if chains >= 2:
        max_r_hat = summary["r_hat"].max()
        logger.info(f"Convergence check: max r_hat={max_r_hat:.4f} (max allowed {MCMC_MAX_RHAT}).")
        if max_r_hat >= MCMC_MAX_RHAT:
            raise RuntimeError(
                f"max r_hat={max_r_hat:.4f} exceeds the allowed maximum of "
                f"{MCMC_MAX_RHAT}. Refusing to save results as converged."
            )
    else:
        logger.warning(
            f"Only {chains} chain(s) — r_hat is not meaningful, skipping gate."
        )

    return summary


def _save_results(
    trace, unified_model, summary, msoa_stats, y_obs, income_z, is_pilot: bool,
    draws: int, tune: int, chains: int,
) -> None:
    """Compute LOO, T*, and persist all outputs."""
    logger.info("Computing Log Likelihood...")
    with unified_model:
        pm.compute_log_likelihood(trace)

    trace.attrs.update(_run_metadata(
        mode="pilot" if is_pilot else "final", draws=draws, tune=tune, chains=chains
    ))

    suffix = "_pilot" if is_pilot else ""
    trace_path = PROCESSED_DIR / f"national_unified_trace{suffix}.nc"
    if trace_path.exists():
        trace_path.unlink(missing_ok=True)
    trace.to_netcdf(trace_path)
    logger.info("Trace saved.")

    logger.info("Computing PSIS-LOO...")
    try:
        loo_result = az.loo(trace, pointwise=True)
        loo_str = str(loo_result)
        high_k = int((loo_result.pareto_k.values > 0.7).sum()) if hasattr(loo_result, "pareto_k") else 0
    except Exception as e:
        loo_str = f"LOO computation failed: {e}"
        high_k = 0

    with open(PROCESSED_DIR / f"nuts_loo{suffix}.txt", "w") as f:
        f.write("--- PSIS-LOO ---\n")
        f.write(loo_str + "\n\n")
        f.write(f"MSOAs with Pareto k > 0.7 (unreliable LOO estimate): {high_k}\n\n")
        f.write("--- DIAGNOSTICS ---\n")
        f.write(str(summary[['ess_bulk', 'ess_tail', 'r_hat']]) + "\n")

    b_inc_mean = trace.posterior['beta_inc'].mean().item()
    T_star = np.exp(y_obs - b_inc_mean * income_z)

    msoa_stats = msoa_stats.assign(T_star_kwh=T_star)
    msoa_stats.to_csv(PROCESSED_DIR / f"msoa_unified_results{suffix}.csv", index=False)
    logger.info(f"Saved T* results (mode={'pilot' if is_pilot else 'final'}).")
    log_memory("Final Exit")


# ---------------------------------------------------------------------------
# Model definition (unchanged)
# ---------------------------------------------------------------------------

def build_unified_model(
    N: int,
    node1: np.ndarray,
    node2: np.ndarray,
    T_var: np.ndarray,
    income_z: np.ndarray,
    theory_log: np.ndarray,
    y_obs: np.ndarray,
    icar_scaling_factor: float,
    zt_z_inv_scalar: float,
    rho_alpha: float = 1.0,
    rho_beta: float = 1.0,
    sigma_spatial_prior_sigma: float = 0.5,
    sigma_err_prior_sigma: float = 0.5,
) -> pm.Model:
    """Build (but do not sample) the unified national Bayesian spatial model.

    Sparse edge-list ICAR (O(E) memory) with BYM2 scaling, RSR projection
    against income_z, and Jensen's variance correction.
    """
    with pm.Model() as unified_model:
        beta_th = pm.Normal("beta_th", mu=-0.3, sigma=0.1)
        beta_inc = pm.Normal("beta_inc", mu=0.0, sigma=0.5)

        rho = pm.Beta("rho", alpha=rho_alpha, beta=rho_beta)
        sigma_spatial = pm.HalfNormal("sigma_spatial", sigma=sigma_spatial_prior_sigma)

        phi_raw = pm.Flat("phi_raw", shape=N)
        pm.Potential("icar_penalty", -0.5 * pm.math.sum((phi_raw[node1] - phi_raw[node2]) ** 2))
        zero_sum_stdev = 0.001
        pm.Potential(
            "icar_zerosum",
            -0.5 * pt.pow(pt.sum(phi_raw) / (zero_sum_stdev * N), 2)
            - pt.log(pt.sqrt(2.0 * np.pi))
            - pt.log(zero_sum_stdev * N),
        )
        phi = pm.Deterministic("phi", phi_raw / np.sqrt(icar_scaling_factor))

        theta_raw = pm.Normal("theta_raw", mu=0.0, sigma=1.0, shape=N)
        theta = pm.Deterministic("theta", theta_raw - pm.math.mean(theta_raw))

        omega = sigma_spatial * (pm.math.sqrt(1 - rho) * theta + pm.math.sqrt(rho) * phi)

        Z_tensor = pt.as_tensor_variable(income_z)
        Zt_omega = pt.sum(Z_tensor * omega)
        projection = Z_tensor * (zt_z_inv_scalar * Zt_omega)
        omega_star = omega - projection

        mu = theory_log - (T_var / 2.0) + beta_th + beta_inc * income_z + omega_star

        sigma_err = pm.HalfNormal("sigma_err", sigma=sigma_err_prior_sigma)
        pm.Normal("y", mu=mu, sigma=sigma_err, observed=y_obs)

    return unified_model


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_national_unified_model(
    rho_alpha: float = 1.0,
    rho_beta: float = 1.0,
    sigma_spatial_prior_sigma: float = 0.5,
    sigma_err_prior_sigma: float = 0.5,
    target_accept: float = 0.99,
    draws_override: int = None,
    tune_override: int = None,
) -> az.InferenceData:
    """Execute the national unified Bayesian inference pipeline.

    Loads data, runs GP emulator, builds spatial graph, samples with NUTS,
    gates convergence, and saves T* results.
    """
    logger.info("--- STAGE 3: UNIFIED NATIONAL BAYESIAN MODEL ---")
    log_memory("Initialization")

    # 1. Load and emulate
    df = _load_population()
    df = _apply_gp_emulator(df)

    # 2. Aggregate to MSOA
    msoa_stats = _aggregate_to_msoa(df)

    # 3. Spatial graph
    msoa_stats, gdf, node1, node2, icar_scaling_factor = _build_spatial_graph(msoa_stats)

    # 4. Prepare tensors
    y_obs, theory_log, T_var, income_z, zt_z_inv_scalar = _prepare_tensors(msoa_stats)

    # 5. Build model
    N = len(msoa_stats)
    unified_model = build_unified_model(
        N=N, node1=node1, node2=node2, T_var=T_var, income_z=income_z,
        theory_log=theory_log, y_obs=y_obs,
        icar_scaling_factor=icar_scaling_factor, zt_z_inv_scalar=zt_z_inv_scalar,
        rho_alpha=rho_alpha, rho_beta=rho_beta,
        sigma_spatial_prior_sigma=sigma_spatial_prior_sigma,
        sigma_err_prior_sigma=sigma_err_prior_sigma,
    )

    # 6. Sample
    is_pilot = PILOT_MODE
    draws = 10 if is_pilot else (draws_override if draws_override is not None else MCMC_SAMPLES)
    tune = 10 if is_pilot else (tune_override if tune_override is not None else MCMC_TUNE)
    chains = 2 if is_pilot else MCMC_CHAINS
    cores = min(chains, MCMC_CORES)

    with unified_model:
        logger.info(f"Sampling: {draws} draws, {tune} tune, {chains} chains, {cores} cores")
        trace = pm.sample(
            draws=draws, tune=tune, chains=chains, cores=cores,
            random_seed=RANDOM_SEED, target_accept=target_accept,
            progressbar=False,
        )

    # 7. Convergence gate
    summary = _check_convergence(trace, chains)

    # 8. Save
    _save_results(
        trace, unified_model, summary, msoa_stats, y_obs, income_z,
        is_pilot=is_pilot, draws=draws, tune=tune, chains=chains,
    )

    return trace


if __name__ == "__main__":
    run_national_unified_model()

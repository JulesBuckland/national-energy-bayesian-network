import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import geopandas as gpd
import libpysal
import os
import joblib
import arviz as az

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
# GP input features — must match columns produced by src/inference/gp_emulator.py
GP_FEATURES = ["floor_area", "wall_u", "ach", "wwr", "form_code", "hdd"]

logger = setup_logging("BayesianUnifiedNational")


def _run_metadata(mode: str, draws: int, tune: int, chains: int) -> dict:
    """Provenance stamp attached to every saved trace/results file, so no
    output file's origin (pilot vs final, code version, when) is ever
    ambiguous. A run in PILOT_MODE writes to distinctly-suffixed files (see
    run_national_unified_model) rather than overwriting the final ones, but
    this stamp is the authoritative record either way.
    """
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
    """Logs the current memory usage of the process.

    Args:
        stage_name (str): The name of the current pipeline stage.
    """
    if psutil is None:
        logger.info(f"[RAM USAGE - {stage_name}]: (psutil not installed, cannot track RAM)")
        return
    try:
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
    except Exception as exc:
        # This is diagnostics only, so it must never be able to abort a run
        # that is otherwise fine - a multi-hour national fit should not die
        # because a memory reading failed.
        logger.warning(f"[RAM USAGE - {stage_name}]: unavailable ({exc})")
        return
    logger.info(f"[RAM USAGE - {stage_name}]: {mem_mb:.2f} MB")

def summarize_divergent_draws(trace, param_names: list) -> dict:
    """Pure summary of scalar posterior parameters, split by whether their draw
    diverged, to diagnose *why* NUTS is diverging (e.g. concentrated at a `rho`
    boundary, or an extreme `sigma`) without changing the convergence gate's
    pass/fail decision at all — this is diagnostic-only, called from the
    failure branch right before the existing RuntimeError is raised.
    """
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
    """Fallback: merge theoretical_gas_kwh from the analytical CSV baseline.

    Returns a new DataFrame by adding a 'theoretical_gas_kwh' column
    based on the 'property_type' and 'property_age' archetype mapping.

    Args:
        df (pd.DataFrame): The household dataframe.

    Raises:
        ValueError: If any archetype fails to map to the baseline CSV.
    """
    from src.config.settings import RAW_DIR
    archetypes_path = RAW_DIR / "physics" / "physics_archetypes_baseline.csv"
    archetypes = pd.read_csv(archetypes_path)
    archetypes = archetypes[["property_type", "property_age", "theoretical_gas_kwh"]].drop_duplicates()
    # Merge and return a new DataFrame (pure function)
    merged = df.merge(archetypes, on=["property_type", "property_age"], how="left")
    if merged["theoretical_gas_kwh"].isna().any():
        raise ValueError("FATAL: Failed to map CSV baseline to some archetypes! Missing values found.")
    return df.assign(theoretical_gas_kwh=merged["theoretical_gas_kwh"].values)


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

    Spatial prior matches pm.ICAR's own logp exactly (pairwise_difference +
    soft zero_sum — see pm.ICAR.dist's logp source) via a sparse edge-list
    formulation (O(E) memory, not the O(N^2)/374MB dense W matrix a direct
    pm.ICAR(W=...) call would require at national scale). Critically this is
    a Flat base + explicit soft zero-sum Potential, NOT an independent
    per-node Normal(0,1) prior — that used to be a real bug here: an
    independent per-node prior competes with the pairwise penalty
    component-wise instead of only constraining the gauge-fixing sum,
    corrupting identifiability of `rho`. The BYM2 geometric-mean scaling
    factor (icar_scaling_factor, from icar_scaling.compute_icar_scaling_factor)
    is applied on top so `rho` is interpretable as the true spatial-vs-
    unstructured variance split (Riebler et al. 2016 / Simpson et al. 2017).

    Args:
        N: number of MSOA nodes.
        node1, node2: 1D edge-list arrays (0-indexed), one entry per edge.
        T_var: within-MSOA log-variance used for the Jensen's correction.
        income_z: standardized income deprivation score per MSOA.
        theory_log: log of the physics-based theoretical baseline per MSOA.
        y_obs: log of the observed empirical thermal energy per MSOA.
        icar_scaling_factor: BYM2 scaling factor for this graph.
        zt_z_inv_scalar: precomputed (Z'Z)^-1 scalar for the RSR projection.
        rho_alpha, rho_beta: Beta(rho_alpha, rho_beta) prior on the spatial
            mixing parameter. Default (1,1) is uniform. Exposed as a
            parameter (rather than hardcoded) to let a diagnostic rerun test
            whether pulling the prior mass away from the rho=1 boundary
            (e.g. Beta(2,2)) changes sampling behaviour, without touching
            production settings until/unless that's confirmed to help.
        sigma_spatial_prior_sigma, sigma_err_prior_sigma: HalfNormal scale
            for the sigma_spatial/sigma_err priors. Default 0.5 each,
            matching production. Exposed for the same reason: to test
            whether a tighter prior (matching where the posterior actually
            lives) improves sampling, before deciding whether to change it
            for real.

    Returns:
        The unsampled pm.Model.
    """
    with pm.Model() as unified_model:
        # Pure Physics Bias
        beta_th = pm.Normal("beta_th", mu=-0.3, sigma=0.1)

        # Socioeconomic Rationing Elasticity
        beta_inc = pm.Normal("beta_inc", mu=0.0, sigma=0.5)

        # Spatial Components (BYM2)
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

        # Unstructured spatial noise
        theta_raw = pm.Normal("theta_raw", mu=0.0, sigma=1.0, shape=N)
        theta = pm.Deterministic("theta", theta_raw - pm.math.mean(theta_raw))

        # The mixture
        omega = sigma_spatial * (pm.math.sqrt(1 - rho) * theta + pm.math.sqrt(rho) * phi)

        # Apply Restricted Spatial Regression (RSR) Projection to the spatial effect
        # This prevents omega from stealing variance from income_z
        # O(N) algebraic optimization: omega_star = omega - Z * (Z'Z)^-1 * (Z' omega)
        Z_tensor = pt.as_tensor_variable(income_z)
        Zt_omega = pt.sum(Z_tensor * omega)
        projection = Z_tensor * (zt_z_inv_scalar * Zt_omega)
        omega_star = omega - projection

        # Unified Structural Mean with Variance Correction
        mu = theory_log - (T_var / 2.0) + beta_th + beta_inc * income_z + omega_star

        # 5. Likelihood
        sigma_err = pm.HalfNormal("sigma_err", sigma=sigma_err_prior_sigma)
        pm.Normal("y", mu=mu, sigma=sigma_err, observed=y_obs)

    return unified_model


def run_national_unified_model(
    rho_alpha: float = 1.0,
    rho_beta: float = 1.0,
    sigma_spatial_prior_sigma: float = 0.5,
    sigma_err_prior_sigma: float = 0.5,
    target_accept: float = 0.99,
    draws_override: int = None,
    tune_override: int = None,
) -> az.InferenceData:
    """Executes the National Unified Bayesian Inference Model.

    This function performs the following pipeline:
    1. Loads the synthetic household population and aggregates it to the MSOA level.
    2. Applies the Gaussian Process (GP) emulator for theoretical heating loads.
    3. Builds a sparse Queen contiguity spatial matrix (O(E) complexity).
    4. Projects the income confounder to create a Restricted Spatial Regression (RSR)
       orthogonal component.
    5. Executes a PyMC NUTS sampler (1D ICAR via edge-lists) to decouple
       socioeconomic rationing from physical building efficiency.
    6. Saves the MCMC trace and the decoupled energy metric (T*).

    Args:
        rho_alpha, rho_beta, sigma_spatial_prior_sigma, sigma_err_prior_sigma:
            passed straight through to build_unified_model() — see its
            docstring. Defaults match production exactly; only pass
            non-default values for a deliberate diagnostic/experimental run,
            never as a silent way to route around a failed convergence gate.
        target_accept: NUTS target acceptance rate passed to pm.sample().
            Default 0.99 matches production. Raising it forces a smaller
            step size (more, cheaper-to-verify leapfrog steps per sample) —
            the standard first-line remedy for divergences that doesn't
            touch the model's priors at all.
        draws_override, tune_override: if set, override MCMC_SAMPLES/
            MCMC_TUNE from settings.py for this call only (PILOT_MODE's
            10/10 still takes priority over these if PILOT_MODE is set).
            Unlike target_accept, more draws/tune is only a legitimate fix
            for poor mixing (high r_hat/low ESS on chains that already have
            zero or near-zero divergences) — never a way to average over
            genuinely divergent, biased sampling.

    Returns:
        az.InferenceData: The generated PyMC posterior trace.
    """
    logger.info("--- STAGE 3: UNIFIED NATIONAL BAYESIAN MODEL ---")
    log_memory("Initialization")

    # 1. Load Data
    target_lad = os.environ.get("E2E_TARGET_LAD")
    target_region = os.environ.get("E2E_TARGET_REGION")
    if target_lad or target_region:
        subset_name = target_lad or target_region
        logger.info(f"*** SUBSET MODE: Reading data for {subset_name} ***")
        data_path = PROCESSED_DIR / "tests" / "e2e_outputs" / "national_synthetic_population_eti.parquet"
    else:
        data_path = PROCESSED_DIR / "national_synthetic_population_eti.parquet"

    if not data_path.exists():
        # Raise rather than return: returning normally here made the CLI report
        # "Run complete." after doing nothing, and exit 0, so a missing input
        # looked like a successful run in both the terminal and CI.
        raise FileNotFoundError(
            f"Synthetic population not found at {data_path}.\n"
            "Build it with 'python -m src.data.population' (needs the licensed "
            "NEED seed in data/raw/), or generate a synthetic stand-in with "
            "'python -m src.data.generate_synthetic_data'. See README.md."
        )

    df = pd.read_parquet(data_path)

    # -----------------------------------------------------------------------
    # GP Emulator: replace power-law HLC scaling with a trained surrogate.
    # Requires columns [floor_area, wall_u, ach, wwr, form_code] in the parquet
    # (added by src/data/population.py after Step 6 modifications).
    # Falls back to the CSV baseline if the GP model file does not exist.
    # -----------------------------------------------------------------------
    if GP_MODEL_PATH.exists():
        logger.info(f"Loading GP emulator from {GP_MODEL_PATH}...")
        payload = joblib.load(GP_MODEL_PATH)
        gp_model  = payload["gp"]
        gp_scaler = payload["scaler"]

        # Check all required feature columns are present
        missing_cols = [c for c in GP_FEATURES if c not in df.columns]
        if missing_cols:
            logger.warning(
                f"GP feature columns missing from parquet: {missing_cols}.\n"
                "Ensure src/data/population.py has been updated (Step 6).\n"
                "Falling back to CSV archetype baseline."
            )
            df = _use_csv_baseline(df)
        else:
            logger.info(f"Running GP predictions for {len(df):,} households...")
            X_hh = df[GP_FEATURES].values.astype(float)
            X_hh_s = gp_scaler.transform(X_hh)

            # Batch the predictions: scoring all households in one call exhausts
            # roughly 10 GB of RAM and is killed by the OS.
            BATCH_SIZE = 20000
            T_preds, T_stds = [], []
            import math
            n_batches = math.ceil(len(X_hh_s) / BATCH_SIZE)
            for i in range(0, len(X_hh_s), BATCH_SIZE):
                batch_X = X_hh_s[i:i+BATCH_SIZE]
                pred, std = gp_model.predict(batch_X, return_std=True)
                T_preds.append(pred)
                T_stds.append(std)

            T_pred = np.concatenate(T_preds)
            T_std = np.concatenate(T_stds)

            # Clamp the Gaussian tails - the surrogate can return small negative
            # values, which are not physically meaningful as energy demand.
            T_pred = np.maximum(0.0, T_pred)

            df = df.assign(
                theoretical_gas_kwh=T_pred * 277.778,
                T_std_kwh=T_std * 277.778
            )

            # E2E Inline Assertion: Bounds Check
            assert (df["theoretical_gas_kwh"] >= 0).all(), "FATAL: Negative theoretical gas prediction detected!"

            log_memory("Post-GP Prediction")
    else:
        logger.warning(
            f"GP emulator not found at {GP_MODEL_PATH}. "
            "Falling back to analytical CSV baseline (power-law HLC scaling)."
        )
        df = _use_csv_baseline(df)

    # Load confounders for Z_inc
    from src.config.settings import MSOA_CONFOUNDERS_NATIONAL
    conf_path = MSOA_CONFOUNDERS_NATIONAL
    confounders = pd.read_csv(conf_path).set_index('msoa_cd')

    # Pre-aggregate to MSOA level (preserving arithmetic mean for mass conservation)
    # We aggregate empirical_thermal_kwh (which includes electric heating now!)
    # Pre-aggregate to MSOA level (preserving arithmetic mean for mass conservation)
    # T_var uses GP predictive variance if available, else empirical within-MSOA log-variance
    if "T_std_kwh" in df.columns:
        # GP predictive variance: propagate per-household GP uncertainty to MSOA mean
        # We use the Delta method approximation for Var(log T) ≈ (sigma / mu)^2
        df = df.assign(log_T_var=(df['T_std_kwh'] / df['theoretical_gas_kwh'].clip(lower=1)) ** 2)
        msoa_stats = df.groupby('msoa21cd').agg(
            y_mean=('empirical_thermal_kwh', 'mean'),
            T_mean=('theoretical_gas_kwh', 'mean'),
            T_var =('log_T_var', 'mean')
        ).reset_index()
        logger.info("Using GP predictive variance for Jensen's correction.")
    else:
        # Fallback: empirical within-MSOA log-variance (original method)
        msoa_stats = df.groupby('msoa21cd').agg(
            y_mean=('empirical_thermal_kwh', 'mean'),
            T_mean=('theoretical_gas_kwh', 'mean'),
            T_var =('empirical_thermal_kwh', lambda x: np.var(np.log(x + 1e-6)))
        ).reset_index()
        logger.info("Using empirical within-MSOA variance for Jensen's correction (GP fallback).")

    msoa_stats = msoa_stats.merge(confounders.reset_index(), left_on='msoa21cd', right_on='msoa_cd', how='inner')
    # (Dynamic PySAL spatial indices are generated later, so no predefined node_idx drop needed)

    # -------------------------------------------------------------
    # DATA LINEAGE TRACKING
    # -------------------------------------------------------------
    from src.utils.tracker import log_distribution
    log_distribution(df, 'theoretical_gas_kwh', '02a_bayesian_input', logger)
    log_distribution(df, 'empirical_thermal_kwh', '02a_bayesian_input', logger)

    initial_len = len(msoa_stats)
    msoa_stats = msoa_stats.dropna(subset=['y_mean', 'T_mean', 'income_dep_score'])
    assert len(msoa_stats) / initial_len > 0.99, "CRITICAL: Spatial merge dropped >1% of data!"

    logger.info(f"Aggregated {len(df)} households into {len(msoa_stats)} MSOAs.")
    log_memory("Post-Aggregation Data Load")

    # 2. Build Sparse Spatial Adjacency Matrix
    from src.config.settings import BOUNDARIES_PATH
    boundaries_path = BOUNDARIES_PATH
    gdf = gpd.read_file(boundaries_path)

    # Align GDF and MSOA stats perfectly
    gdf = gdf[gdf['MSOA21CD'].isin(msoa_stats['msoa21cd'])].sort_values('MSOA21CD').reset_index(drop=True)
    msoa_stats = msoa_stats.sort_values('msoa21cd').reset_index(drop=True)

    # E2E Inline Assertion: Matrix Dimension Match
    assert len(gdf) == len(msoa_stats), f"FATAL: Dimension mismatch! GDF has {len(gdf)} but stats has {len(msoa_stats)}"
    if len(gdf) == 0:
        raise ValueError("FATAL: GeoDataFrame is empty after filtering! Check spatial boundary data.")

    # Build Queen contiguity weights
    w = libpysal.weights.Queen.from_dataframe(gdf, ids=gdf['MSOA21CD'].tolist(), silence_warnings=True)

    # Convert to node1, node2 lists for PyMC ICAR (extremely RAM efficient)
    node1, node2 = [], []
    for i, neighbors in w.neighbors.items():
        for j in neighbors:
            if w.id2i[i] < w.id2i[j]:
                node1.append(w.id2i[i])
                node2.append(w.id2i[j])

    node1 = np.array(node1)
    node2 = np.array(node2)

    logger.info(f"Built spatial graph: {len(msoa_stats)} nodes, {len(node1)} edges. No chunking required!")
    log_memory("Sparse Graph Contiguity Built")

    # Prepare Tensors
    y_obs = np.log(msoa_stats['y_mean'].values)
    theory_log = np.log(msoa_stats['T_mean'].values)

    # -------------------------------------------------------------
    # SHAPE ASSERTIONS
    # -------------------------------------------------------------
    assert node1.ndim == 1 and node2.ndim == 1, "FATAL: Graph arrays must be 1D vectors"
    assert node1.shape == node2.shape, "FATAL: node1 and node2 graph connectivity arrays have mismatched shapes!"

    # Ensure node1 and node2 only contain valid indices
    assert np.all((node1 >= 0) & (node1 < len(msoa_stats))), "FATAL: node1 contains out-of-bounds indices"
    assert np.all((node2 >= 0) & (node2 < len(msoa_stats))), "FATAL: node2 contains out-of-bounds indices"

    # BYM2 geometric-mean scaling factor (Riebler et al. 2016 / Simpson et al. 2017),
    # computed once from the graph structure alone — a plain numpy/scipy computation,
    # not a PyMC operation, so it happens before the model context below.
    from src.inference.icar_scaling import compute_icar_scaling_factor
    icar_scaling_factor = compute_icar_scaling_factor(node1, node2, len(msoa_stats))
    logger.info(f"ICAR BYM2 scaling factor: {icar_scaling_factor:.4f}")

    T_var = msoa_stats['T_var'].values

    # Z_inc standardization
    std_val = msoa_stats['income_dep_score'].std()
    if np.isnan(std_val) or std_val == 0:
        income_z = np.zeros(len(msoa_stats))
    else:
        income_z = (msoa_stats['income_dep_score'].values - msoa_stats['income_dep_score'].mean()) / std_val

    # 3. Restricted Spatial Regression (RSR) Projection Matrix
    # We project out Z from the ICAR to prevent Hodges-Reich confounding.
    # To prevent O(N^2) dense matrix multiplication in the MCMC loop, we use O(N) scalar math:
    Z = income_z.reshape(-1, 1)
    Zt_Z_inv_scalar = np.linalg.pinv(Z.T @ Z)[0, 0]

    log_memory("RSR Orthogonal Projection Created")

    # 4. Bayesian Model Definition
    N = len(msoa_stats)
    n_edges = len(node1)
    unified_model = build_unified_model(
        N=N, node1=node1, node2=node2, T_var=T_var, income_z=income_z,
        theory_log=theory_log, y_obs=y_obs,
        icar_scaling_factor=icar_scaling_factor, zt_z_inv_scalar=Zt_Z_inv_scalar,
        rho_alpha=rho_alpha, rho_beta=rho_beta,
        sigma_spatial_prior_sigma=sigma_spatial_prior_sigma,
        sigma_err_prior_sigma=sigma_err_prior_sigma,
    )

    # Determine run mode: PILOT_MODE controls MCMC draws independently of data subsetting.
    # Pilot output is routed to distinctly-suffixed files (never overwrites the
    # final national result) and every saved trace carries a metadata stamp
    # (see _run_metadata) recording which mode produced it.
    is_pilot = PILOT_MODE

    draws = 10 if is_pilot else (draws_override if draws_override is not None else MCMC_SAMPLES)
    tune = 10 if is_pilot else (tune_override if tune_override is not None else MCMC_TUNE)
    chains = 2 if is_pilot else MCMC_CHAINS
    # Each chain peaks at ~1.2 GB; with ≥16 GB RAM, parallel chains are safe.
    # On the original 8 GB machine this was pinned to 1; 32 GB allows full parallelism.
    cores = min(chains, MCMC_CORES)

    # --- Sampling: PyMC NUTS with sequential chains for 8GB RAM ---
    with unified_model:
        sample_kwargs = {
            "draws": draws, "tune": tune,
            "chains": chains, "cores": cores,
            "random_seed": RANDOM_SEED, "target_accept": target_accept,
            # rich's live progress bar writes Unicode (e.g. U+2009 thin space)
            # that crashes under Windows' legacy cp1252 console encoding
            # (UnicodeEncodeError from rich._win32_console) - logger.info calls
            # already report progress, so the bar is disabled rather than relied on.
            "progressbar": False,
            # NOTE: tried restricting var_names to drop phi_raw/theta_raw
            # (unused by any downstream code, ~1GB each at N=6853/5000
            # draws/4 chains) to save memory. Reverted: pm.compute_log_likelihood()
            # below requires every free RV -- including phi_raw/theta_raw --
            # to be present in the trace, so this isn't a safe trim despite
            # nothing else reading those two variables directly.
        }
        logger.info(f"Sampling with PyMC NUTS: {draws} draws, {tune} tune, {chains} chains, {cores} cores")
        trace = pm.sample(**sample_kwargs)

    # --- Convergence gate: refuse to persist anything as a result until the
    # fit is verified, not just logged (a warning here previously let
    # unconverged output flow straight into the manuscript). ---
    n_divergences = int(trace.sample_stats["diverging"].sum())
    logger.info(f"Convergence check: {n_divergences} divergences (max allowed {MCMC_MAX_DIVERGENCES}).")
    if n_divergences > MCMC_MAX_DIVERGENCES:
        # Diagnostic only: does NOT change the gate's decision below. Dumps the
        # trace to a distinctly-suffixed, clearly-non-final path and logs which
        # scalar hyperparameters differ between divergent and non-divergent
        # draws, so a human can diagnose *why* (e.g. rho pinned at a boundary)
        # instead of guessing before deciding on a real fix.
        diag_summary = summarize_divergent_draws(
            trace, ["rho", "sigma_spatial", "sigma_err", "beta_th", "beta_inc"]
        )
        logger.warning(f"DIAGNOSTIC (gate will still fail): divergent vs non-divergent draw stats: {diag_summary}")
        diag_path = PROCESSED_DIR / "national_unified_trace_DIAGNOSTIC_FAILED_GATE.nc"
        if diag_path.exists():
            diag_path.unlink(missing_ok=True)
        trace.to_netcdf(diag_path)
        logger.warning(f"DIAGNOSTIC trace (not a valid result, gate failed) saved to {diag_path}")
        raise RuntimeError(
            f"FATAL: {n_divergences} divergences exceeds the allowed maximum of "
            f"{MCMC_MAX_DIVERGENCES}. Refusing to save results as converged."
        )
    # round_to="none" keeps ess_bulk/ess_tail/r_hat as real floats. Without it,
    # this arviz version (1.2.0) returns display-rounded *strings* for every
    # numeric column, which silently breaks both the f"{max_r_hat:.4f}" format
    # below and the >= MCMC_MAX_RHAT comparison (string vs float).
    summary = az.summary(trace, round_to="none")
    if chains >= 2:
        max_r_hat = summary["r_hat"].max()
        logger.info(f"Convergence check: max r_hat={max_r_hat:.4f} (max allowed {MCMC_MAX_RHAT}).")
        if max_r_hat >= MCMC_MAX_RHAT:
            raise RuntimeError(
                f"FATAL: max r_hat={max_r_hat:.4f} exceeds the allowed maximum of "
                f"{MCMC_MAX_RHAT}. Refusing to save results as converged."
            )
    else:
        logger.warning(
            f"Only {chains} chain(s) sampled — r_hat is not meaningful with a single "
            "chain, so it was not gated (divergence check above still applies)."
        )

    logger.info("Computing Log Likelihood...")
    with unified_model:
        pm.compute_log_likelihood(trace)

    trace.attrs.update(_run_metadata(mode="pilot" if is_pilot else "final", draws=draws, tune=tune, chains=chains))

    suffix = "_pilot" if is_pilot else ""
    logger.info("Model fitted and converged. Saving Trace...")
    trace_path = PROCESSED_DIR / f"national_unified_trace{suffix}.nc"
    if trace_path.exists():
        trace_path.unlink(missing_ok=True) # Idempotency: Overwrite cleanly
    trace.to_netcdf(trace_path)

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

    # 6. Extract Output and Compute True Decoupled T*
    # We extract the posterior means
    b_inc_mean = trace.posterior['beta_inc'].mean().item()

    # Partial Residualization: We ONLY subtract behavioral rationing. We KEEP omega_star (unobserved physics)
    # T*_m = exp( log(y_m) - beta_inc * (Z - Z_ref) )
    # Let Z_ref = 0 (average income)
    T_star = np.exp(y_obs - b_inc_mean * income_z)

    msoa_stats = msoa_stats.assign(T_star_kwh=T_star)
    msoa_stats.to_csv(PROCESSED_DIR / f"msoa_unified_results{suffix}.csv", index=False)
    logger.info(f"Saved true empirically decoupled T* results (mode={'pilot' if is_pilot else 'final'}).")
    log_memory("Final Exit")

    return trace

if __name__ == "__main__":
    run_national_unified_model()

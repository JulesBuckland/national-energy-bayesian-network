from typing import cast

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import logging
import gc
from src.config.settings import (
    PROCESSED_DIR, HEATING_DEFICIT_FILE,
    MCMC_SAMPLES, MCMC_TUNE, RANDOM_SEED
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("NationalNonSpatialPilot")

def build_and_sample_pilot(
    df: pd.DataFrame, n_fit: int = 30000, draws: int = 1000
) -> pd.DataFrame:
    y_obs = np.log(df['empirical_gas_kwh'].values)
    theory_need_log = np.log(df['theoretical_gas_kwh'].values)
    income_score = cast(np.ndarray, df['income_dep_score'].values)
    
    # Global scaling for income
    i_mean, i_std = income_score.mean(), income_score.std()
    income_z = (income_score - i_mean) / i_std
    
    with pm.Model() as model:
        # Priors
        beta_theory = pm.Normal("beta_theory", mu=-0.3, sigma=0.1)
        beta_income = pm.Normal("beta_income", mu=0.0, sigma=0.5)
        
        # Likelihood (using MSOA means)
        # mu = physics_offset - global_rationing
        mu = theory_need_log + beta_theory + beta_income * income_z
        
        # Sigma for residual MSOA variance
        sigma_msoa = pm.HalfNormal("sigma_msoa", 0.5)
        
        y = pm.Normal("y", mu=mu, sigma=sigma_msoa, observed=y_obs)
        
        # Sampling with ADVI for speed
        approx = pm.fit(n=n_fit, method='advi', random_seed=RANDOM_SEED)
        trace = approx.sample(draws)
        
        summary = cast(pd.DataFrame, az.summary(trace, var_names=["beta_theory", "beta_income"]))
    return summary

def run_pilot_model() -> None:
    logger.info("--- STAGE 3.1: NATIONAL NON-SPATIAL PILOT (Global Estimation) ---")
    
    # 1. Load Data
    deficit_path = PROCESSED_DIR / HEATING_DEFICIT_FILE
    conf_path = PROCESSED_DIR / "msoa_confounders_national.csv"
    
    if not deficit_path.exists() or not conf_path.exists():
        logger.error("Required data files not found.")
        return

    deficit_data = pd.read_csv(deficit_path)
    conf_data = pd.read_csv(conf_path)
    
    # Merge
    df = deficit_data.merge(conf_data, left_on='msoa21cd', right_on='msoa_cd')
    
    logger.info(f"Running non-spatial model for {len(df)} MSOAs...")
    
    summary = build_and_sample_pilot(df)
    
    logger.info("\n" + str(summary))
    
    # Save results to processed dir for injection
    summary.to_csv(PROCESSED_DIR / "national_pilot_coefficients.csv")
    logger.info(f"Pilot coefficients saved to {PROCESSED_DIR / 'national_pilot_coefficients.csv'}")

if __name__ == "__main__":
    run_pilot_model()

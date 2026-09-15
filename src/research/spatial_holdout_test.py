import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
from pathlib import Path
import os
import sys
import logging
import gc
from sklearn.model_selection import GroupShuffleSplit

# Add root to path
sys.path.append(os.getcwd())
from src.config.settings import (
    PROCESSED_DIR,
    HEATING_DEFICIT_FILE,
    RANDOM_SEED,
    LOOKUP_PATH,
)

# Silence spam
logging.getLogger("pymc").setLevel(logging.ERROR)
logging.getLogger("arviz").setLevel(logging.ERROR)

def make_spatial_holdout(df: pd.DataFrame, lookup: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split whole LADs, rather than individual MSOAs, into train and test.

    A row-wise random split leaks neighbouring areas into both partitions and
    is not a spatial generalisation test.  LAD is the smallest geography
    available in the checked-in lookup that is shared by the model output and
    provides a deterministic, auditable group boundary.
    """
    required = {"msoa21cd", "ladnm"}
    missing = required - set(lookup.columns)
    if missing:
        raise ValueError(f"Spatial lookup is missing columns: {sorted(missing)}")
    groups = lookup[["msoa21cd", "ladnm"]].drop_duplicates("msoa21cd")
    merged = df.merge(groups, on="msoa21cd", how="left", validate="one_to_one")
    if merged["ladnm"].isna().any():
        raise ValueError("Spatial holdout lookup does not cover every model MSOA")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=RANDOM_SEED)
    train_idx, test_idx = next(splitter.split(merged, groups=merged["ladnm"]))
    train_df = merged.iloc[train_idx].copy()
    test_df = merged.iloc[test_idx].copy()
    if set(train_df["ladnm"]) & set(test_df["ladnm"]):
        raise AssertionError("Spatial holdout leaked an LAD across train and test")
    return train_df, test_df


def run_spatial_holdout():
    print("--- V10: LAD-grouped spatial generalization holdout test (10%) ---")
    
    deficit_path = PROCESSED_DIR / HEATING_DEFICIT_FILE
    conf_path = PROCESSED_DIR / "msoa_confounders_national.csv"
    
    df = pd.read_csv(deficit_path).merge(
        pd.read_csv(conf_path), left_on="msoa21cd", right_on="msoa_cd", validate="one_to_one"
    )
    lookup = pd.read_csv(LOOKUP_PATH, usecols=["msoa21cd", "ladnm"])
    train_df, test_df = make_spatial_holdout(df, lookup)
    
    print(f"Total MSOAs: {len(df)}")
    print(f"Training Set: {len(train_df)} MSOAs across {train_df['ladnm'].nunique()} LADs")
    print(f"Holdout Set: {len(test_df)} MSOAs across {test_df['ladnm'].nunique()} LADs")
    
    # Pre-process Training
    y_obs_train = np.log(train_df['empirical_gas_kwh'].values)
    theory_need_log_train = np.log(train_df['theoretical_gas_kwh'].values)
    income_score_train = train_df['income_dep_score'].values
    income_z_train = (income_score_train - income_score_train.mean()) / income_score_train.std()
    
    # Fit Model on Training Data (Reduced iterations for speed in this phase)
    print("\nFitting BYM2 model on 90% Training Set...")
    with pm.Model() as train_model:
        beta_theory = pm.Normal("beta_theory", mu=-0.3, sigma=0.1)
        beta_income = pm.Normal("beta_income", mu=0.0, sigma=0.5)
        sigma = pm.HalfNormal("sigma", 0.5)
        
        # We simplify the spatial component for the holdout test to global params 
        # to test generalization of the core decoupling logic
        mu = theory_need_log_train + beta_theory + beta_income * income_z_train
        y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs_train)
        
        trace = pm.fit(method='advi', n=10000, random_seed=RANDOM_SEED, progressbar=False).sample(1000)
    
    # Extract Global Posteriors
    beta_th_mean = trace.posterior['beta_theory'].mean().item()
    beta_inc_mean = trace.posterior['beta_income'].mean().item()
    
    print(f"Recovered Global Posteriors:")
    print(f"Beta_th: {beta_th_mean:.3f}")
    print(f"Beta_inc: {beta_inc_mean:.3f}")
    
    # Pre-process Holdout
    y_obs_test = np.log(test_df['empirical_gas_kwh'].values)
    theory_need_log_test = np.log(test_df['theoretical_gas_kwh'].values)
    income_score_test = test_df['income_dep_score'].values
    # Standardize test set using training moments
    income_z_test = (income_score_test - income_score_train.mean()) / income_score_train.std()
    
    # 2. Predict on Holdout
    print("\nPredicting on 10% Holdout Set...")
    mu_pred_test = theory_need_log_test + beta_th_mean + beta_inc_mean * income_z_test
    y_pred_test = np.exp(mu_pred_test) # Transform back to kWh
    y_true_test = test_df['empirical_gas_kwh'].values
    
    # 3. Evaluate
    mae = np.mean(np.abs(y_pred_test - y_true_test))
    mape = np.mean(np.abs(y_pred_test - y_true_test) / y_true_test) * 100
    rmse = np.sqrt(np.mean((y_pred_test - y_true_test)**2))
    
    # R-squared
    ss_res = np.sum((y_true_test - y_pred_test)**2)
    ss_tot = np.sum((y_true_test - np.mean(y_true_test))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    print("\n==========================================")
    print("Spatial Generalization Holdout Results")
    print("==========================================")
    print(f"Mean Absolute Error (MAE): {mae:.1f} kWh")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"Root Mean Squared Error (RMSE): {rmse:.1f} kWh")
    print(f"Out-of-Sample R-squared: {r2:.3f}")
    
    # Save Report
    report_path = PROCESSED_DIR / "holdout_results.txt"
    with open(report_path, "w") as f:
        f.write("V10 Methodological Audit: 10% LAD-Grouped Spatial Holdout Test\n")
        f.write("===============================================================\n\n")
        f.write(f"Training Set: {len(train_df)} MSOAs across {train_df['ladnm'].nunique()} LADs\n")
        f.write(f"Holdout Set: {len(test_df)} MSOAs across {test_df['ladnm'].nunique()} LADs\n")
        f.write("No LAD occurs in both partitions.\n\n")
        f.write("Global Posteriors (Trained on 90%):\n")
        f.write(f"  Beta_th: {beta_th_mean:.3f}\n")
        f.write(f"  Beta_inc: {beta_inc_mean:.3f}\n\n")
        f.write("Out-of-Sample Performance:\n")
        f.write(f"  MAE: {mae:.1f} kWh\n")
        f.write(f"  MAPE: {mape:.2f}%\n")
        f.write(f"  RMSE: {rmse:.1f} kWh\n")
        f.write(f"  R-squared: {r2:.3f}\n")
        
    print(f"\nReport saved to {report_path}")

if __name__ == "__main__":
    run_spatial_holdout()

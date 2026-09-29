import logging

import pandas as pd


def calculate_distribution_stats(df: pd.DataFrame, column_name: str) -> dict:
    if column_name not in df.columns:
        raise ValueError(f"Missing lineage column: {column_name}")
    stats = df[column_name].describe()
    return {
        "count": stats["count"],
        "mean": stats["mean"],
        "std": stats["std"],
        "min": stats["min"],
        "max": stats["max"],
    }


def log_distribution(
    df: pd.DataFrame, column_name: str, stage_name: str, logger: logging.Logger
) -> None:
    """
    Logs the statistical distribution of a key variable to ensure it has not drifted
    silently between pipeline stages.
    """
    try:
        stats = calculate_distribution_stats(df, column_name)
    except ValueError as e:
        logger.error(f"FATAL LINEAGE TRACKING: {column_name} missing at stage '{stage_name}'!")
        raise e

    logger.info(f"--- Data Lineage Check: [{stage_name}] | Var: {column_name} ---")
    logger.info(f"  Count: {stats['count']:,.0f}")
    logger.info(f"  Mean : {stats['mean']:,.2f}")
    logger.info(f"  Std  : {stats['std']:,.2f}")
    logger.info(f"  Min  : {stats['min']:,.2f}")
    logger.info(f"  Max  : {stats['max']:,.2f}")
    logger.info("----------------------------------------------------------")

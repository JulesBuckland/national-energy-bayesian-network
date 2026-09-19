"""Python-side reader for src/inference/inla/fit_inla.R's CSV/JSON output.

Adapts INLA's native output shape into what the existing results pipeline
(src/utils/compute_spatial_effect_summary.py, msoa_unified_results.csv
consumers) needs, rather than forcing it into the old NUTS-shaped
az.InferenceData structure (per tasks/inla_migration_plan.md's inventory
table).

Key simplification vs. the NUTS path: because the RSR projection is now
enforced AT FIT TIME via fit_inla.R's extraconstr mechanism (hand-verified
in tests/unit/test_inference_inla_rsr.R), the bym2 f-term's combined field
`b` (the first n rows of inla_random_effects.csv) already IS omega_star --
there is no separate post-hoc projection step to reconstruct here, unlike
compute_spatial_effect_summary.py's NUTS-era omega/omega_star rebuild from
raw rho/sigma_spatial/theta/phi posterior draws.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd


class InlaGateFailedError(RuntimeError):
    """Raised when fit_inla.R's quality gate rejected the fit -- mirrors
    model_unified.py's RuntimeError on the NUTS divergence/r_hat gate:
    refuse to treat an unconverged/unreliable fit as a valid result."""


def read_inla_fixed_effects(path: Path) -> pd.DataFrame:
    """Reads inla_fixed_effects.csv (INLA's summary.fixed, one row per
    fixed effect: '(Intercept)' = beta_th, 'income_z' = beta_inc)."""
    df = pd.read_csv(path, index_col=0)
    if "(Intercept)" not in df.index:
        raise ValueError("Expected an '(Intercept)' row (beta_th) in fixed effects.")
    if "income_z" not in df.index:
        raise ValueError("Expected an 'income_z' row (beta_inc) in fixed effects.")
    return df


def read_inla_random_effects(path: Path, n: int) -> dict:
    """Splits INLA's 2n-row bym2 random-effect summary into the combined
    field `b` (rows 1:n, == omega_star, RSR already applied at fit time)
    and the spatial-only field `u` (rows n+1:2n).

    Returns:
        dict with keys "b" and "u", each a DataFrame with columns
        [mean, sd, "0.025quant", "0.5quant", "0.975quant"] (INLA's own
        column names, passed through unchanged), n rows each.
    """
    df = pd.read_csv(path)
    if len(df) != 2 * n:
        raise ValueError(f"Expected {2 * n} rows (2n for bym2), got {len(df)}.")
    return {
        "b": df.iloc[:n].reset_index(drop=True),
        "u": df.iloc[n:].reset_index(drop=True),
    }


def read_inla_hyperpar_transformed(path: Path) -> dict:
    """Reads the single-row rho/sigma_spatial/sigma_err summary."""
    df = pd.read_csv(path)
    if len(df) != 1:
        raise ValueError(f"Expected exactly one row in hyperpar, got {len(df)}.")
    return df.iloc[0].to_dict()


def read_inla_cpo(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def read_inla_ic(path: Path) -> dict:
    df = pd.read_csv(path)
    if len(df) != 1:
        raise ValueError(f"Expected exactly one row in IC, got {len(df)}.")
    return df.iloc[0].to_dict()


def read_inla_metadata(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def build_spatial_effect_summary_from_inla(random_effects_b: pd.DataFrame, msoa_codes: np.ndarray) -> pd.DataFrame:
    """Produces the same schema as compute_spatial_effect_summary.py's
    NUTS-era output (msoa21cd, effect_mean, effect_sd, effect_ci_2.5,
    effect_ci_97.5), sourced directly from INLA's `b` field -- no
    omega_star reconstruction needed (see module docstring).

    Args:
        random_effects_b: the "b" DataFrame from read_inla_random_effects
            (n rows, INLA's own summary.random column names).
        msoa_codes: the n MSOA codes, in the same row order as the graph
            nodes used to build the model (id=1..n positional alignment,
            same convention compute_spatial_effect_summary.py's NUTS path
            already relies on -- see that module's docstring).
    """
    n = len(msoa_codes)
    if len(random_effects_b) != n:
        raise ValueError(
            f"random_effects_b has {len(random_effects_b)} rows but {n} MSOA codes were given."
        )
    return pd.DataFrame(
        {
            "msoa21cd": msoa_codes,
            "effect_mean": random_effects_b["mean"].values,
            "effect_sd": random_effects_b["sd"].values,
            "effect_ci_2.5": random_effects_b["0.025quant"].values,
            "effect_ci_97.5": random_effects_b["0.975quant"].values,
        }
    )


def load_inla_results(output_dir: Path, n: int) -> dict:
    """Effectful orchestrator: reads every file fit_inla.R writes for one
    run and returns them as a single dict, OR raises InlaGateFailedError
    if the quality gate rejected the fit (detected by the canonical
    metadata file being absent while a _DIAGNOSTIC_FAILED_GATE one exists,
    exactly mirroring fit_inla.R's own suffixing convention).
    """
    output_dir = Path(output_dir)
    metadata_path = output_dir / "inla_metadata.json"
    diagnostic_metadata_path = output_dir / "inla_metadata_DIAGNOSTIC_FAILED_GATE.json"

    if not metadata_path.exists():
        if diagnostic_metadata_path.exists():
            diag_meta = read_inla_metadata(diagnostic_metadata_path)
            raise InlaGateFailedError(
                f"fit_inla.R's quality gate rejected this fit: {diag_meta.get('gate_problems')}. "
                f"Diagnostic output is at {output_dir}, but no canonical result was produced."
            )
        raise FileNotFoundError(f"No inla_metadata.json found in {output_dir} -- was fit_inla.R run?")

    metadata = read_inla_metadata(metadata_path)
    fixed_effects = read_inla_fixed_effects(output_dir / "inla_fixed_effects.csv")
    random_effects = read_inla_random_effects(output_dir / "inla_random_effects.csv", n)
    hyperpar = read_inla_hyperpar_transformed(output_dir / "inla_hyperpar_transformed.csv")
    cpo = read_inla_cpo(output_dir / "inla_cpo.csv")
    ic = read_inla_ic(output_dir / "inla_ic.csv")

    return {
        "metadata": metadata,
        "fixed_effects": fixed_effects,
        "random_effects": random_effects,
        "hyperpar": hyperpar,
        "cpo": cpo,
        "ic": ic,
    }

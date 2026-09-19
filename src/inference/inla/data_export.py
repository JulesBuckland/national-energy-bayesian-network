"""Python -> R data handoff for the R-INLA inference boundary.

Mirrors src/inference/model_unified.py's pure-core/thin-effectful-shell
split: two pure frame-builders (independently testable with hand-computed
expected values) plus one effectful orchestrator that writes them to disk.

R-INLA graphs are 1-indexed; the rest of this codebase's node1/node2 edge
lists (built from libpysal.weights.Queen in model_unified.py) are 0-indexed
to match PyMC/numpy convention, so build_inla_edge_frame re-indexes here,
once, at the boundary, rather than pushing that concern into the R script.
"""
import numpy as np
import pandas as pd
from pathlib import Path


def build_inla_node_frame(
    T_var: np.ndarray,
    income_z: np.ndarray,
    theory_log: np.ndarray,
    y_obs: np.ndarray,
) -> pd.DataFrame:
    """One row per MSOA node, 1-indexed `id` matching R-INLA's graph convention.

    Args:
        T_var: within-MSOA log-variance used for the Jensen's correction offset.
        income_z: standardized income deprivation score per MSOA.
        theory_log: log of the physics-based theoretical baseline per MSOA.
        y_obs: log of the observed empirical thermal energy per MSOA.

    Returns:
        DataFrame with columns [id, T_var, income_z, theory_log, y_obs].
    """
    n = len(y_obs)
    if len(T_var) != n:
        raise ValueError(f"T_var length {len(T_var)} != y_obs length {n}")
    if len(income_z) != n:
        raise ValueError(f"income_z length {len(income_z)} != y_obs length {n}")
    if len(theory_log) != n:
        raise ValueError(f"theory_log length {len(theory_log)} != y_obs length {n}")
    return pd.DataFrame(
        {
            "id": np.arange(1, n + 1, dtype=np.int64),
            "T_var": np.asarray(T_var, dtype=float),
            "income_z": np.asarray(income_z, dtype=float),
            "theory_log": np.asarray(theory_log, dtype=float),
            "y_obs": np.asarray(y_obs, dtype=float),
        }
    )


def build_inla_edge_frame(node1: np.ndarray, node2: np.ndarray) -> pd.DataFrame:
    """Re-index a 0-indexed (node1, node2) edge list to R-INLA's 1-indexed convention.

    Args:
        node1, node2: 1D 0-indexed edge-list arrays, one entry per edge
            (same format model_unified.py builds from libpysal.weights.Queen).

    Returns:
        DataFrame with columns [node1, node2], both 1-indexed.
    """
    node1 = np.asarray(node1)
    node2 = np.asarray(node2)
    if node1.ndim != 1 or node2.ndim != 1:
        raise ValueError("Edge arrays must be 1D.")
    if node1.shape != node2.shape:
        raise ValueError("node1/node2 must have matching shapes.")
    return pd.DataFrame(
        {
            "node1": (node1 + 1).astype(np.int64),
            "node2": (node2 + 1).astype(np.int64),
        }
    )


def export_inla_inputs(
    node1: np.ndarray,
    node2: np.ndarray,
    T_var: np.ndarray,
    income_z: np.ndarray,
    theory_log: np.ndarray,
    y_obs: np.ndarray,
    output_dir: Path,
) -> dict:
    """Write the node and edge CSVs fit_inla.R reads, into output_dir.

    Returns:
        dict with keys "nodes_path"/"edges_path" (both pathlib.Path),
        for the caller to pass straight to the Rscript subprocess invocation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes_df = build_inla_node_frame(T_var, income_z, theory_log, y_obs)
    edges_df = build_inla_edge_frame(node1, node2)

    nodes_path = output_dir / "inla_nodes.csv"
    edges_path = output_dir / "inla_edges.csv"
    nodes_df.to_csv(nodes_path, index=False)
    edges_df.to_csv(edges_path, index=False)

    return {"nodes_path": nodes_path, "edges_path": edges_path}

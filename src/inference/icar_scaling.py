"""Riebler et al. (2016) / Simpson et al. (2017) BYM2 scaling factor for an
ICAR spatial prior, computed from the sparse graph structure alone (no dense
N x N matrix — feasible at national scale, ~6,853 nodes).

The ICAR precision matrix Q = D - A (degree diagonal minus adjacency) is
singular (rank N-1: the all-ones vector is its null space), so we perturb
the diagonal slightly to make it invertible, then take the geometric mean of
the diagonal of its inverse. Dividing an unscaled ICAR sample by
sqrt(scaling_factor) gives a spatial field whose typical (geometric-mean)
marginal variance is 1, making the BYM2 mixing weight `rho` interpretable
as the actual proportion of variance attributable to spatial clustering.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu


def compute_icar_scaling_factor(node1: np.ndarray, node2: np.ndarray, n_nodes: int) -> float:
    """Compute the BYM2 scaling factor for an ICAR prior on a graph given as
    an edge list (node1[k], node2[k]) pairs, 0-indexed, each edge listed once.

    Args:
        node1, node2: 1D integer arrays of equal length, one entry per edge.
        n_nodes: number of graph nodes (N).

    Returns:
        The geometric mean of the diagonal of the (jitter-perturbed)
        generalized inverse of the ICAR precision matrix.
    """
    if node1.shape != node2.shape:
        raise ValueError("node1 and node2 must have matching shapes.")
    if node1.ndim != 1:
        raise ValueError("Edge arrays must be 1D.")
    node1 = node1.astype(np.int64)
    node2 = node2.astype(np.int64)

    if len(node1) == 0:
        # No edges at all (e.g. a single-MSOA test fixture): there is no
        # spatial structure to scale against, so scaling is a no-op.
        return 1.0

    degree = np.zeros(n_nodes)
    np.add.at(degree, node1, 1)
    np.add.at(degree, node2, 1)

    rows = np.concatenate([node1, node2, np.arange(n_nodes)])
    cols = np.concatenate([node2, node1, np.arange(n_nodes)])
    data = np.concatenate([-np.ones(len(node1)), -np.ones(len(node2)), degree])
    Q = sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

    # Perturb the diagonal so Q is invertible (Q is singular by construction:
    # rank N-1). This is the standard practical trick used by Riebler et al.
    # and INLA's inla.scale.model — jitter scaled to the matrix's own size so
    # it is negligible relative to the true diagonal entries.
    jitter = Q.diagonal().max() * np.sqrt(np.finfo(float).eps)
    Q_pert = (Q + sp.eye(n_nodes) * jitter).tocsc()

    lu = splu(Q_pert)
    diag_inv = np.empty(n_nodes)
    e_i = np.zeros(n_nodes)
    for i in range(n_nodes):
        e_i[i] = 1.0
        diag_inv[i] = lu.solve(e_i)[i]
        e_i[i] = 0.0

    return float(np.exp(np.mean(np.log(diag_inv))))

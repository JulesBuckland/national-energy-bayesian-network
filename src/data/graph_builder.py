import numpy as np


def build_graph(num_nodes: int = 5) -> np.ndarray:
    """
    Builds a sparse edge-list for the ICAR spatial graph.
    Returns a numpy array of shape (E, 2) where E is the number of edges.
    """
    # For a 5-node pilot, we create a simple linear graph: 0-1-2-3-4
    # This simulates 5 buildings bordering each other in a line.
    edges = [[i, i + 1] for i in range(num_nodes - 1)]

    return np.array(edges, dtype=np.int32)

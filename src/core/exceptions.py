"""Domain-specific exceptions.

Each subclasses the built-in it replaced (ValueError / RuntimeError), so
existing ``except ValueError`` / ``except RuntimeError`` handlers and
``pytest.raises`` checks keep catching them.
"""


class DataValidationError(ValueError):
    """Input data is invalid, empty, or fails a pipeline integrity check."""


class ConvergenceError(RuntimeError):
    """An MCMC or INLA fit failed its convergence / quality gate."""


class ProvenanceError(ValueError):
    """Keys or lineage between independently-produced tables do not line up."""

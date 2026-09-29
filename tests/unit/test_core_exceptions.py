import pytest

from src.core.exceptions import ConvergenceError, DataValidationError, ProvenanceError
from src.inference.inla.read_results import InlaGateFailedError
from src.utils.provenance import assert_unique_keys
import pandas as pd


def test_domain_exceptions_remain_catchable_as_builtins():
    assert issubclass(DataValidationError, ValueError)
    assert issubclass(ProvenanceError, ValueError)
    assert issubclass(ConvergenceError, RuntimeError)
    assert issubclass(InlaGateFailedError, ConvergenceError)


def test_provenance_helpers_raise_provenance_error():
    with pytest.raises(ProvenanceError, match="duplicate"):
        assert_unique_keys(pd.DataFrame({"k": [1, 1]}), ["k"], label="t")

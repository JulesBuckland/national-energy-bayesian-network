import pandas as pd
import pytest

from src.utils.provenance import assert_keys_subset, assert_unique_keys


def test_assert_unique_keys_rejects_duplicate_key():
    frame = pd.DataFrame({"msoa_cd": ["E1", "E1"]})
    with pytest.raises(ValueError, match="duplicate"):
        assert_unique_keys(frame, ["msoa_cd"], label="fixture")


def test_assert_keys_subset_rejects_missing_key():
    required = pd.DataFrame({"msoa_cd": ["E1", "E2"]})
    available = pd.DataFrame({"msoa_cd": ["E1"]})
    with pytest.raises(ValueError, match="absent"):
        assert_keys_subset(
            required,
            available,
            "msoa_cd",
            required_label="required",
            available_label="available",
        )

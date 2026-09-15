import pandas as pd

from src.research.spatial_holdout_test import make_spatial_holdout


def test_make_spatial_holdout_keeps_lads_whole():
    model = pd.DataFrame({
        "msoa21cd": ["m1", "m2", "m3", "m4", "m5", "m6"],
        "value": [1, 2, 3, 4, 5, 6],
    })
    lookup = pd.DataFrame({
        "msoa21cd": ["m1", "m2", "m3", "m4", "m5", "m6"],
        "ladnm": ["A", "A", "B", "B", "C", "C"],
    })

    train, test = make_spatial_holdout(model, lookup)

    assert set(train["msoa21cd"]) | set(test["msoa21cd"]) == set(model["msoa21cd"])
    assert not set(train["msoa21cd"]) & set(test["msoa21cd"])
    assert not set(train["ladnm"]) & set(test["ladnm"])

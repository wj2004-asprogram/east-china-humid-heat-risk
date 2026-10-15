"""Unit tests for heat risk classification and area fractions."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.risk import CATEGORY_LABELS, classify_heat_risk, compute_area_fractions, find_hottest_risk_day


@pytest.mark.parametrize(
    "hi_c, expected",
    [
        (20.0, 0),
        (26.69, 0),
        (26.7, 1),
        (32.2, 2),
        (40.59, 2),
        (40.6, 3),
        (54.4, 4),
        (60.0, 4),
    ],
)
def test_classify_thresholds(hi_c, expected):
    cat = classify_heat_risk(xr.DataArray([hi_c]))
    assert int(cat[0]) == expected


def _category_grid(values):
    """Categories on a (date, latitude, longitude) grid at 0°N and 60°N, two longitudes."""
    return xr.DataArray(
        np.asarray(values, dtype=np.int8),
        dims=("date", "latitude", "longitude"),
        coords={
            "date": pd.date_range("2022-07-01", periods=len(values)),
            "latitude": [0.0, 60.0],
            "longitude": [118.0, 119.0],
        },
    )


def test_area_fractions_use_cos_lat_weights():
    # Row at 0°N (weight 1) is Danger, row at 60°N (weight 0.5) is No Risk.
    cat = _category_grid([[[3, 3], [0, 0]]])
    df = compute_area_fractions(cat, "latitude", "longitude", "date")

    assert list(df.columns) == ["date", *CATEGORY_LABELS.values()]
    assert df.loc[0, "date"] == "2022-07-01"
    assert df.loc[0, "Danger"] == pytest.approx(100.0 * 2 / 3)
    assert df.loc[0, "No Risk"] == pytest.approx(100.0 * 1 / 3)
    assert df.loc[0, list(CATEGORY_LABELS.values())].sum() == pytest.approx(100.0)


def test_find_hottest_risk_day():
    cat = _category_grid([
        [[3, 0], [0, 0]],  # day 1: some Danger
        [[4, 3], [3, 0]],  # day 2: more Danger + Extreme Danger
        [[2, 2], [2, 2]],  # day 3: none
    ])
    date_str, frac, idx = find_hottest_risk_day(cat, "date", "latitude", "longitude")
    assert (date_str, idx) == ("2022-07-02", 1)
    assert frac == pytest.approx((1 + 1 + 0.5) / 3)

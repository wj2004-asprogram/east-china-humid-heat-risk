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


def _land_mask(values):
    """Land mask on the same 2 x 2 grid as _category_grid (True = land)."""
    return xr.DataArray(
        np.asarray(values, dtype=bool),
        dims=("latitude", "longitude"),
        coords={"latitude": [0.0, 60.0], "longitude": [118.0, 119.0]},
    )


def test_area_fractions_count_land_only():
    # The Danger cell at (0°N, 119°E) is sea and must not be counted.
    # Land weights: (0°N, 118°E) = 1, (60°N, 118°E) = 0.5, (60°N, 119°E) = 0.5.
    cat = _category_grid([[[2, 3], [0, 0]]])
    land = _land_mask([[True, False], [True, True]])
    df = compute_area_fractions(cat, "latitude", "longitude", "date", land)

    assert df.loc[0, "Extreme Caution"] == pytest.approx(100.0 * 1 / 2)
    assert df.loc[0, "No Risk"] == pytest.approx(100.0 * 1 / 2)
    assert df.loc[0, "Danger"] == pytest.approx(0.0)
    assert df.loc[0, list(CATEGORY_LABELS.values())].sum() == pytest.approx(100.0)


def test_hottest_risk_day_ignores_sea():
    cat = _category_grid([
        [[0, 4], [0, 4]],  # day 1: Extreme Danger over the sea column only
        [[3, 0], [0, 0]],  # day 2: Danger on land
    ])
    land = _land_mask([[True, False], [True, False]])
    date_str, frac, idx = find_hottest_risk_day(cat, "date", "latitude", "longitude", land)
    assert (date_str, idx) == ("2022-07-02", 1)
    assert frac == pytest.approx(1 / 1.5)


def test_empty_land_mask_is_an_error():
    cat = _category_grid([[[0, 0], [0, 0]]])
    with pytest.raises(ValueError):
        compute_area_fractions(cat, "latitude", "longitude", "date", _land_mask([[False, False], [False, False]]))

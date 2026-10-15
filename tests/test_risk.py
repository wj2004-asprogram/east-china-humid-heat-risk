"""Unit tests for heat risk classification and area fractions."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.risk import (
    CATEGORY_LABELS,
    MISSING_CATEGORY,
    VALID_COVERAGE_LABEL,
    classify_heat_risk,
    compute_area_fractions,
    find_hottest_risk_day,
)

CATEGORIES = list(CATEGORY_LABELS.values())


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
        (np.nan, -1),  # missing data is never "No Risk"
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

    assert list(df.columns) == ["date", *CATEGORIES, VALID_COVERAGE_LABEL]
    assert df.loc[0, "date"] == "2022-07-01"
    # No missing data: the whole area is valid, fractions as without the Missing rule.
    assert df.loc[0, VALID_COVERAGE_LABEL] == 100.0
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


# ---------------------------------------------------------------------------
# Missing data (daily maximum heat index NaN)
# ---------------------------------------------------------------------------

def _hi_grid(values):
    """Daily maximum heat index (°C) on the same grid as _category_grid."""
    return xr.DataArray(
        np.asarray(values, dtype=float),
        dims=("date", "latitude", "longitude"),
        coords={
            "date": pd.date_range("2022-07-01", periods=len(values)),
            "latitude": [0.0, 60.0],
            "longitude": [118.0, 119.0],
        },
    )


def test_all_missing_day():
    nan = np.nan
    hi = _hi_grid([
        [[45.0, 30.0], [20.0, 20.0]],  # day 1: complete
        [[nan, nan], [nan, nan]],      # day 2: no data at all
    ])
    cat = classify_heat_risk(hi)
    assert (cat.isel(date=1) == MISSING_CATEGORY).all()
    assert cat.dtype == np.int8

    df = compute_area_fractions(cat, "latitude", "longitude", "date")
    # Day 2: no "No Risk" (or any category), zero coverage.
    assert df.loc[1, CATEGORIES].isna().all()
    assert df.loc[1, VALID_COVERAGE_LABEL] == 0.0
    # Day 1 is unaffected.
    assert df.loc[0, VALID_COVERAGE_LABEL] == 100.0
    assert df.loc[0, "Danger"] == pytest.approx(100.0 * 1 / 3)


def test_partially_missing_day():
    # Weights: 1 at 0°N, 0.5 at 60°N; all four cells are land (total 3).
    # (0°N, 118°E) Danger (45 °C), (0°N, 119°E) missing,
    # (60°N, 118°E) No Risk (20 °C), (60°N, 119°E) Caution (30 °C).
    # Valid area = 1 + 0.5 + 0.5 = 2 -> coverage 2/3.
    hi = _hi_grid([[[45.0, np.nan], [20.0, 30.0]]])
    df = compute_area_fractions(classify_heat_risk(hi), "latitude", "longitude", "date")

    assert df.loc[0, VALID_COVERAGE_LABEL] == pytest.approx(100.0 * 2 / 3)
    assert df.loc[0, "Danger"] == pytest.approx(100.0 * 1 / 2)
    assert df.loc[0, "No Risk"] == pytest.approx(100.0 * 0.5 / 2)
    assert df.loc[0, "Caution"] == pytest.approx(100.0 * 0.5 / 2)
    assert df.loc[0, "Extreme Caution"] == 0.0
    assert df.loc[0, "Extreme Danger"] == 0.0
    assert df.loc[0, CATEGORIES].sum() == pytest.approx(100.0)


def test_missing_sea_cell_does_not_reduce_land_coverage():
    # The missing cell (0°N, 119°E) is sea, so the whole land area is valid.
    hi = _hi_grid([[[45.0, np.nan], [20.0, 30.0]]])
    land = _land_mask([[True, False], [True, True]])
    df = compute_area_fractions(classify_heat_risk(hi), "latitude", "longitude", "date", land)

    assert df.loc[0, VALID_COVERAGE_LABEL] == pytest.approx(100.0)
    assert df.loc[0, "Danger"] == pytest.approx(100.0 * 1 / 2)


def test_hottest_risk_day_ignores_missing_cells():
    cat = _category_grid([
        # day 1: Danger on 1 of 2 valid weight units (the 0°N, 119°E cell is missing)
        [[3, MISSING_CATEGORY], [0, 0]],
        # day 2: Danger on 1 of 3 weight units, nothing missing
        [[3, 0], [0, 0]],
    ])
    date_str, frac, idx = find_hottest_risk_day(cat, "date", "latitude", "longitude")
    # Shares of the valid area: day 1 = 1/2, day 2 = 1/3.
    assert (date_str, idx) == ("2022-07-01", 0)
    assert frac == pytest.approx(1 / 2)


def test_hottest_risk_day_skips_days_without_data():
    m = MISSING_CATEGORY
    cat = _category_grid([[[m, m], [m, m]], [[0, 3], [0, 0]]])
    date_str, frac, idx = find_hottest_risk_day(cat, "date", "latitude", "longitude")
    assert (date_str, idx, frac) == ("2022-07-02", 1, pytest.approx(1 / 3))

    with pytest.raises(ValueError):
        find_hottest_risk_day(cat.isel(date=[0]), "date", "latitude", "longitude")

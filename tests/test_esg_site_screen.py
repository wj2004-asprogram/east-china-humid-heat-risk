"""Small independent examples for site-screening failure and boundary cases."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tools.esg_site_screen import nearest_cell, screen, summer_metrics


def test_threshold_inclusive_and_runs_do_not_bridge_winters():
    dates = pd.date_range("2020-06-01", "2020-08-31").append(
        pd.date_range("2021-06-01", "2021-08-31"))
    values = np.zeros(184)
    values[90:94] = 40.6
    metrics = summer_metrics(dates, values, 40.6)
    assert [row["days_ge_threshold"] for row in metrics] == [2, 2]
    assert [row["max_consecutive_days"] for row in metrics] == [2, 2]


@pytest.mark.parametrize("mode", ["nan", "missing_date", "infinity"])
def test_incomplete_summers_do_not_become_zero_risk(mode):
    dates = pd.date_range("2020-06-01", "2020-08-31")
    values = np.full(92, 50.0)
    if mode == "missing_date":
        dates, values = dates[:-1], values[:-1]
    else:
        values[0] = np.nan if mode == "nan" else np.inf
    row = summer_metrics(dates, values, 40.6)[0]
    assert row["valid_days"] == 91
    assert not row["complete_summer"]
    assert row["days_ge_threshold"] is None
    assert row["max_consecutive_days"] is None


def test_missing_entire_year_is_retained():
    dates = pd.date_range("2020-06-01", "2020-08-31").append(
        pd.date_range("2022-06-01", "2022-08-31"))
    rows = summer_metrics(dates, np.zeros(184), 40.6)
    assert rows[1]["year"] == 2021
    assert rows[1]["valid_days"] == 0
    assert rows[1]["days_ge_threshold"] is None


@pytest.mark.parametrize("latitude,longitude", [(np.nan, 120), (31, np.inf), (20, 120), (31, 124)])
def test_no_coordinate_clipping_or_extrapolation(latitude, longitude):
    with pytest.raises(ValueError):
        nearest_cell([32, 31, 30], [119, 120, 121], latitude, longitude)


def test_nearest_grid_and_distance():
    assert nearest_cell([32, 31, 30], [119, 120, 121], 31, 120) == (1, 1, 0.0)
    assert nearest_cell([32, 31, 30], [119, 120, 121], 31.1, 120.1)[:2] == (1, 1)


def test_sites_share_a_grid_and_sea_is_not_relocated():
    dates = pd.date_range("2020-06-01", "2020-08-31")
    hi = xr.DataArray(np.full((92, 2, 2), 50.0), dims=("date", "latitude", "longitude"),
                      coords={"date": dates, "latitude": [31, 30], "longitude": [120, 121]})
    land = xr.DataArray([[1, 0], [1, 1]], dims=("latitude", "longitude"),
                        coords={"latitude": [31, 30], "longitude": [120, 121]})
    sites = pd.DataFrame([
        {"site_id": str(i), "name": "test", "latitude": lat, "longitude": lon,
         "source_url": "https://example.org", "coordinate_method": "test"}
        for i, (lat, lon) in enumerate([(31, 120), (30.9, 120.1), (31, 121), (20, 120)])])
    summary, annual = screen(sites, hi, land, 40.6)
    assert summary.screening_status.tolist() == ["ok", "ok", "non_land_grid", "invalid_coordinate"]
    assert summary.grid_id[0] == summary.grid_id[1]
    assert summary.mean_days_ge_threshold_per_complete_summer[:2].tolist() == [92, 92]
    assert len(annual) == 2
    with pytest.raises(ValueError, match="match exactly"):
        screen(sites, hi, land.assign_coords(latitude=[31.1, 30]), 40.6)


def test_duplicate_dates_rejected():
    with pytest.raises(ValueError, match="unique"):
        summer_metrics(pd.to_datetime(["2020-06-01", "2020-06-01"]), [50, 50], 40.6)

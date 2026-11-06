"""Full registry tests using deterministic small-grid hourly summers."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.probability_complete_features import complete_daily_night, assemble_common_year, neighbor_statistics


def fixture():
    times = pd.date_range("2000-05-31T16:00", periods=92*24, freq="h")
    temp = 20+np.arange(92*24)[:, None, None]/1000+np.zeros((92*24, 2, 2))
    hourly = xr.Dataset({"t2m": (("valid_time", "latitude", "longitude"), temp+273.15, {"units": "K"}),
                         "d2m": (("valid_time", "latitude", "longitude"), temp-3+273.15, {"units": "K"})},
                        coords={"valid_time": times, "latitude": [30., 30.25], "longitude": [115., 115.25]})
    registry = pd.DataFrame({"latitude": [30., 30., 30.25, 30.25],
                             "longitude": [115., 115.25, 115., 115.25], "spatial_role": ["train"]*4})
    sources = [[j for j in range(4) if j != i] for i in range(4)]
    return hourly, registry, sources


def test_full_registry_window_counts_and_night_cooling():
    hourly, registry, sources = fixture()
    daily = complete_daily_night(hourly, 2000)
    assert daily.tdmin_c[0, 0, 0].item() == pytest.approx(17)
    assert daily.night_cooling20to07_c[1, 0, 0].item() == pytest.approx(-.011)
    assert np.isnan(daily.night_tmean_c[0, 0, 0])
    payload, report, feature_registry = assemble_common_year(daily, registry, sources)
    assert payload["X1"].shape == payload["X2"].shape == (336, 117)
    assert report["common_samples"] == 336  # June 1 night does not remove June 9.
    groups = feature_registry["groups"]
    assert sum(g in ("C", "T") for g in groups) == 33
    assert sum(g in ("C", "T", "M") for g in groups) == 87
    names = feature_registry["names"]
    assert payload["X2"][0, names.index("tmean_c_anchor_minus0")] == pytest.approx(20+(6*24+11.5)/1000)
    assert payload["X2"][0, names.index("tmean_c_last7_mean")] == pytest.approx(20+(3*24+11.5)/1000)


def test_missing_interior_night_hour_invalidates_cooling_and_common_rows():
    hourly, registry, sources = fixture()
    hourly.t2m.values[6*24+22, 0, 0] = np.nan
    daily = complete_daily_night(hourly, 2000)
    assert np.isnan(daily.night_cooling20to07_c[7, 0, 0])
    _, report, _ = assemble_common_year(daily, registry, sources)
    assert report["common_samples"] < 336
    with pytest.raises(ValueError, match="Missing UTC hour"):
        complete_daily_night(hourly.isel(valid_time=slice(1, None)), 2000)
    with pytest.raises(ValueError, match="restricted"):
        complete_daily_night(hourly, 2020)


def test_neighbor_fixed_membership_no_missing_renormalization():
    values = np.array([[1., 2., 3., 4.], [1., np.nan, 3., 4.]])
    mean, std = neighbor_statistics(values, [[1, 2, 3], [], [], []])
    assert mean[0, 0] == 3
    assert std[0, 0] == pytest.approx(np.sqrt(2/3))
    assert np.isnan(mean[1, 0])
    assert np.isnan(mean[:, 1:]).all()


def test_labels_use_daily_hi_before_feature_storage_cast():
    hourly, registry, sources = fixture()
    daily = complete_daily_night(hourly, 2000)
    daily["himax_c"] = xr.full_like(daily.himax_c, 40.6000001, dtype=np.float64)
    payload, _, feature_registry = assemble_common_year(daily, registry, sources)
    assert (payload["y"] == 1).all()
    # The stored feature can lie slightly below 40.6 after float32 conversion;
    # labels must not be rebuilt from this rounded matrix column.
    index = feature_registry["names"].index("himax_c_anchor_minus0")
    assert float(payload["X1"][0, index]) < 40.6

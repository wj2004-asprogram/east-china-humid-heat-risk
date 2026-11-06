"""Hand-computable calendar and causal-window feature checks."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.probability_features import (
    daily_features, completed_night_features, assemble_pilot_samples,
)


def hourly(start, periods, temperature=20., dewpoint=15.):
    times = pd.date_range(start, periods=periods, freq="h")
    t = np.broadcast_to(np.asarray(temperature), (periods,)).copy()
    td = np.broadcast_to(np.asarray(dewpoint), (periods,)).copy()
    return xr.Dataset({"t2m": (("valid_time", "latitude", "longitude"), (t+273.15)[:, None, None], {"units": "K"}),
                       "d2m": (("valid_time", "latitude", "longitude"), (td+273.15)[:, None, None], {"units": "K"})},
                      coords={"valid_time": times, "latitude": [35.], "longitude": [115.]})


def test_daily_bjt_cross_month_hand_values_and_true_mean_dewpoint():
    # July 1 starts June 30 16 UTC. T=Td gives exactly saturated RH.
    data = hourly("2000-06-30T16:00", 48, np.arange(48.), np.arange(48.))
    daily = daily_features(data, "2000-07-01", "2000-07-02")
    np.testing.assert_allclose(daily.tmean_c.values.ravel(), [11.5, 35.5])
    np.testing.assert_allclose(daily.tmin_c.values.ravel(), [0, 24])
    np.testing.assert_allclose(daily.tmax_c.values.ravel(), [23, 47])
    np.testing.assert_allclose(daily.tdmean_c.values.ravel(), [11.5, 35.5])
    np.testing.assert_allclose(daily.tdmax_c.values.ravel(), [23, 47])
    np.testing.assert_allclose(daily.rhmean_pct, 100.)
    assert daily.tdmean_c.attrs["units"] == "degC"


def test_daily_missing_values_propagate_and_missing_timestamp_raises():
    data = hourly("2000-05-31T16:00", 24)
    data.d2m.values[4, 0, 0] = np.nan
    daily = daily_features(data, "2000-06-01", "2000-06-01")
    assert daily.tmean_c.item() == pytest.approx(20.)
    for name in ("tdmean_c", "tdmax_c", "rhmean_pct", "himax_c"):
        assert np.isnan(daily[name].item())
    with pytest.raises(ValueError, match="No complete hourly"):
        daily_features(data.isel(valid_time=np.arange(23)), "2000-06-01", "2000-06-01")


def test_night_has_twelve_hours_excludes_end_and_target_morning():
    # D=June 10: D-2 20 BJT = June 8 12 UTC; D-1 08 BJT = June 9 00 UTC.
    data = hourly("2000-06-08T12:00", 48, 999., 999.)
    data.t2m.values[:12, 0, 0] = np.arange(10., 22.)+273.15
    data.d2m.values[:12, 0, 0] = np.arange(10., 22.)+273.15
    night = completed_night_features(data, "2000-06-10", 1)
    assert night.night_tmean_c.item() == pytest.approx(15.5)
    assert night.night_tmin_c.item() == pytest.approx(10.)
    assert night.night_tdmean_c.item() == pytest.approx(15.5)
    assert night.night_rhmean_pct.item() == pytest.approx(100.)
    assert night.attrs["source_last_hour_utc"] == "2000-06-08T23:00:00Z"
    assert night.attrs["source_end_utc_exclusive"] == "2000-06-09T00:00:00Z"


def test_d2_night_ignores_d1_and_nan_never_skipped():
    data = hourly("2000-06-07T12:00", 60)
    data.t2m.values[24:, 0, 0] = np.nan
    assert completed_night_features(data, "2000-06-10", 2).night_tmean_c.item() == pytest.approx(20.)
    assert np.isnan(completed_night_features(data, "2000-06-10", 1).night_tmean_c.item())
    data.d2m.values[0, 0, 0] = np.nan
    assert np.isnan(completed_night_features(data, "2000-06-10", 2).night_tdmean_c.item())
    with pytest.raises(ValueError, match="Incomplete 12-hour"):
        completed_night_features(data.isel(valid_time=slice(1, None)), "2000-06-10", 2)


@pytest.mark.parametrize("change", ["duplicate", "off_hour", "units"])
def test_reject_bad_hourly_contract(change):
    data = hourly("2000-06-07T12:00", 12)
    if change == "duplicate":
        data = xr.concat([data, data.isel(valid_time=[0])], dim="valid_time")
    elif change == "off_hour":
        data = data.assign_coords(valid_time=data.valid_time+np.timedelta64(30, "m"))
    else:
        data.t2m.attrs["units"] = "degC"
    with pytest.raises(ValueError):
        completed_night_features(data, "2000-06-10", 2)


def test_assembly_uses_real_offsets_and_shared_id_without_target_labels():
    # Each local day has its day number as its temperature.
    data = hourly("2000-05-31T16:00", 15*24, np.repeat(np.arange(1., 16.), 24), 0.)
    daily = daily_features(data, "2000-06-01", "2000-06-15")
    rows = assemble_pilot_samples(daily, data, ["2000-06-09"], [(35., 115.)]).set_index("cutoff_lag_days")
    assert rows.loc[1, "sample_id"] == rows.loc[2, "sample_id"]
    assert rows.loc[1, "tmean_c_day_lag1"] == pytest.approx(8.)
    assert np.isnan(rows.loc[2, "tmean_c_day_lag1"])
    assert rows.loc[2, "tmean_c_day_lag2"] == pytest.approx(7.)
    assert rows.loc[2, "tmean_c_day_lag8"] == pytest.approx(1.)
    assert rows.loc[2, "source_end_utc_exclusive"] == "2000-06-07T16:00:00Z"
    assert rows.loc[2, "night_end_utc_exclusive"] == "2000-06-07T00:00:00Z"
    assert rows.finite_feature_inputs.all()
    assert not any("label" in name for name in rows.columns)
    with pytest.raises(ValueError, match="training years"):
        assemble_pilot_samples(daily, data, ["2020-06-09"], [(35., 115.)])

"""Tests for local calendar days (Beijing time = UTC+8)."""

from datetime import date

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.time_utils import local_daily


REF = pd.Timestamp("2022-06-30 00:00")


def _hours_since_ref(utc_time) -> float:
    return (pd.Timestamp(utc_time) - REF) / pd.Timedelta(hours=1)


def _hourly(start_utc: str, end_utc: str) -> xr.DataArray:
    """Hourly series whose value is its own UTC time in hours since REF."""
    times = pd.date_range(start_utc, end_utc, freq="h")
    return xr.DataArray([_hours_since_ref(t) for t in times], dims="time", coords={"time": times})


def _dates(da) -> list:
    return list(pd.DatetimeIndex(da["time"].values).strftime("%Y-%m-%d"))


def test_beijing_day_runs_from_16_utc_to_15_utc():
    # As downloaded for Beijing days: 16:00 UTC on 2022-06-30 to 23:00 UTC on 2022-07-02.
    da = _hourly("2022-06-30 16:00", "2022-07-02 23:00")
    period = (date(2022, 7, 1), date(2022, 7, 2))
    daily_max = local_daily(da, "time", 8, "max", *period)
    daily_mean = local_daily(da, "time", 8, "mean", *period)

    # Labelled with Beijing dates; the partial Beijing day 2022-07-03
    # (16:00-23:00 UTC on 2022-07-02) is dropped.
    assert _dates(daily_max) == ["2022-07-01", "2022-07-02"]

    # Beijing 2022-07-01 = 16:00 UTC on 2022-06-30 ... 15:00 UTC on 2022-07-01.
    first = daily_max.sel(time=pd.Timestamp("2022-07-01")).item()
    assert first == _hours_since_ref("2022-07-01 15:00")
    first_mean = daily_mean.sel(time=pd.Timestamp("2022-07-01")).item()
    assert first_mean == pytest.approx(
        (_hours_since_ref("2022-06-30 16:00") + _hours_since_ref("2022-07-01 15:00")) / 2
    )
    last = daily_max.sel(time=pd.Timestamp("2022-07-02")).item()
    assert last == _hours_since_ref("2022-07-02 15:00")


def test_utc_offset_zero_gives_utc_days():
    da = _hourly("2022-07-01 00:00", "2022-07-02 23:00")
    daily_max = local_daily(da, "time", 0, "max", date(2022, 7, 1), date(2022, 7, 2))
    assert _dates(daily_max) == ["2022-07-01", "2022-07-02"]
    assert daily_max.values.tolist() == [
        _hours_since_ref("2022-07-01 23:00"), _hours_since_ref("2022-07-02 23:00"),
    ]


def test_west_of_utc_uses_next_utc_day():
    # UTC-5: local day D = 05:00 UTC on D ... 04:00 UTC on D+1.
    da = _hourly("2022-07-01 00:00", "2022-07-02 04:00")
    daily_max = local_daily(da, "time", -5, "max", date(2022, 7, 1), date(2022, 7, 1))
    assert _dates(daily_max) == ["2022-07-01"]
    assert daily_max.item() == _hours_since_ref("2022-07-02 04:00")


def test_boundary_hours_belong_to_the_right_beijing_day():
    # A spike at 15:00 UTC on 2022-07-01 is the last hour of Beijing day
    # 2022-07-01 (23:00 Beijing); a spike at 16:00 UTC on 2022-07-01 is the
    # first hour of Beijing day 2022-07-02 (00:00 Beijing).
    times = pd.date_range("2022-06-30 16:00", "2022-07-02 15:00", freq="h")
    da = xr.DataArray([0.0] * len(times), dims="time", coords={"time": times})
    da.loc["2022-07-01 15:00"] = 1.0
    da.loc["2022-07-01 16:00"] = 2.0
    da.loc["2022-06-30 16:00"] = 3.0  # first hour of the download = first hour of 2022-07-01

    daily_max = local_daily(da, "time", 8, "max", date(2022, 7, 1), date(2022, 7, 2))
    daily_sum = local_daily(da, "time", 8, "mean", date(2022, 7, 1), date(2022, 7, 2)) * 24
    assert daily_max.values.tolist() == [3.0, 2.0]
    assert daily_sum.values.tolist() == pytest.approx([1.0 + 3.0, 2.0])


def test_beijing_day_across_a_month_boundary():
    # Beijing 2022-08-01 = 16:00 UTC on 2022-07-31 ... 15:00 UTC on 2022-08-01.
    da = _hourly("2022-07-31 16:00", "2022-08-02 23:00")
    daily_max = local_daily(da, "time", 8, "max", date(2022, 8, 1), date(2022, 8, 2))
    daily_mean = local_daily(da, "time", 8, "mean", date(2022, 8, 1), date(2022, 8, 2))

    assert _dates(daily_max) == ["2022-08-01", "2022-08-02"]
    assert daily_max.values[0] == _hours_since_ref("2022-08-01 15:00")
    assert daily_mean.values[0] == pytest.approx(
        (_hours_since_ref("2022-07-31 16:00") + _hours_since_ref("2022-08-01 15:00")) / 2
    )


def test_utc_raw_file_with_beijing_days_is_an_error():
    # A raw file downloaded for UTC days lacks 16:00-23:00 UTC of the day
    # before the period, so Beijing day 2022-07-01 is incomplete: stop
    # instead of silently dropping it.
    da = _hourly("2022-07-01 00:00", "2022-07-02 23:00")
    with pytest.raises(ValueError, match="2022-07-01"):
        local_daily(da, "time", 8, "max", date(2022, 7, 1), date(2022, 7, 2))


def test_beijing_raw_file_with_utc_days_gives_utc_results():
    # The other way round is harmless: the extra 8 hours before the period
    # form an incomplete UTC day, which is dropped.
    with_edge = local_daily(_hourly("2022-06-30 16:00", "2022-07-02 23:00"),
                            "time", 0, "max", date(2022, 7, 1), date(2022, 7, 2))
    without = local_daily(_hourly("2022-07-01 00:00", "2022-07-02 23:00"),
                          "time", 0, "max", date(2022, 7, 1), date(2022, 7, 2))
    xr.testing.assert_identical(with_edge, without)


def test_no_complete_day_is_an_error():
    da = _hourly("2022-07-01 00:00", "2022-07-01 12:00")
    with pytest.raises(ValueError):
        local_daily(da, "time", 8, "max", date(2022, 7, 1), date(2022, 7, 1))


def test_missing_hour_makes_the_day_missing():
    # skipna=False: one NaN hour in Beijing day 2022-07-01 makes that day's
    # max and mean NaN; 2022-07-02 is unaffected.
    da = _hourly("2022-06-30 16:00", "2022-07-02 23:00")
    da.loc["2022-07-01 03:00"] = np.nan
    period = (date(2022, 7, 1), date(2022, 7, 2))
    daily_max = local_daily(da, "time", 8, "max", *period)
    daily_mean = local_daily(da, "time", 8, "mean", *period)

    assert np.isnan(daily_max.values[0]) and np.isnan(daily_mean.values[0])
    assert daily_max.values[1] == _hours_since_ref("2022-07-02 15:00")


# ---------------------------------------------------------------------------
# Complete days: exactly the 24 hourly time steps, each once
# ---------------------------------------------------------------------------

def _hourly_at(times) -> xr.DataArray:
    times = pd.DatetimeIndex(times)
    return xr.DataArray([_hours_since_ref(t) for t in times], dims="time", coords={"time": times})


def test_missing_hour_plus_duplicated_hour_is_an_error():
    # Beijing day 2022-07-01 with 03:00 UTC missing and 04:00 UTC twice still
    # has 24 time steps, but it is not complete.
    times = pd.date_range("2022-06-30 16:00", "2022-07-02 23:00", freq="h")
    times = times.drop(pd.Timestamp("2022-07-01 03:00")).append(pd.DatetimeIndex(["2022-07-01 04:00"]))
    with pytest.raises(ValueError, match="Duplicate.*2022-07-01 04:00"):
        local_daily(_hourly_at(times), "time", 8, "max", date(2022, 7, 1), date(2022, 7, 2))


def test_duplicate_outside_the_period_is_an_error_too():
    # Never de-duplicate silently, even in an edge day that would be dropped.
    times = pd.date_range("2022-06-30 16:00", "2022-07-02 23:00", freq="h")
    times = times.append(pd.DatetimeIndex(["2022-07-02 20:00"]))
    with pytest.raises(ValueError, match="2022-07-02 20:00"):
        local_daily(_hourly_at(times), "time", 8, "max", date(2022, 7, 1), date(2022, 7, 2))


def test_non_hourly_time_step_is_an_error():
    times = pd.date_range("2022-06-30 16:00", "2022-07-02 23:00", freq="h")
    times = times.append(pd.DatetimeIndex(["2022-07-01 12:30"]))
    with pytest.raises(ValueError, match="2022-07-01 12:30"):
        local_daily(_hourly_at(times), "time", 8, "max", date(2022, 7, 1), date(2022, 7, 2))


def test_missing_hour_inside_the_period_is_an_error():
    # 23 hours on Beijing day 2022-07-02 (10:00 UTC on 07-02 missing): stop.
    times = pd.date_range("2022-06-30 16:00", "2022-07-02 23:00", freq="h")
    times = times.drop(pd.Timestamp("2022-07-02 10:00"))
    with pytest.raises(ValueError, match="2022-07-02"):
        local_daily(_hourly_at(times), "time", 8, "max", date(2022, 7, 1), date(2022, 7, 2))

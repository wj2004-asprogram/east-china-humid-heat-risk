"""Tests for the daily maximum heat index climatology (phase 3 baseline)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.climatology import by_year_and_day, climatology_for_years, noleap_day_of_year


YEARS = [2003, 2004, 2005, 2006]  # 2004 is a leap year


def _daily(years=YEARS, months=(6, 7, 8)) -> xr.DataArray:
    """
    Daily values on (date, latitude, longitude) for the given months of
    each year. Value = year + day of month / 100 + cell offset, so every
    year, date and cell is recognisable.
    """
    dates = pd.DatetimeIndex([
        d for y in years
        for d in pd.date_range(f"{y}-{months[0]:02d}-01", pd.Timestamp(f"{y}-{months[-1]:02d}-01") + pd.offsets.MonthEnd(0))
    ])
    cell = np.array([[0.0, 0.1], [0.2, 0.3]])
    values = (dates.year.values + dates.day.values / 100.0)[:, None, None] + cell
    return xr.DataArray(
        values, dims=("date", "latitude", "longitude"),
        coords={"date": dates, "latitude": [30.0, 29.0], "longitude": [115.0, 116.0]},
        attrs={"units": "Celsius"},
    )


def test_noleap_day_of_year():
    dates = pd.DatetimeIndex(["2003-06-01", "2004-06-01", "2004-03-01", "2004-02-28", "2003-12-31", "2004-12-31"])
    assert noleap_day_of_year(dates).tolist() == [152, 152, 60, 59, 365, 365]
    with pytest.raises(ValueError):
        noleap_day_of_year(pd.DatetimeIndex(["2004-02-29"]))


def test_by_year_and_day_aligns_calendar_dates_across_leap_years():
    by_year = by_year_and_day(_daily())

    assert by_year.dims == ("year", "day_of_year", "latitude", "longitude")
    assert by_year["year"].values.tolist() == YEARS
    assert by_year["day_of_year"].values.tolist() == list(range(152, 244))  # 1 June to 31 August
    assert by_year["month_day"].values[[0, 29, 30, -1]].tolist() == ["06-01", "06-30", "07-01", "08-31"]
    assert not by_year.isnull().any()

    # 1 June is the same day of year in 2003 and in leap year 2004.
    assert by_year.sel(year=2004, day_of_year=152, latitude=30.0, longitude=115.0).item() == pytest.approx(2004.01)
    assert by_year.sel(year=2003, day_of_year=243, latitude=29.0, longitude=116.0).item() == pytest.approx(2003.31 + 0.3)


def test_29_february_is_left_out():
    by_year = by_year_and_day(_daily(years=[2003, 2004], months=(2, 3)))
    assert by_year["day_of_year"].values.tolist() == list(range(32, 91))  # 1 February to 31 March
    # 1 March 2004 (leap year) has the same day of year as 1 March 2003.
    assert by_year.sel(year=2004, day_of_year=60, latitude=30.0, longitude=115.0).item() == pytest.approx(2004.01)


def test_climatology_uses_only_the_given_years():
    by_year = by_year_and_day(_daily())
    train = [2003, 2004]
    clim = climatology_for_years(by_year, train)

    assert clim.dims == ("day_of_year", "latitude", "longitude")
    # Mean of 2003 and 2004 = 2003.5 + day / 100 + cell.
    assert clim.sel(day_of_year=152, latitude=30.0, longitude=115.0).item() == pytest.approx(2003.51)
    assert clim.sel(day_of_year=243, latitude=29.0, longitude=116.0).item() == pytest.approx(2003.81 + 0.3)
    assert clim.attrs["climatology_years"] == "2003, 2004"

    # No leakage: changing the data of the other (test) years changes nothing.
    changed = by_year.copy()
    changed.loc[{"year": [2005, 2006]}] = 99.0
    xr.testing.assert_identical(climatology_for_years(changed, train), clim)

    # Years need not be contiguous; order and repeats do not matter.
    other = climatology_for_years(by_year, [2006, 2003, 2006])
    assert other.attrs["climatology_years"] == "2003, 2006"
    assert other.sel(day_of_year=152, latitude=30.0, longitude=115.0).item() == pytest.approx(2004.51)


@pytest.mark.parametrize("years", [[], [2002, 2003], [2010]])
def test_climatology_rejects_unknown_or_no_years(years):
    with pytest.raises(ValueError):
        climatology_for_years(by_year_and_day(_daily()), years)


def test_running_mean_is_cut_at_the_season_edges():
    by_year = by_year_and_day(_daily())
    plain = climatology_for_years(by_year, YEARS)
    smooth = climatology_for_years(by_year, YEARS, window_days=5)
    cell = {"latitude": 30.0, "longitude": 115.0}

    # Inside the season: mean over 5 consecutive days of year.
    np.testing.assert_allclose(
        smooth.sel(day_of_year=180, **cell), plain.sel(day_of_year=slice(178, 182), **cell).mean(),
    )
    # 1 June: only 1-3 June (no May data); 31 August: only 29-31 August.
    np.testing.assert_allclose(smooth.sel(day_of_year=152, **cell), plain.sel(day_of_year=slice(152, 154), **cell).mean())
    np.testing.assert_allclose(smooth.sel(day_of_year=243, **cell), plain.sel(day_of_year=slice(241, 243), **cell).mean())
    assert not smooth.isnull().any()
    assert smooth.attrs["window_days"] == 5

    for bad in (0, 4, -3):
        with pytest.raises(ValueError):
            climatology_for_years(by_year, YEARS, window_days=bad)


def test_missing_values_stay_missing():
    # One missing cell-day in a training year makes that day's climatology
    # missing for that cell (no averaging over partial data); with a
    # running mean, every window that contains it.
    daily = _daily()
    daily.loc[{"date": "2004-07-15", "latitude": 30.0, "longitude": 115.0}] = np.nan
    by_year = by_year_and_day(daily)
    doy = 196  # 15 July

    clim = climatology_for_years(by_year, [2003, 2004])
    assert np.isnan(clim.sel(day_of_year=doy, latitude=30.0, longitude=115.0))
    assert int(clim.isnull().sum()) == 1
    # Not used when 2004 is not a training year.
    assert not climatology_for_years(by_year, [2003, 2005]).isnull().any()

    smooth = climatology_for_years(by_year, [2003, 2004], window_days=3)
    missing = smooth.sel(latitude=30.0, longitude=115.0).isnull()
    assert smooth["day_of_year"].values[missing.values].tolist() == [doy - 1, doy, doy + 1]

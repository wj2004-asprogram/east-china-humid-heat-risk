"""
climatology.py

Daily climatology of the daily maximum heat index, per grid cell and day
of year: the climatology baseline of phase 3 (next-day heat risk).

No climatology over all years is stored, so test years cannot leak into
the baseline. The step stores the daily maximum heat index of every year
on a (year, day_of_year) grid, and `climatology_for_years` computes the
climatology of the years it is given, e.g. the training years only:

    from src.climatology import climatology_for_years, load_by_year
    by_year = load_by_year(cfg.paths.climatology_nc)
    baseline = climatology_for_years(by_year, years=range(2000, 2018))

Day of year is counted in a 365-day calendar (29 February left out), so
a calendar date has the same day of year in every year: 1 June = 152,
also in leap years.
"""

import numpy as np
import pandas as pd
import xarray as xr

from src.config import RunConfig
from src.era5_utils import detect_date_dim, load_dataset, require_var


VAR = "daily_max_heat_index_c"

# A non-leap year, to label days of year with their calendar date.
NON_LEAP_YEAR = 2001


def noleap_day_of_year(dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Day of year in a 365-day calendar: the same for a calendar date in
    every year (1 March = 60, 1 June = 152). 29 February is not allowed.
    """
    if ((dates.month == 2) & (dates.day == 29)).any():
        raise ValueError("29 February has no day of year in the 365-day calendar.")
    return np.asarray(dates.dayofyear - (dates.is_leap_year & (dates.month > 2)), dtype=int)


def by_year_and_day(daily: xr.DataArray, date_dim: str = "date") -> xr.DataArray:
    """
    Rearrange daily values from (date, ...) to (year, day_of_year, ...).

    Days of year that a year does not have are NaN. 29 February is left
    out (with a message). A `month_day` coordinate ("06-01") labels each
    day of year.
    """
    dates = pd.DatetimeIndex(daily[date_dim].values)
    feb29 = np.asarray((dates.month == 2) & (dates.day == 29))
    if feb29.any():
        print(f"Leaving out 29 February ({int(feb29.sum())} days): not in the 365-day calendar.")
        daily = daily.isel({date_dim: ~feb29})
        dates = dates[~feb29]

    day_of_year = noleap_day_of_year(dates)
    values = daily.transpose(date_dim, ...)
    other_dims = values.dims[1:]

    years = np.unique(dates.year)
    days = np.unique(day_of_year)
    data = np.full((len(years), len(days)) + values.shape[1:], np.nan,
                   dtype=np.result_type(values.dtype, np.float32))
    data[np.searchsorted(years, dates.year), np.searchsorted(days, day_of_year)] = values.values

    month_day = (pd.Timestamp(NON_LEAP_YEAR, 1, 1) + pd.to_timedelta(days - 1, unit="D")).strftime("%m-%d")
    return xr.DataArray(
        data,
        dims=("year", "day_of_year") + other_dims,
        coords={
            "year": years,
            "day_of_year": days,
            "month_day": ("day_of_year", np.asarray(month_day)),
            **{dim: values[dim] for dim in other_dims},
        },
        attrs=dict(daily.attrs),
    )


def _running_mean(clim: xr.DataArray, window_days: int) -> xr.DataArray:
    """
    Centred running mean over `window_days` days of year. The window is
    cut at the first and last day of the data (it never reaches across the
    gap between seasons); it is NaN wherever a value in the window is NaN.
    """
    windows = clim.rolling(day_of_year=window_days, center=True).construct("window")
    # True where a window position falls on a day of the data, False in
    # the padding beyond its first and last day.
    inside = (
        xr.ones_like(clim["day_of_year"], dtype=float)
        .rolling(day_of_year=window_days, center=True).construct("window")
        .notnull()
    )
    has_missing = (windows.isnull() & inside).any("window")
    return windows.where(inside).mean("window").where(~has_missing)


def climatology_for_years(by_year: xr.DataArray, years, window_days: int = 1) -> xr.DataArray:
    """
    Climatology of the daily maximum heat index per day of year and grid
    cell, from the given years only (e.g. the training years of phase 3).

    Parameters
    ----------
    by_year : xr.DataArray
        Daily maximum heat index (°C) on (year, day_of_year, ...), as
        written by this step (see load_by_year).
    years : iterable of int
        Years to average over. Every year must be in `by_year`.
    window_days : int
        1 (default): plain mean per day of year. An odd number > 1: then
        also a centred running mean over that many days of year, cut at the
        edges of the season (e.g. 1 June uses 1-8 June for 15 days).

    Returns
    -------
    xr.DataArray on (day_of_year, ...). NaN where any value used is
    missing (no averaging over partial data, as for the daily values).
    """
    years = sorted({int(y) for y in years})
    if not years:
        raise ValueError("No years given for the climatology.")
    unknown = sorted(set(years) - set(by_year["year"].values.tolist()))
    if unknown:
        raise ValueError(f"Years not in the data: {unknown}.")
    if window_days < 1 or window_days % 2 == 0:
        raise ValueError(f"window_days must be an odd number >= 1, got {window_days}.")

    clim = by_year.sel(year=years).mean("year", skipna=False)
    if window_days > 1:
        if not np.all(np.diff(by_year["day_of_year"].values) == 1):
            raise ValueError("A running mean needs consecutive days of year.")
        clim = _running_mean(clim, window_days)

    clim.attrs = {
        "units": by_year.attrs.get("units", "Celsius"),
        "long_name": "climatology of the daily maximum heat index",
        "climatology_years": ", ".join(str(y) for y in years),
        "window_days": window_days,
        "description": (
            f"Mean over {len(years)} years per day of year (365-day calendar)"
            + (f", then a centred {window_days}-day running mean cut at the season edges"
               if window_days > 1 else "")
            + "; NaN where any value used is missing"
        ),
    }
    return clim


def load_by_year(path) -> xr.DataArray:
    """Read the (year, day_of_year) daily maximum heat index written by this step."""
    return load_dataset(path)[VAR]


def run(cfg: RunConfig) -> None:
    input_path = cfg.paths.humid_heat_nc
    output_path = cfg.paths.climatology_nc
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input file:  {input_path}")
    ds = load_dataset(input_path)
    daily = require_var(ds, VAR)
    by_year = by_year_and_day(daily, detect_date_dim(daily))
    by_year.attrs["description"] = (
        "Daily maximum heat index of each year by day of year (365-day calendar), "
        "input of the climatology baseline; NaN = missing"
    )

    ds_out = xr.Dataset(
        {VAR: by_year},
        attrs={
            "title": "Daily maximum heat index by year and day of year (climatology input)",
            "period": cfg.period_label,
            "source": "Based on ERA5 hourly single-level reanalysis",
            "created_with": "src/climatology.py",
            "daily_aggregation": ds.attrs.get("daily_aggregation", ""),
            "day_of_year": "365-day calendar: 29 February left out; 1 June = 152 in every year",
            "usage": (
                "Climatology of chosen years only (e.g. the phase 3 training years): "
                "src.climatology.climatology_for_years(load_by_year(path), years). "
                "No climatology over all years is stored, so test years cannot leak "
                "into the baseline."
            ),
        },
    )
    ds_out.to_netcdf(output_path, encoding={VAR: {"zlib": True, "complevel": 4}})

    years = by_year["year"].values
    month_day = by_year["month_day"].values
    print(f"Years:        {years[0]} to {years[-1]} ({len(years)})")
    print(f"Days of year: {by_year['day_of_year'].values[0]} to {by_year['day_of_year'].values[-1]} "
          f"({month_day[0]} to {month_day[-1]}, {len(month_day)} days)")
    print(f"Missing values: {int(by_year.isnull().sum())}")
    print(f"Output file: {output_path}")
    print("\nDone.")

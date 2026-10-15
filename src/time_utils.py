"""
time_utils.py

Aggregate hourly ERA5 data to local calendar days.

ERA5 timestamps are UTC. With a UTC offset of +8 hours (Beijing time),
local day D runs from 16:00 UTC on D-1 to 15:00 UTC on D.
"""

from datetime import date

import numpy as np
import pandas as pd


HOURS_PER_DAY = 24


def local_daily(obj, time_coord: str, utc_offset_hours: int, how: str,
                start_date: date, end_date: date):
    """
    Aggregate hourly data (UTC timestamps) to local calendar days.

    Each timestamp is shifted by `utc_offset_hours` and the values are
    grouped by local calendar date. A missing value (NaN) in any hour makes
    that cell-day missing (skipna=False): no maxima or means over partial
    days. Days without all 24 hourly values
    (e.g. the partial days at the edges of a download) are dropped, and
    only days from `start_date` to `end_date` are kept. A ValueError is
    raised if any day of the period is not complete.

    Parameters
    ----------
    obj : xr.Dataset or xr.DataArray
        Hourly data with UTC timestamps along `time_coord`.
    time_coord : str
        Name of the hourly time coordinate.
    utc_offset_hours : int
        Local time minus UTC, e.g. 8 for Beijing time.
    how : str
        Daily reduction: "max" or "mean".
    start_date, end_date : datetime.date
        First and last local day of the period.

    Returns
    -------
    Same type as `obj`, with one value per complete local day along
    `time_coord`, labelled with the local date (00:00).
    """
    local_time = obj[time_coord].values + np.timedelta64(utc_offset_hours, "h")
    local = obj.assign_coords({time_coord: local_time})

    daily = getattr(local.resample({time_coord: "1D"}), how)(skipna=False, keep_attrs=True)

    # Keep complete local days only.
    hours_per_day = pd.Series(1, index=pd.DatetimeIndex(local_time)).resample("1D").count()
    incomplete = hours_per_day.index[hours_per_day != HOURS_PER_DAY]
    if len(incomplete) > 0:
        print(f"Dropping incomplete local days (UTC{utc_offset_hours:+d}, < {HOURS_PER_DAY} hours): "
              + ", ".join(d.strftime("%Y-%m-%d") for d in incomplete))
    complete = hours_per_day.index[hours_per_day == HOURS_PER_DAY]
    daily = daily.sel({time_coord: complete})

    # Keep the configured period only.
    daily = daily.sel({time_coord: slice(start_date.isoformat(), end_date.isoformat())})

    # Every day of the period must be complete. A missing first day usually
    # means a raw file downloaded for UTC days was given to a config with a
    # UTC offset: stop rather than silently drop the day.
    expected = pd.date_range(start_date, end_date, freq="D")
    missing = expected.difference(pd.DatetimeIndex(daily[time_coord].values))
    if len(missing) > 0:
        raise ValueError(
            f"No complete hourly data for local day(s) (UTC{utc_offset_hours:+d}) "
            + ", ".join(d.strftime("%Y-%m-%d") for d in missing)
            + ". The raw file must cover every hour of these days in UTC; a raw file "
            "downloaded for UTC days cannot be used with a non-zero utc_offset_hours."
        )

    daily[time_coord].attrs["long_name"] = f"local calendar day (UTC{utc_offset_hours:+d})"
    return daily

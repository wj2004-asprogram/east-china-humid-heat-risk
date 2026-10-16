"""
plot_utils.py

Helpers shared by the daily time-series plots, so that a multi-year run
(e.g. 25 summers) stays readable.
"""

import pandas as pd
import xarray as xr


# Longer series are drawn as thin lines without markers.
MAX_DAYS_WITH_MARKERS = 62


def break_at_gaps(series: xr.DataArray, time_dim: str) -> xr.DataArray:
    """
    Daily series with NaN on the days missing between its first and last
    day (the gaps between seasons), so that a plotted line breaks there
    instead of joining the end of one summer to the start of the next.
    """
    days = pd.DatetimeIndex(series[time_dim].values)
    every_day = pd.date_range(days.min(), days.max(), freq="D")
    if len(every_day) == len(days):
        return series
    return series.reindex({time_dim: every_day})


def line_style(n_days: int) -> dict:
    """Marker and line width for a daily series of `n_days` days."""
    if n_days <= MAX_DAYS_WITH_MARKERS:
        return {"marker": "o", "linewidth": 1.5}
    return {"marker": None, "linewidth": 0.8}

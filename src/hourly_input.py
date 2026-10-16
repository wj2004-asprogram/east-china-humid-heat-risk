"""
hourly_input.py

Read the raw hourly ERA5 data block by block, so that a multi-year run
never holds more than about two months of hourly data in memory, and
put the daily results of the blocks back together.

A block is a run of local days processed together (RunConfig.month_blocks):
one calendar month of a multi-year config, the whole period of a
single-period config. The hourly data of a block of a multi-year config
cover exactly the UTC hours of its local days. For Beijing time (UTC+8):
  - June:  16:00 UTC on 31 May (edge file) to 15:00 UTC on 30 June;
  - July:  16:00 UTC on 30 June (June file) to 15:00 UTC on 31 July;
  - August: 16:00 UTC on 31 July to 15:00 UTC on 31 August.
The days between two seasons belong to no block, so they are never
treated as missing data.
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterator

import pandas as pd
import xarray as xr

from src.config import RunConfig
from src.era5_utils import detect_time_coord, load_dataset


@dataclass(frozen=True)
class HourlyBlock:
    """Hourly raw data (UTC timestamps) for the local days first_day ... last_day."""

    data: xr.Dataset
    first_day: date
    last_day: date


def utc_window(first_day: date, last_day: date, utc_offset_hours: int) -> tuple:
    """
    First and last UTC hour of the local days first_day ... last_day, e.g.
    16:00 UTC on 31 May and 15:00 UTC on 30 June for Beijing-time June.
    """
    start = pd.Timestamp(first_day) - pd.Timedelta(hours=utc_offset_hours)
    end = pd.Timestamp(last_day) + pd.Timedelta(hours=23 - utc_offset_hours)
    return start, end


def iter_hourly_blocks(cfg: RunConfig) -> Iterator[HourlyBlock]:
    """
    Yield the hourly raw data block by block, in time order.

    Single-period config: one block, the raw file as downloaded (the
    period plus the edge hours; local_daily drops the partial edge days).

    Multi-year config: one block per calendar month, read from the raw
    files that overlap the block's UTC window (at most the month's file
    and the previous month's file or an edge file), cut to that window.
    Files are read once and kept only while a block still needs them.
    The raw files must share one grid (concatenation with join="exact").
    """
    if not cfg.is_multi_year:
        yield HourlyBlock(load_dataset(cfg.paths.raw_era5), cfg.start_date, cfg.end_date)
        return

    # First and last UTC hour of every raw file (days and hours are sorted).
    spans = [
        (raw_file,
         datetime(raw_file.year, raw_file.month, raw_file.days[0], raw_file.hours[0]),
         datetime(raw_file.year, raw_file.month, raw_file.days[-1], raw_file.hours[-1]))
        for raw_file in cfg.raw_files
    ]

    loaded = {}  # path -> Dataset, for the files of the current block only
    for first_day, last_day in cfg.month_blocks:
        start, end = utc_window(first_day, last_day, cfg.utc_offset_hours)
        needed = sorted(
            ((first, raw_file) for raw_file, first, last in spans if first <= end and last >= start),
            key=lambda item: item[0],
        )
        loaded = {
            raw_file.path: loaded[raw_file.path] if raw_file.path in loaded else load_dataset(raw_file.path)
            for _, raw_file in needed
        }
        parts = list(loaded.values())

        time_coord = detect_time_coord(parts[0])
        hourly = parts[0] if len(parts) == 1 else xr.concat(
            parts, dim=time_coord, data_vars="all", coords="minimal", compat="equals", join="exact",
        )
        # A copy, so that a step converting units never changes a file
        # kept for the next block.
        hourly = hourly.sel({time_coord: slice(start, end)}).copy()
        yield HourlyBlock(hourly, first_day, last_day)


def concat_days(parts: list, date_dim: str, cfg: RunConfig):
    """
    Concatenate the daily results of the blocks along `date_dim` and check
    that they hold every local day of the run once, in order.
    """
    daily = parts[0] if len(parts) == 1 else xr.concat(
        parts, dim=date_dim, data_vars="all", coords="minimal", compat="equals", join="exact",
    )
    days = pd.DatetimeIndex(daily[date_dim].values)
    if not days.equals(cfg.local_days):
        raise ValueError(
            f"Daily results do not match the local days of {cfg.period_label}: "
            f"got {len(days)} days ({days.min()} to {days.max()}), expected {len(cfg.local_days)}."
        )
    return daily

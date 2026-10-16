"""
plot_utils.py

Daily time series of multi-year runs (e.g. 25 summers): the seasons are
drawn side by side on a compressed x axis. On a calendar axis the months
between two summers would take most of the width.

x is the position of a day in the run's list of local days
(RunConfig.local_days), plus SEASON_GAP_DAYS of blank space before each
later season. Each season is drawn as its own line, so no line joins two
seasons; a light vertical line marks each season start, and the year is
labelled there. Single-period runs keep their
calendar date axis and do not use these helpers.
"""

import calendar
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import RunConfig


# Blank x units (days) between two seasons.
SEASON_GAP_DAYS = 10

# At most this many year labels; with more seasons only every 2nd, 5th
# or 10th season start is labelled.
MAX_YEAR_LABELS = 13
YEAR_LABEL_STEPS = (1, 2, 5, 10)

# Multi-year time series: wider figure (inches), thin lines without markers.
MULTI_YEAR_FIG_WIDTH = 14
MULTI_YEAR_LINE = {"linestyle": "-", "linewidth": 0.8}
SEPARATOR_COLOR = "0.85"


@dataclass(frozen=True)
class SeasonAxis:
    """Compressed x axis of a multi-year run."""

    days: pd.DatetimeIndex  # every local day of the run, in order
    x: np.ndarray  # x position of each day
    season: np.ndarray  # index of the season of each day (0, 1, ...)
    starts: np.ndarray  # x position of the first day of each season
    years: tuple  # year of each season
    label: str  # x axis label, e.g. "Year (June–August, UTC+8 calendar days)"


def season_axis(cfg: RunConfig) -> SeasonAxis:
    """The compressed x axis for the local days of a multi-year config."""
    days = cfg.local_days
    season = np.concatenate([
        np.full((last - first).days + 1, i) for i, (first, last) in enumerate(cfg.seasons)
    ])
    x = np.arange(len(days)) + SEASON_GAP_DAYS * season
    first_days = np.r_[0, np.flatnonzero(np.diff(season)) + 1]

    first_month, last_month = (calendar.month_name[m] for m in (cfg.months[0], cfg.months[-1]))
    months = first_month if first_month == last_month else f"{first_month}–{last_month}"
    return SeasonAxis(
        days=days,
        x=x,
        season=season,
        starts=x[first_days],
        years=tuple(int(year) for year in days[first_days].year),
        label=f"Year ({months}, {cfg.day_definition} calendar days)",
    )


def plot_seasons(ax, axis: SeasonAxis, dates, values, label=None, **style) -> list:
    """
    Plot a daily series of a multi-year run on the compressed axis, one
    line per season. `dates` must be the run's local days (one value each).
    Only the first season's line carries `label` (for the legend).
    """
    if not pd.DatetimeIndex(np.asarray(dates)).equals(axis.days):
        raise ValueError("The series must hold one value for each local day of the run, in order.")
    values = np.asarray(values)
    lines = []
    for i in range(len(axis.starts)):
        in_season = axis.season == i
        lines += ax.plot(axis.x[in_season], values[in_season],
                         label=label if i == 0 else "_nolegend_", **style)
    return lines


def format_season_axis(ax, axis: SeasonAxis) -> None:
    """Year ticks at season starts (thinned out if many), a separator at each later season start, x limits."""
    # At the season start, so a separator and an x grid line never sit
    # side by side; behind the data (zorder 0).
    for start in axis.starts[1:]:
        ax.axvline(start, color=SEPARATOR_COLOR, linewidth=0.8, zorder=0)

    n_seasons = len(axis.starts)
    step = next((s for s in YEAR_LABEL_STEPS if -(-n_seasons // s) <= MAX_YEAR_LABELS), YEAR_LABEL_STEPS[-1])
    ax.set_xticks(axis.starts[::step], [str(year) for year in axis.years[::step]])
    ax.set_xlim(axis.x[0] - SEASON_GAP_DAYS / 2, axis.x[-1] + SEASON_GAP_DAYS / 2)

"""Tests for the compressed season axis of multi-year time series."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.plot_utils import (  # noqa: E402
    SEASON_GAP_DAYS,
    format_season_axis,
    plot_seasons,
    season_axis,
)


CONFIG = """
name: axis
period:
  years: {first: 2003, last: 2005}
  months: [6, 7, 8]
region: {north: 36, west: 112, south: 25, east: 123}
cities:
  - {name: Nanjing,  lat: 32.06, lon: 118.79, color: "#d62728"}
utc_offset_hours: 8
land_sea_mask_threshold: 0.5
file_tag: axis
figures_dir: figures/axis
"""
SUMMER_DAYS = 92


@pytest.fixture
def cfg(tmp_path):
    config_file = tmp_path / "axis.yaml"
    config_file.write_text(CONFIG, encoding="utf-8")
    return load_config(config_file, project_root=tmp_path)


def _season_ranges(axis):
    """(first x, last x) of each season."""
    return [(axis.x[axis.season == i].min(), axis.x[axis.season == i].max()) for i in range(len(axis.starts))]


def test_days_are_contiguous_within_a_season_and_seasons_separated(cfg):
    axis = season_axis(cfg)

    assert len(axis.x) == len(cfg.local_days) == 3 * SUMMER_DAYS
    steps = np.diff(axis.x)
    new_season = np.diff(axis.season) == 1
    assert (steps[~new_season] == 1).all()  # one x unit per day within a season
    assert (steps[new_season] == 1 + SEASON_GAP_DAYS).all()  # gap between seasons
    assert new_season.sum() == 2

    assert axis.starts.tolist() == [0, SUMMER_DAYS + SEASON_GAP_DAYS, 2 * (SUMMER_DAYS + SEASON_GAP_DAYS)]
    assert axis.years == (2003, 2004, 2005)
    assert [str(d.date()) for d in axis.days[np.searchsorted(axis.x, axis.starts)]] == [
        "2003-06-01", "2004-06-01", "2005-06-01",
    ]
    assert axis.label == "Year (June–August, UTC+8 calendar days)"


def test_no_line_joins_two_seasons(cfg):
    axis = season_axis(cfg)
    fig, ax = plt.subplots()
    values = np.arange(len(cfg.local_days), dtype=float)
    lines = plot_seasons(ax, axis, cfg.local_days, values, label="Nanjing", color="red")

    assert len(lines) == len(ax.lines) == 3  # one line per season
    for line, (first, last) in zip(lines, _season_ranges(axis)):
        x = line.get_xdata()
        assert x.min() == first and x.max() == last  # within one season only
        assert (np.diff(x) == 1).all()
    # The legend shows each series once.
    assert [line.get_label() for line in lines] == ["Nanjing", "_nolegend_", "_nolegend_"]
    plt.close(fig)


def test_ticks_and_separators_on_season_starts(cfg):
    axis = season_axis(cfg)
    fig, ax = plt.subplots()
    plot_seasons(ax, axis, cfg.local_days, np.zeros(len(cfg.local_days)))
    format_season_axis(ax, axis)

    assert ax.get_xticks().tolist() == axis.starts.tolist()
    assert [t.get_text() for t in ax.get_xticklabels()] == ["2003", "2004", "2005"]
    # A separator at each later season start, after the previous season's
    # last day (seasons do not touch).
    separators = [line.get_xdata()[0] for line in ax.lines[3:]]  # after the 3 season lines
    assert separators == axis.starts[1:].tolist()
    ranges = _season_ranges(axis)
    for x, (_, previous_last) in zip(separators, ranges[:-1]):
        assert x - previous_last == 1 + SEASON_GAP_DAYS
    plt.close(fig)


def test_year_labels_are_thinned_out_for_25_summers():
    cfg = load_config(PROJECT_ROOT / "configs" / "jja_2000_2024_bjt.yaml")
    axis = season_axis(cfg)
    fig, ax = plt.subplots()
    format_season_axis(ax, axis)

    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == [str(y) for y in range(2000, 2025, 2)]  # every 2nd summer: 13 labels
    assert set(ax.get_xticks()) <= set(axis.starts.tolist())  # ticks on season starts only
    plt.close(fig)


def test_series_must_match_the_local_days(cfg):
    axis = season_axis(cfg)
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="local day"):
        plot_seasons(ax, axis, cfg.local_days[1:], np.zeros(len(cfg.local_days) - 1))
    plt.close(fig)

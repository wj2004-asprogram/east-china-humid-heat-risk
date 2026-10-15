"""Tests for the heat risk maps."""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

from src.plot_heat_risk_maps import color_limits  # noqa: E402


def test_color_limits_use_valid_land_cells():
    nan = np.nan
    values = np.array([[30.0, 60.0], [35.0, nan]])
    land = np.array([[True, False], [True, True]])  # 60 °C is over the sea
    assert color_limits(values, land) == (30.0, 35.0)


def test_color_limits_fall_back_to_all_valid_cells():
    nan = np.nan
    values = np.array([[nan, 60.0], [nan, 20.0]])
    land = np.array([[True, False], [True, False]])  # no valid land cell
    assert color_limits(values, land) == (20.0, 60.0)

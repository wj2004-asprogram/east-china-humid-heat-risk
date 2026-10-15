"""Unit tests for unit conversions, relative humidity and the heat index."""

import numpy as np
import pytest
import xarray as xr

from src.indices import (
    celsius_to_fahrenheit,
    fahrenheit_to_celsius,
    heat_index_rothfusz,
    kelvin_to_celsius,
    relative_humidity,
)


def _da(values):
    return xr.DataArray(np.atleast_1d(np.asarray(values, dtype=float)))


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

def test_unit_conversions():
    assert kelvin_to_celsius(273.15) == pytest.approx(0.0)
    assert celsius_to_fahrenheit(0.0) == pytest.approx(32.0)
    assert celsius_to_fahrenheit(100.0) == pytest.approx(212.0)
    assert fahrenheit_to_celsius(212.0) == pytest.approx(100.0)
    assert fahrenheit_to_celsius(celsius_to_fahrenheit(36.6)) == pytest.approx(36.6)


# ---------------------------------------------------------------------------
# Relative humidity
# ---------------------------------------------------------------------------

def test_rh_is_100_when_dewpoint_equals_temperature():
    t = _da([0.0, 15.0, 30.0])
    assert np.allclose(relative_humidity(t, t), 100.0)


def test_rh_known_value():
    # Psychrometric tables: T = 30 °C, Td = 20 °C gives RH ≈ 55 %.
    rh = relative_humidity(_da(30.0), _da(20.0))
    assert float(rh[0]) == pytest.approx(55.0, abs=0.5)


def test_rh_is_clipped_to_100():
    # Dewpoint above air temperature (supersaturation / rounding) is clipped.
    rh = relative_humidity(_da(20.0), _da(21.0))
    assert float(rh[0]) == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Heat index
# ---------------------------------------------------------------------------

# NWS heat index chart values (°F) for RH >= 40 %, where the current code
# applies the Rothfusz regression. Chart values are rounded to whole °F.
NWS_CHART_F = [
    # (T °F, RH %, HI °F)
    (90, 50, 95),
    (90, 70, 106),
    (96, 55, 112),
    (100, 40, 109),
    (86, 85, 102),
    (84, 90, 98),   # RH > 85 % and 80-87 °F: high-humidity adjustment applies
]


@pytest.mark.parametrize("t_f, rh, hi_f", NWS_CHART_F)
def test_heat_index_matches_nws_chart(t_f, rh, hi_f):
    hi_c = heat_index_rothfusz(_da(fahrenheit_to_celsius(t_f)), _da(rh))
    # 1 °F tolerance (chart rounding) = 0.56 °C
    assert float(hi_c[0]) == pytest.approx(fahrenheit_to_celsius(hi_f), abs=0.6)


def test_heat_index_equals_air_temperature_when_cool():
    # Below 70 °F (~21 °C) the heat index is set to the air temperature.
    hi_c = heat_index_rothfusz(_da(15.0), _da(60.0))
    assert float(hi_c[0]) == pytest.approx(15.0)


@pytest.mark.xfail(
    strict=True,
    reason="Known issue, fixed in phase 1b: for RH < 40 % the code switches to the "
           "simple formula, while NOAA keeps the Rothfusz regression, so the heat "
           "index jumps by several °C at RH = 40 %.",
)
def test_heat_index_is_continuous_across_rh_40():
    t = _da([40.0, 40.0])
    hi_c = heat_index_rothfusz(t, _da([39.99, 40.01]))
    assert abs(float(hi_c[1] - hi_c[0])) < 0.1

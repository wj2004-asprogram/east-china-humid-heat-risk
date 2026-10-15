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

# NWS heat index chart values (°F). The chart starts at RH = 40 %; drier
# air is checked against MetPy below. Chart values are rounded to whole °F.
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


def _metpy_heat_index_c(t_c, rh):
    """Independent reference: MetPy's NWS heat index (°C)."""
    mpcalc = pytest.importorskip("metpy.calc")
    from metpy.units import units

    hi = mpcalc.heat_index(
        units.Quantity(np.asarray(t_c, dtype=float), "degC"),
        units.Quantity(np.asarray(rh, dtype=float), "percent"),
        mask_undefined=False,
    )
    return hi.to("degC").magnitude


def _grid(t_values, rh_values):
    t, rh = np.meshgrid(t_values, rh_values)
    return t.ravel(), rh.ravel()


def test_heat_index_matches_metpy_when_hot():
    # T >= 28 °C (82.4 °F): both our rule (average of simple formula and T
    # >= 80 °F) and MetPy's rule (simple formula >= 79 °F) select the
    # Rothfusz regression, so the two must agree for every humidity,
    # including hot, dry air (RH < 40 %) and both adjustments
    # (RH < 13 % for 80-112 °F, RH > 85 % for 80-87 °F).
    t_c, rh = _grid(
        [28.0, 30.0, 33.0, 36.0, 38.0, 40.0, 42.0, 44.0, 46.0],
        [3.0, 8.0, 12.5, 20.0, 30.0, 39.9, 40.1, 55.0, 70.0, 86.0, 95.0, 100.0],
    )
    hi_c = heat_index_rothfusz(_da(t_c), _da(rh))
    np.testing.assert_allclose(hi_c.values, _metpy_heat_index_c(t_c, rh), atol=0.01)


def test_heat_index_matches_metpy_when_cool():
    # 5-24 °C: both rules give the simple formula itself (not averaged with T).
    t_c, rh = _grid([5.0, 10.0, 15.0, 20.0, 24.0], [10.0, 40.0, 70.0, 100.0])
    hi_c = heat_index_rothfusz(_da(t_c), _da(rh))
    np.testing.assert_allclose(hi_c.values, _metpy_heat_index_c(t_c, rh), atol=0.01)


def test_heat_index_is_continuous_across_rh_40():
    # NOAA has no switch at RH = 40 %; v0.2 jumped by ~8 °C here at 40 °C.
    t = _da([40.0, 40.0])
    hi_c = heat_index_rothfusz(t, _da([39.99, 40.01]))
    assert abs(float(hi_c[1] - hi_c[0])) < 0.1


def test_heat_index_adjustments_include_80_f():
    # NOAA/WPC applies the adjustments for T "between" 80 and 112 / 87 °F;
    # the NWS calculator includes the bounds. Hand calculation at exactly
    # T = 80 °F (also MetPy 1.7.1: 86.3419 °F at RH 90 %):
    #   RH 90 %: simple = 81.93, average with T = 80.965 >= 80 -> regression
    #            85.6419 + high-humidity adjustment (90-85)/10*(87-80)/5 = 0.7
    #            -> 86.3419 °F
    #   RH 10 %: simple = 78.17, average with T = 79.085 < 80 -> simple formula,
    #            no adjustment (it belongs to the regression) -> 78.17 °F.
    #            MetPy gives 77.91 °F here because it also adjusts the simple
    #            formula, so MetPy is not the reference for this case.
    t_c = _da([fahrenheit_to_celsius(80.0)] * 2)
    hi_f = celsius_to_fahrenheit(heat_index_rothfusz(t_c, _da([90.0, 10.0])))
    assert float(hi_f[0]) == pytest.approx(86.3419, abs=1e-3)
    assert float(hi_f[1]) == pytest.approx(78.17, abs=1e-3)

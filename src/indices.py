"""
indices.py

Unit conversions, relative humidity, and the NOAA heat index.

All functions are pure (no file I/O) so they can be unit-tested with
known values.
"""

import numpy as np
import xarray as xr


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

def kelvin_to_celsius(t_k):
    """Convert temperature from Kelvin to Celsius."""
    return t_k - 273.15


def celsius_to_fahrenheit(t_c):
    """Convert temperature from Celsius to Fahrenheit."""
    return t_c * 9.0 / 5.0 + 32.0


def fahrenheit_to_celsius(t_f):
    """Convert temperature from Fahrenheit to Celsius."""
    return (t_f - 32.0) * 5.0 / 9.0


# ---------------------------------------------------------------------------
# Relative humidity
# ---------------------------------------------------------------------------

def relative_humidity(t_c: xr.DataArray, td_c: xr.DataArray) -> xr.DataArray:
    """
    Calculate relative humidity (%) from temperature and dewpoint in Celsius.

    Uses the August–Roche–Magnus approximation:

        es = 6.112 * exp(17.67 * T  / (T  + 243.5))   [hPa]
        e  = 6.112 * exp(17.67 * Td / (Td + 243.5))   [hPa]
        RH = 100 * e / es

    Result is clipped to [0, 100].
    """
    es = 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))  # saturation vapour pressure
    e  = 6.112 * np.exp(17.67 * td_c / (td_c + 243.5))  # actual vapour pressure
    rh = 100.0 * e / es
    rh = rh.clip(0.0, 100.0)
    return rh


# ---------------------------------------------------------------------------
# Heat index (NOAA / Rothfusz)
# ---------------------------------------------------------------------------

def _heat_index_simple(t_f, rh):
    """
    NOAA simple heat index formula (°F), consistent with Steadman (1979)
    for heat index values below about 80 °F:

        HI = 0.5 * (T + 61.0 + (T - 68.0) * 1.2 + RH * 0.094)

    T in °F, RH in %.
    """
    return 0.5 * (t_f + 61.0 + (t_f - 68.0) * 1.2 + rh * 0.094)


def _heat_index_regression(t_f, rh):
    """
    Rothfusz (1990) regression (°F) with the NOAA adjustments for very dry
    (RH < 13 %) and very humid (RH > 85 %) air. T in °F, RH in %.
    """
    hi_f = (
        -42.379
        + 2.04901523 * t_f
        + 10.14333127 * rh
        - 0.22475541 * t_f * rh
        - 0.00683783 * t_f ** 2
        - 0.05481717 * rh ** 2
        + 0.00122874 * t_f ** 2 * rh
        + 0.00085282 * t_f * rh ** 2
        - 0.00000199 * t_f ** 2 * rh ** 2
    )

    # --- Adjustment 1: RH < 13 % and 80 °F < T < 112 °F ---
    cond1 = (rh < 13.0) & (t_f > 80.0) & (t_f < 112.0)
    # Clip sqrt argument to avoid negative values outside the condition range
    sqrt_arg = np.maximum((17.0 - np.abs(t_f - 95.0)) / 17.0, 0.0)
    adj1 = ((13.0 - rh) / 4.0) * np.sqrt(sqrt_arg)
    hi_f = xr.where(cond1, hi_f - adj1, hi_f)

    # --- Adjustment 2: RH > 85 % and 80 °F < T < 87 °F ---
    cond2 = (rh > 85.0) & (t_f > 80.0) & (t_f < 87.0)
    adj2 = ((rh - 85.0) / 10.0) * ((87.0 - t_f) / 5.0)
    hi_f = xr.where(cond2, hi_f + adj2, hi_f)

    return hi_f


def heat_index_rothfusz(
    t_c: xr.DataArray,
    rh: xr.DataArray,
) -> xr.DataArray:
    """
    Calculate hourly heat index (°C) following the NOAA/WPC algorithm.

    Steps
    -----
    1. Convert temperature to °F.
    2. Compute the simple formula
       HI_simple = 0.5 * (T + 61.0 + (T - 68.0) * 1.2 + RH * 0.094).
    3. If the average of HI_simple and T is >= 80 °F, the heat index is the
       Rothfusz regression with the NOAA adjustments for RH < 13 % and
       RH > 85 %, whatever the humidity.
    4. Otherwise the heat index is HI_simple itself (the average with T is
       only used for the decision in step 3).
    5. Convert back to °C.

    The index is meant for warm conditions; no special rule is applied to
    cold air.

    References
    ----------
    https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml
    Rothfusz, L.P. (1990). NWS Southern Region Technical Attachment SR 90-23.
    Steadman, R.G. (1979). J. Appl. Meteor., 18(7), 861–873.
    """
    t_f = celsius_to_fahrenheit(t_c)

    simple_f = _heat_index_simple(t_f, rh)
    use_regression = (simple_f + t_f) / 2.0 >= 80.0
    hi_f = xr.where(use_regression, _heat_index_regression(t_f, rh), simple_f)

    return fahrenheit_to_celsius(hi_f)

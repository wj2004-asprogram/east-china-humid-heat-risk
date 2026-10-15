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

def _heat_index_simple(t_f: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """
    Simple (first-approximation) heat index formula.

    Used by NOAA when the full Rothfusz regression is not applicable.
    Returns heat index in °F.
    """
    hi = (0.5 * (t_f + 61.0
                 + (t_f - 68.0) * 1.2
                 + rh * 0.094))
    return (hi + t_f) / 2.0


def heat_index_rothfusz(
    t_c: xr.DataArray,
    rh: xr.DataArray,
) -> xr.DataArray:
    """
    Calculate hourly heat index (°C) using the NOAA / Rothfusz formula.

    Steps
    -----
    1. Convert temperature to °F.
    2. Apply the Rothfusz regression where T >= 80 °F and RH >= 40 %.
    3. Apply NOAA adjustments for extreme RH conditions.
    4. For T < 80 °F or RH < 40 %, use the simple formula, averaging
       with the air temperature.
    5. Convert back to °C.

    Known issue (to be fixed in phase 1b, see docs/ROADMAP.md): NOAA applies
    the Rothfusz regression whenever the simple estimate reaches 80 °F,
    whatever the humidity. Switching to the simple formula for RH < 40 %
    underestimates the heat index on hot, dry afternoons.

    References
    ----------
    https://www.weather.gov/media/ffc/ta_htindx.PDF
    Steadman, R.G. (1979). J. Appl. Meteor., 18(7), 861–873.
    """
    t_f = celsius_to_fahrenheit(t_c)

    # --- Rothfusz regression (core) ---
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

    # --- Simple formula for cooler / low-RH conditions ---
    simple_hi = _heat_index_simple(t_f.values, rh.values)
    # Where Rothfusz doesn't apply, use the simple formula
    low_cond = (t_f < 80.0) | (rh < 40.0)
    hi_f = xr.where(low_cond, simple_hi, hi_f)

    # --- For very cold conditions, just use the air temperature ---
    cold_cond = t_f < 70.0  # ~21 °C
    hi_f = xr.where(cold_cond, t_f, hi_f)

    # --- Convert back to Celsius ---
    hi_c = fahrenheit_to_celsius(hi_f)
    return hi_c

"""
calculate_indices.py

Calculate humid-heat indices from ERA5 temperature and humidity variables.

Supported indices:
    - Heat Index (HI) — NOAA formula using T and RH
    - Humidex — Canadian index using T and Td
    - Simplified Wet-Bulb Globe Temperature (sWBGT)

Usage:
    python src/calculate_indices.py
"""

import xarray as xr
import numpy as np
from pathlib import Path


# --- Thermodynamic helpers ---------------------------------------------------

def saturation_vapour_pressure(T_C: np.ndarray) -> np.ndarray:
    """
    Saturation vapour pressure (hPa) via the August–Roche–Magnus formula.

    Parameters
    ----------
    T_C : air temperature in °C

    Returns
    -------
    es : saturation vapour pressure in hPa
    """
    return 6.1094 * np.exp(17.625 * T_C / (T_C + 243.04))


def relative_humidity(T_C: np.ndarray, Td_C: np.ndarray) -> np.ndarray:
    """
    Relative humidity (%) from temperature and dewpoint.

    Parameters
    ----------
    T_C  : air temperature, °C
    Td_C : dewpoint temperature, °C

    Returns
    -------
    RH : relative humidity, %
    """
    e = saturation_vapour_pressure(Td_C)
    es = saturation_vapour_pressure(T_C)
    return 100.0 * e / es


# --- Humid-heat indices ------------------------------------------------------

def heat_index(T_C: np.ndarray, RH_pct: np.ndarray) -> np.ndarray:
    """
    NOAA Heat Index (°C) — Rothfusz regression, adjusted for low/high RH.

    Parameters
    ----------
    T_C    : air temperature, °C
    RH_pct : relative humidity, %

    Returns
    -------
    HI : heat index, °C
    """
    T_F = T_C * 9.0 / 5.0 + 32.0  # convert to Fahrenheit for NOAA formula

    # Simple formula first
    hi_F_simple = (
        0.5 * (T_F + 61.0 + (T_F - 68.0) * 1.2 + RH_pct * 0.094)
    )

    # Full Rothfusz regression
    hi_F_full = (
        -42.379
        + 2.04901523 * T_F
        + 10.14333127 * RH_pct
        - 0.22475541 * T_F * RH_pct
        - 6.83783e-3 * T_F**2
        - 5.481717e-2 * RH_pct**2
        + 1.22874e-3 * T_F**2 * RH_pct
        + 8.5282e-4 * T_F * RH_pct**2
        - 1.99e-6 * T_F**2 * RH_pct**2
    )

    # Adjustment for low RH
    hi_F = np.where(
        RH_pct < 13,
        hi_F_simple - ((13 - RH_pct) / 4) * np.sqrt((17 - np.abs(T_F - 95.0)) / 17),
        hi_F_full
    )

    # Adjustment for high RH
    hi_F = np.where(
        (RH_pct > 85) & (T_F > 80) & (T_F < 87),
        hi_F + ((RH_pct - 85) / 10) * ((87 - T_F) / 5),
        hi_F
    )

    # Use simple average when HI < 80°F
    hi_F = np.where(hi_F < 80, 0.5 * (T_F + hi_F_simple), hi_F)

    return (hi_F - 32.0) * 5.0 / 9.0  # back to °C


def humidex(T_C: np.ndarray, Td_C: np.ndarray) -> np.ndarray:
    """
    Canadian Humidex (dimensionless; °C-equivalent).

    Parameters
    ----------
    T_C  : air temperature, °C
    Td_C : dewpoint temperature, °C

    Returns
    -------
    humidex value
    """
    e = saturation_vapour_pressure(Td_C)  # hPa
    return T_C + 0.5555 * (e - 10.0)


# --- Main --------------------------------------------------------------------

def main():
    project_root = Path(__file__).resolve().parent.parent
    input_path = project_root / "data" / "processed" / "era5_east_china_jja.nc"
    output_path = project_root / "data" / "processed" / "era5_east_china_indices.nc"

    print(f"Loading {input_path} ...")
    ds = xr.open_dataset(input_path)

    # Convert K to °C
    T_C = ds["t2m"] - 273.15
    Td_C = ds["d2m"] - 273.15

    RH = relative_humidity(T_C.values, Td_C.values)

    print("Calculating Heat Index ...")
    ds["heat_index"] = (["time", "latitude", "longitude"], heat_index(T_C.values, RH))

    print("Calculating Humidex ...")
    ds["humidex"] = (["time", "latitude", "longitude"], humidex(T_C.values, Td_C.values))

    # Keep T and Td in °C as additional variables for convenience
    ds["t2m_C"] = T_C
    ds["d2m_C"] = Td_C

    print(f"Saving to {output_path} ...")
    ds.to_netcdf(output_path)
    print("Done.")


if __name__ == "__main__":
    main()

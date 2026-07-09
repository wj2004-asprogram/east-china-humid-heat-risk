"""
process_humid_heat.py

Read ERA5 hourly single-level data for East China, calculate humid heat risk
indices, and save the result as a new NetCDF file.

Indices computed:
  - t2m_c, d2m_c   : 2m temperature / dewpoint in Celsius
  - rh              : relative humidity (%, from August–Roche–Magnus)
  - heat_index_c    : hourly heat index (°C, NOAA/Rothfusz)
  - daily_max_heat_index_c : daily maximum heat index
  - daily_mean_rh           : daily mean relative humidity

Usage:
    python src/process_humid_heat.py
"""

import numpy as np
import xarray as xr
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Coordinate / variable detection helpers
# ---------------------------------------------------------------------------

def detect_time_coord(ds: xr.Dataset) -> str:
    """
    Detect the time coordinate name.

    ERA5 data from CDS may use "time" or "valid_time" depending on
    the download method (CDS API vs CDS Beta).
    """
    for candidate in ["time", "valid_time"]:
        if candidate in ds.coords:
            return candidate
    raise KeyError(
        f"Cannot find a time coordinate (tried 'time', 'valid_time'). "
        f"Available coords: {list(ds.coords)}"
    )


def detect_lat_coord(ds: xr.Dataset) -> str:
    """Detect the latitude coordinate name."""
    for candidate in ["latitude", "lat"]:
        if candidate in ds.coords:
            return candidate
    raise KeyError(
        f"Cannot find latitude coordinate (tried 'latitude', 'lat'). "
        f"Available coords: {list(ds.coords)}"
    )


def detect_lon_coord(ds: xr.Dataset) -> str:
    """Detect the longitude coordinate name."""
    for candidate in ["longitude", "lon"]:
        if candidate in ds.coords:
            return candidate
    raise KeyError(
        f"Cannot find longitude coordinate (tried 'longitude', 'lon'). "
        f"Available coords: {list(ds.coords)}"
    )


def detect_t2m_var(ds: xr.Dataset) -> str:
    """Detect the 2m temperature variable name."""
    for candidate in ["t2m", "2m_temperature"]:
        if candidate in ds.data_vars:
            return candidate
    raise KeyError(
        f"Cannot find a 2m temperature variable (tried 't2m', '2m_temperature'). "
        f"Available data vars: {list(ds.data_vars)}"
    )


def detect_d2m_var(ds: xr.Dataset) -> str:
    """Detect the 2m dewpoint temperature variable name."""
    for candidate in ["d2m", "2m_dewpoint_temperature"]:
        if candidate in ds.data_vars:
            return candidate
    raise KeyError(
        f"Cannot find a 2m dewpoint variable (tried 'd2m', '2m_dewpoint_temperature'). "
        f"Available data vars: {list(ds.data_vars)}"
    )


# ---------------------------------------------------------------------------
# 2. Unit conversion
# ---------------------------------------------------------------------------

def kelvin_to_celsius(t_k: xr.DataArray) -> xr.DataArray:
    """Convert temperature from Kelvin to Celsius."""
    return t_k - 273.15


def celsius_to_fahrenheit(t_c: xr.DataArray) -> xr.DataArray:
    """Convert temperature from Celsius to Fahrenheit."""
    return t_c * 9.0 / 5.0 + 32.0


def fahrenheit_to_celsius(t_f: xr.DataArray) -> xr.DataArray:
    """Convert temperature from Fahrenheit to Celsius."""
    return (t_f - 32.0) * 5.0 / 9.0


# ---------------------------------------------------------------------------
# 3. Relative humidity
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
# 4. Heat index (NOAA / Rothfusz)
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


# ---------------------------------------------------------------------------
# 5. Main processing pipeline
# ---------------------------------------------------------------------------

def main():
    # --- Paths ---
    project_root = Path(__file__).resolve().parent.parent
    input_path = project_root / "data" / "raw" / "era5_east_china_2022_07_01_07.nc"
    output_path = project_root / "data" / "processed" / "humid_heat_2022_07_01_07.nc"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # =======================================================================
    # Step 1: Open the NetCDF file
    # =======================================================================
    print(f"Input file:  {input_path}")
    ds = xr.open_dataset(input_path)

    # =======================================================================
    # Step 2: Auto-detect coordinate and variable names
    # =======================================================================
    time_coord = detect_time_coord(ds)
    lat_coord = detect_lat_coord(ds)
    lon_coord = detect_lon_coord(ds)
    t_var = detect_t2m_var(ds)
    d_var = detect_d2m_var(ds)

    print(f"Detected time coordinate:      '{time_coord}'")
    print(f"Detected latitude coordinate:  '{lat_coord}'")
    print(f"Detected longitude coordinate: '{lon_coord}'")
    print(f"Detected temperature variable: '{t_var}'")
    print(f"Detected dewpoint variable:    '{d_var}'")

    # =======================================================================
    # Step 3: Convert temperature to Celsius
    # =======================================================================
    t2m_c = kelvin_to_celsius(ds[t_var])
    t2m_c.attrs["units"] = "Celsius"
    t2m_c.attrs["long_name"] = "2 metre temperature"
    t2m_c.attrs["description"] = "2m air temperature converted from Kelvin to Celsius"

    d2m_c = kelvin_to_celsius(ds[d_var])
    d2m_c.attrs["units"] = "Celsius"
    d2m_c.attrs["long_name"] = "2 metre dewpoint temperature"
    d2m_c.attrs["description"] = (
        "2m dewpoint temperature converted from Kelvin to Celsius"
    )

    # =======================================================================
    # Step 4: Calculate relative humidity
    # =======================================================================
    rh = relative_humidity(t2m_c, d2m_c)
    rh.attrs["units"] = "%"
    rh.attrs["long_name"] = "relative humidity"
    rh.attrs["description"] = (
        "Relative humidity calculated from 2m temperature and dewpoint "
        "using the August–Roche–Magnus approximation; clipped to [0, 100]"
    )

    # =======================================================================
    # Step 5: Calculate hourly heat index
    # =======================================================================
    heat_index_c = heat_index_rothfusz(t2m_c, rh)
    heat_index_c.attrs["units"] = "Celsius"
    heat_index_c.attrs["long_name"] = "heat index"
    heat_index_c.attrs["description"] = (
        "Hourly heat index computed with the NOAA/Rothfusz formula; "
        "for air temperature < 26.7 °C or RH < 40 % a simple "
        "approximation is used"
    )

    # =======================================================================
    # Step 6: Build output dataset
    # =======================================================================
    ds_out = xr.Dataset(
        {
            "t2m_c": t2m_c,
            "d2m_c": d2m_c,
            "rh": rh,
            "heat_index_c": heat_index_c,
        },
        attrs={
            "title": "Humid heat risk indices for East China",
            "period": "2022-07-01 to 2022-07-07",
            "source": "ERA5 hourly single-level reanalysis",
            "created_with": "process_humid_heat.py",
        },
    )

    # =======================================================================
    # Step 7: Resample time coord → "time" for daily calculations
    # =======================================================================
    if time_coord != "time":
        ds_out = ds_out.rename({time_coord: "time"})

    # Daily maximum heat index
    daily_max_heat_index_c = (
        ds_out["heat_index_c"]
        .resample(time="1D")
        .max(keep_attrs=True)
        .rename({"time": "date"})
    )
    daily_max_heat_index_c.attrs["units"] = "Celsius"
    daily_max_heat_index_c.attrs["long_name"] = "daily maximum heat index"
    daily_max_heat_index_c.attrs["description"] = (
        "Daily maximum of the hourly NOAA/Rothfusz heat index"
    )

    # Daily mean relative humidity
    daily_mean_rh = (
        ds_out["rh"]
        .resample(time="1D")
        .mean(keep_attrs=True)
        .rename({"time": "date"})
    )
    daily_mean_rh.attrs["units"] = "%"
    daily_mean_rh.attrs["long_name"] = "daily mean relative humidity"
    daily_mean_rh.attrs["description"] = (
        "Daily mean of hourly relative humidity"
    )

    # Add daily variables to dataset
    ds_out["daily_max_heat_index_c"] = daily_max_heat_index_c
    ds_out["daily_mean_rh"] = daily_mean_rh

    # =======================================================================
    # Step 8: Save
    # =======================================================================
    ds_out.to_netcdf(output_path)
    print(f"\nOutput file: {output_path}")

    # =======================================================================
    # Step 9: Quick statistics
    # =======================================================================
    print("\n=== Variable statistics ===")
    for vname in [
        "t2m_c",
        "d2m_c",
        "rh",
        "heat_index_c",
        "daily_max_heat_index_c",
        "daily_mean_rh",
    ]:
        da = ds_out[vname]
        print(
            f"  {vname:<26s}  "
            f"min={da.min().values:.2f}  "
            f"max={da.max().values:.2f}  "
            f"({da.attrs.get('units', '')})"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()

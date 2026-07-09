"""
process_temperature.py

Open a raw ERA5 NetCDF file, convert 2m temperature from Kelvin to Celsius,
calculate daily maximum temperature, and save the result.

Usage:
    python src/process_temperature.py
"""

import xarray as xr
from pathlib import Path


def detect_time_coord(ds: xr.Dataset) -> str:
    """
    Detect the time coordinate name in the dataset.

    ERA5 data from CDS may use "time" or "valid_time" depending on
    the download method (CDS API vs CDS Beta / new CDS).
    """
    for candidate in ["time", "valid_time"]:
        if candidate in ds.coords:
            return candidate
    raise KeyError(
        f"Cannot find a time coordinate (tried 'time', 'valid_time'). "
        f"Available coords: {list(ds.coords)}"
    )


def detect_t2m_var(ds: xr.Dataset) -> str:
    """
    Detect the 2m temperature variable name.

    CDS may deliver it as "t2m" (short name) or "2m_temperature" (long name).
    """
    for candidate in ["t2m", "2m_temperature"]:
        if candidate in ds.data_vars:
            return candidate
    raise KeyError(
        f"Cannot find a 2m temperature variable (tried 't2m', '2m_temperature'). "
        f"Available data vars: {list(ds.data_vars)}"
    )


def kelvin_to_celsius(t_k: xr.DataArray) -> xr.DataArray:
    """Convert temperature from Kelvin to Celsius."""
    return t_k - 273.15


def daily_max(ds: xr.Dataset, time_coord: str) -> xr.Dataset:
    """
    Resample hourly data to daily maximum over each grid point.

    Uses xarray's resample method with the detected time coordinate.
    """
    daily = ds.resample({time_coord: "1D"}).max(keep_attrs=True)
    return daily


def main():
    # --- Paths ---
    project_root = Path(__file__).resolve().parent.parent
    input_path = project_root / "data" / "raw" / "era5_east_china_2022_07_01_07.nc"
    output_path = project_root / "data" / "processed" / "daily_tmax_2022_07_01_07.nc"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- 1. Open the file ---
    print(f"Opening {input_path} ...")
    ds = xr.open_dataset(input_path)

    # --- 2. Print dataset summary ---
    print("\n=== Dataset summary ===")
    print(ds)

    # --- 3. Print all variable names ---
    print("\n=== Variable names ===")
    print("Data variables:", list(ds.data_vars))
    print("Coordinates:", list(ds.coords))

    # --- 4. Detect time coordinate ---
    time_coord = detect_time_coord(ds)
    print(f"\nDetected time coordinate: '{time_coord}'")

    # --- 5. Detect and convert 2m temperature ---
    t_var = detect_t2m_var(ds)
    print(f"Detected 2m temperature variable: '{t_var}'")
    print(f"  Unit before conversion: {ds[t_var].attrs.get('units', 'unknown')}")

    ds[t_var] = kelvin_to_celsius(ds[t_var])
    ds[t_var].attrs["units"] = "Celsius"
    ds[t_var].attrs["long_name"] = "2 metre temperature (daily maximum)"

    print(f"  Converted from Kelvin to Celsius.")

    # --- 6. Calculate daily maximum ---
    print("\nCalculating daily maximum temperature over each grid point ...")
    ds_daily = daily_max(ds, time_coord)

    # Quick peek at the result
    print(f"\nOriginal time steps: {ds.sizes[time_coord]}")
    print(f"Daily time steps:    {ds_daily.sizes[time_coord]}")
    print(f"\nDaily max temperature preview (first day, first few grid points):")
    print(ds_daily[t_var].isel({time_coord: 0})[:3, :3].values)

    # --- 7. Save result ---
    print(f"\nSaving to {output_path} ...")
    ds_daily.to_netcdf(output_path)
    print("Done.")


if __name__ == "__main__":
    main()

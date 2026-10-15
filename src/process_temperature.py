"""
process_temperature.py

Open the raw ERA5 NetCDF file, convert 2m temperature from Kelvin to
Celsius, calculate daily maximum temperature, and save the result.

Daily values are local calendar days (config `utc_offset_hours`; 8 =
Beijing time), see src/time_utils.py.
"""

import xarray as xr

from src.config import RunConfig
from src.era5_utils import detect_t2m_var, detect_time_coord, load_dataset
from src.indices import kelvin_to_celsius
from src.time_utils import local_daily


def run(cfg: RunConfig) -> None:
    input_path = cfg.paths.raw_era5
    output_path = cfg.paths.daily_tmax_nc
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- 1. Open the file ---
    print(f"Opening {input_path} ...")
    ds = load_dataset(input_path)

    print("\n=== Dataset summary ===")
    print(ds)

    # --- 2. Detect time coordinate and temperature variable ---
    time_coord = detect_time_coord(ds)
    t_var = detect_t2m_var(ds)
    print(f"\nDetected time coordinate: '{time_coord}'")
    print(f"Detected 2m temperature variable: '{t_var}'")
    print(f"  Unit before conversion: {ds[t_var].attrs.get('units', 'unknown')}")

    # --- 3. Convert to Celsius ---
    ds[t_var] = kelvin_to_celsius(ds[t_var])
    ds[t_var].attrs["units"] = "Celsius"
    ds[t_var].attrs["long_name"] = "2 metre temperature (daily maximum)"

    # --- 4. Daily maximum ---
    print(f"\nCalculating daily maximum temperature over each grid point "
          f"(local days, UTC{cfg.utc_offset_hours:+d}) ...")
    ds_daily = local_daily(ds, time_coord, cfg.utc_offset_hours, "max", cfg.start_date, cfg.end_date)
    ds_daily.attrs["daily_aggregation"] = f"local calendar days, UTC{cfg.utc_offset_hours:+d}"
    print(f"Original time steps: {ds.sizes[time_coord]}")
    print(f"Daily time steps:    {ds_daily.sizes[time_coord]}")

    # --- 5. Save ---
    print(f"\nSaving to {output_path} ...")
    ds_daily.to_netcdf(output_path)
    print("Done.")

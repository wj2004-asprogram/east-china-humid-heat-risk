"""
process_temperature.py

Open the raw ERA5 NetCDF file(s), convert 2m temperature from Kelvin to
Celsius, calculate daily maximum temperature, and save the result.

Daily values are local calendar days (config `utc_offset_hours`; 8 =
Beijing time), see src/time_utils.py. Multi-year runs are processed one
month at a time (src/hourly_input.py) and the daily values concatenated.
"""

from src.config import RunConfig
from src.era5_utils import detect_t2m_var, detect_time_coord
from src.hourly_input import concat_days, iter_hourly_blocks
from src.indices import kelvin_to_celsius
from src.time_utils import local_daily


def run(cfg: RunConfig) -> None:
    output_path = cfg.paths.daily_tmax_nc
    output_path.parent.mkdir(parents=True, exist_ok=True)

    daily_parts = []
    for i, block in enumerate(iter_hourly_blocks(cfg)):
        ds = block.data

        # --- 1. Detect time coordinate and temperature variable (first block) ---
        time_coord = detect_time_coord(ds)
        t_var = detect_t2m_var(ds)
        if i == 0:
            print("=== Dataset summary (first block) ===")
            print(ds)
            print(f"\nDetected time coordinate: '{time_coord}'")
            print(f"Detected 2m temperature variable: '{t_var}'")
            print(f"  Unit before conversion: {ds[t_var].attrs.get('units', 'unknown')}")
            print(f"\nCalculating daily maximum temperature over each grid point "
                  f"(local days, UTC{cfg.utc_offset_hours:+d}) ...")

        # --- 2. Convert to Celsius ---
        ds[t_var] = kelvin_to_celsius(ds[t_var])
        ds[t_var].attrs["units"] = "Celsius"
        ds[t_var].attrs["long_name"] = "2 metre temperature (daily maximum)"

        # --- 3. Daily maximum ---
        daily = local_daily(ds, time_coord, cfg.utc_offset_hours, "max", block.first_day, block.last_day)
        print(f"  {block.first_day} to {block.last_day}: {ds.sizes[time_coord]} hourly time steps "
              f"-> {daily.sizes[time_coord]} days")
        daily_parts.append(daily)

    ds_daily = concat_days(daily_parts, time_coord, cfg)
    ds_daily.attrs["daily_aggregation"] = f"local calendar days, UTC{cfg.utc_offset_hours:+d}"
    print(f"Daily time steps: {ds_daily.sizes[time_coord]}")

    # --- 4. Save ---
    print(f"\nSaving to {output_path} ...")
    ds_daily.to_netcdf(output_path)
    print("Done.")

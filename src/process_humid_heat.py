"""
process_humid_heat.py

Read ERA5 hourly single-level data for East China, calculate humid heat
risk indices, and save the result as a new NetCDF file.

Indices computed:
  - t2m_c, d2m_c   : 2m temperature / dewpoint in Celsius
  - rh              : relative humidity (%, from August–Roche–Magnus)
  - heat_index_c    : hourly heat index (°C, NOAA/Rothfusz)
  - daily_max_heat_index_c : daily maximum heat index
  - daily_mean_rh           : daily mean relative humidity

Hourly variables keep the UTC timestamps of ERA5 ("time"). Daily values
("date") are local calendar days (config `utc_offset_hours`; 8 = Beijing
time), see src/time_utils.py.
"""

import xarray as xr

from src.config import RunConfig
from src.era5_utils import (
    detect_d2m_var,
    detect_lat_coord,
    detect_lon_coord,
    detect_t2m_var,
    detect_time_coord,
    load_dataset,
)
from src.indices import heat_index_rothfusz, kelvin_to_celsius, relative_humidity
from src.time_utils import local_daily


def run(cfg: RunConfig) -> None:
    input_path = cfg.paths.raw_era5
    output_path = cfg.paths.humid_heat_nc
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # =======================================================================
    # Step 1: Open the NetCDF file
    # =======================================================================
    print(f"Input file:  {input_path}")
    ds = load_dataset(input_path)

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
        "Hourly heat index following the NOAA/WPC algorithm: Rothfusz "
        "regression (with the RH < 13 % and RH > 85 % adjustments) where the "
        "average of the simple formula and the air temperature is >= 80 °F "
        "(26.7 °C), otherwise the simple formula"
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
            "period": cfg.period_label,
            "source": "ERA5 hourly single-level reanalysis",
            "created_with": "src/process_humid_heat.py",
            "daily_aggregation": f"local calendar days, UTC{cfg.utc_offset_hours:+d}",
        },
    )

    # =======================================================================
    # Step 7: Daily aggregation (local calendar days) on a "time" coordinate
    # =======================================================================
    if time_coord != "time":
        ds_out = ds_out.rename({time_coord: "time"})

    def to_local_days(da: xr.DataArray, how: str) -> xr.DataArray:
        return local_daily(
            da, "time", cfg.utc_offset_hours, how, cfg.start_date, cfg.end_date,
        ).rename({"time": "date"})

    # Daily maximum heat index
    daily_max_heat_index_c = to_local_days(ds_out["heat_index_c"], "max")
    daily_max_heat_index_c.attrs["units"] = "Celsius"
    daily_max_heat_index_c.attrs["long_name"] = "daily maximum heat index"
    daily_max_heat_index_c.attrs["description"] = (
        "Daily maximum of the hourly NOAA/Rothfusz heat index over local "
        f"calendar days (UTC{cfg.utc_offset_hours:+d})"
    )

    # Daily mean relative humidity
    daily_mean_rh = to_local_days(ds_out["rh"], "mean")
    daily_mean_rh.attrs["units"] = "%"
    daily_mean_rh.attrs["long_name"] = "daily mean relative humidity"
    daily_mean_rh.attrs["description"] = (
        "Daily mean of hourly relative humidity over local calendar days "
        f"(UTC{cfg.utc_offset_hours:+d})"
    )

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

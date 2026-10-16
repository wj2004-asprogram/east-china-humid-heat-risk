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

Multi-year runs are processed one month at a time (src/hourly_input.py)
and only the daily variables are saved: the hourly fields of 25 summers
would take several GB, and no later step reads them.
"""

import xarray as xr

from src.config import RunConfig
from src.era5_utils import (
    detect_d2m_var,
    detect_lat_coord,
    detect_lon_coord,
    detect_t2m_var,
    detect_time_coord,
)
from src.hourly_input import concat_days, iter_hourly_blocks
from src.indices import heat_index_rothfusz, kelvin_to_celsius, relative_humidity
from src.time_utils import local_daily


HOURLY_VARIABLES = ["t2m_c", "d2m_c", "rh", "heat_index_c"]
DAILY_VARIABLES = ["daily_max_heat_index_c", "daily_mean_rh"]


def hourly_indices(ds: xr.Dataset, verbose: bool = False) -> xr.Dataset:
    """
    Hourly temperature, dewpoint (°C), relative humidity and heat index
    from raw ERA5 data, on a time coordinate named "time" (UTC).
    """
    # =======================================================================
    # Step 1: Auto-detect coordinate and variable names
    # =======================================================================
    time_coord = detect_time_coord(ds)
    t_var = detect_t2m_var(ds)
    d_var = detect_d2m_var(ds)

    if verbose:
        print(f"Detected time coordinate:      '{time_coord}'")
        print(f"Detected latitude coordinate:  '{detect_lat_coord(ds)}'")
        print(f"Detected longitude coordinate: '{detect_lon_coord(ds)}'")
        print(f"Detected temperature variable: '{t_var}'")
        print(f"Detected dewpoint variable:    '{d_var}'")

    # =======================================================================
    # Step 2: Convert temperature to Celsius
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
    # Step 3: Calculate relative humidity
    # =======================================================================
    rh = relative_humidity(t2m_c, d2m_c)
    rh.attrs["units"] = "%"
    rh.attrs["long_name"] = "relative humidity"
    rh.attrs["description"] = (
        "Relative humidity calculated from 2m temperature and dewpoint "
        "using the August–Roche–Magnus approximation; clipped to [0, 100]"
    )

    # =======================================================================
    # Step 4: Calculate hourly heat index
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

    hourly = xr.Dataset({"t2m_c": t2m_c, "d2m_c": d2m_c, "rh": rh, "heat_index_c": heat_index_c})
    if time_coord != "time":
        hourly = hourly.rename({time_coord: "time"})
    return hourly


def run(cfg: RunConfig) -> None:
    output_path = cfg.paths.humid_heat_nc
    output_path.parent.mkdir(parents=True, exist_ok=True)
    utc_label = f"UTC{cfg.utc_offset_hours:+d}"

    # =======================================================================
    # Steps 1-5: hourly indices and daily aggregation (local calendar days),
    # block by block (one month at a time for multi-year runs)
    # =======================================================================
    hourly_parts, daily_max_parts, daily_rh_parts = [], [], []
    for i, block in enumerate(iter_hourly_blocks(cfg)):
        hourly = hourly_indices(block.data, verbose=(i == 0))

        def to_local_days(da: xr.DataArray, how: str) -> xr.DataArray:
            return local_daily(
                da, "time", cfg.utc_offset_hours, how, block.first_day, block.last_day,
            ).rename({"time": "date"})

        daily_max_parts.append(to_local_days(hourly["heat_index_c"], "max"))
        daily_rh_parts.append(to_local_days(hourly["rh"], "mean"))
        if not cfg.is_multi_year:
            hourly_parts.append(hourly)
        print(f"  {block.first_day} to {block.last_day}: {hourly.sizes['time']} hourly time steps")

    # Daily maximum heat index
    daily_max_heat_index_c = concat_days(daily_max_parts, "date", cfg)
    daily_max_heat_index_c.attrs["units"] = "Celsius"
    daily_max_heat_index_c.attrs["long_name"] = "daily maximum heat index"
    daily_max_heat_index_c.attrs["description"] = (
        "Daily maximum of the hourly NOAA/Rothfusz heat index over local "
        f"calendar days ({utc_label}); NaN if any hour is missing"
    )

    # Daily mean relative humidity
    daily_mean_rh = concat_days(daily_rh_parts, "date", cfg)
    daily_mean_rh.attrs["units"] = "%"
    daily_mean_rh.attrs["long_name"] = "daily mean relative humidity"
    daily_mean_rh.attrs["description"] = (
        "Daily mean of hourly relative humidity over local calendar days "
        f"({utc_label}); NaN if any hour is missing"
    )

    # =======================================================================
    # Step 6: Build output dataset
    # =======================================================================
    attrs = {
        "title": "Humid heat risk indices for East China",
        "period": cfg.period_label,
        "source": "ERA5 hourly single-level reanalysis",
        "created_with": "src/process_humid_heat.py",
        "daily_aggregation": f"local calendar days, {utc_label}",
    }
    if hourly_parts:
        ds_out = xr.concat(hourly_parts, dim="time") if len(hourly_parts) > 1 else hourly_parts[0]
        ds_out.attrs = attrs
    else:
        attrs["hourly_variables"] = (
            "not saved for multi-year runs; recompute them from the raw files if needed"
        )
        ds_out = xr.Dataset(attrs=attrs)
    ds_out["daily_max_heat_index_c"] = daily_max_heat_index_c
    ds_out["daily_mean_rh"] = daily_mean_rh

    # =======================================================================
    # Step 7: Save
    # =======================================================================
    ds_out.to_netcdf(output_path)
    print(f"\nOutput file: {output_path}")

    # =======================================================================
    # Step 8: Quick statistics
    # =======================================================================
    print("\n=== Variable statistics ===")
    for vname in [v for v in HOURLY_VARIABLES + DAILY_VARIABLES if v in ds_out]:
        da = ds_out[vname]
        print(
            f"  {vname:<26s}  "
            f"min={da.min().values:.2f}  "
            f"max={da.max().values:.2f}  "
            f"({da.attrs.get('units', '')})"
        )

    print("\nDone.")

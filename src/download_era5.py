"""
download_era5.py

Download from the Copernicus Climate Data Store (CDS), for the configured
region:
  - ERA5 hourly single-level 2m temperature and 2m dewpoint temperature
    for the configured period, plus the UTC hours of the neighbouring day
    needed to complete the first (or last) local day (config
    `utc_offset_hours`); both requests are merged into one raw file;
  - the ERA5 land-sea mask (time-invariant, a single time step).

Before running:
    1. Register at https://cds.climate.copernicus.eu/
    2. Set up your CDS API key (see https://cds.climate.copernicus.eu/how-to-api)
    3. Accept the ERA5 licence on the dataset page.
"""

from datetime import timedelta

import xarray as xr

from src.config import RunConfig
from src.era5_utils import detect_time_coord, load_dataset


VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
]


def _request(cfg: RunConfig, variables: list, days: list, hours) -> dict:
    """CDS request for `variables` on `days` (all in one month) at `hours` (UTC) over the region."""
    return {
        "product_type": ["reanalysis"],
        "variable": variables,
        "year": [f"{days[0].year:04d}"],
        "month": [f"{days[0].month:02d}"],
        "day": [f"{d.day:02d}" for d in days],
        "time": [f"{hour:02d}:00" for hour in hours],
        "area": cfg.cds_area,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def build_request(cfg: RunConfig) -> dict:
    """
    Build the CDS API request for every UTC hour of the configured period.

    Only periods within a single calendar month are supported for now:
    the CDS combines the year/month/day lists, so a multi-month request
    would also fetch days outside the period. Multi-month downloads
    (one request per month) are planned for phase 2.
    """
    if (cfg.start_date.year, cfg.start_date.month) != (cfg.end_date.year, cfg.end_date.month):
        raise NotImplementedError(
            f"Period {cfg.period_label} spans several months; "
            "multi-month downloads are not supported yet (phase 2)."
        )

    n_days = (cfg.end_date - cfg.start_date).days + 1
    days = [cfg.start_date + timedelta(days=i) for i in range(n_days)]
    return _request(cfg, VARIABLES, days, range(24))  # all 24 UTC hours


def build_edge_request(cfg: RunConfig):
    """
    Build the CDS API request for the UTC hours outside the period that
    belong to its first or last local day, or None for UTC days.

    East of UTC (offset h > 0), local day D starts at (24 - h):00 UTC on
    D-1, so the last h UTC hours of the day before the period are needed.
    For Beijing time (h = 8): 16:00-23:00 UTC on the day before. West of
    UTC (h < 0), the first |h| UTC hours of the day after the period are
    needed.
    """
    h = cfg.utc_offset_hours
    if h > 0:
        return _request(cfg, VARIABLES, [cfg.start_date - timedelta(days=1)], range(24 - h, 24))
    if h < 0:
        return _request(cfg, VARIABLES, [cfg.end_date + timedelta(days=1)], range(0, -h))
    return None


def build_land_sea_mask_request(cfg: RunConfig) -> dict:
    """
    Build the CDS API request for the ERA5 land-sea mask over the region.

    The land-sea mask does not change in time, so a single time step
    (00 UTC on the first day of the period) is enough.
    """
    return _request(cfg, ["land_sea_mask"], [cfg.start_date], [0])


def merge_hourly_files(input_files, output_file) -> None:
    """Concatenate hourly ERA5 NetCDF files along time (sorted) into one file."""
    parts = [load_dataset(path) for path in input_files]

    time_coord = detect_time_coord(parts[0])
    merged = xr.concat(parts, dim=time_coord).sortby(time_coord)
    if merged.indexes[time_coord].has_duplicates:
        raise ValueError(f"Duplicate time steps when merging {list(map(str, input_files))}.")

    # Let xarray choose the on-disk encoding again: the parts may have been
    # packed with different scale factors.
    for var in merged.variables.values():
        var.encoding = {}

    # Write to a temporary name and rename only on success, so an
    # interrupted write never leaves a truncated raw file that a later run
    # would skip as "already exists".
    partial_file = output_file.with_name(output_file.name + ".part")
    partial_file.unlink(missing_ok=True)
    merged.to_netcdf(partial_file)
    partial_file.replace(output_file)


def _retrieve(request: dict, output_file, what: str) -> None:
    """Retrieve one CDS request into `output_file`, unless the file already exists."""
    import cdsapi  # imported here so the other steps run without cdsapi

    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        print(f"{what}: file already exists: {output_file}")
        print("Delete it first if you want to download again.")
        return

    print(f"Downloading {what} ...")
    print(f"  Variables: {request['variable']}")
    print(f"  Days:      {request['year'][0]}-{request['month'][0]}-{{{','.join(request['day'])}}}")
    print(f"  UTC hours: {request['time'][0]} to {request['time'][-1]}")
    print(f"  Output:    {output_file}")

    # Download to a temporary name and rename only on success, so an
    # interrupted download never leaves a truncated file at the final path.
    partial_file = output_file.with_name(output_file.name + ".part")
    partial_file.unlink(missing_ok=True)

    client = cdsapi.Client()
    client.retrieve("reanalysis-era5-single-levels", request, str(partial_file))
    partial_file.replace(output_file)

    print("Download completed.")


def run(cfg: RunConfig) -> None:
    """Download the raw ERA5 file and the land-sea mask, unless they already exist."""
    print(f"Region: N={cfg.north}°, W={cfg.west}°, S={cfg.south}°, E={cfg.east}°")

    raw = cfg.paths.raw_era5
    edge_request = build_edge_request(cfg)
    if raw.exists() or edge_request is None:
        _retrieve(build_request(cfg), raw, f"ERA5 for {cfg.period_label}")
    else:
        # Two requests (they can fall in different months), merged into one
        # file. Parts already downloaded are kept, so a failed run can resume.
        period_part = raw.with_name(raw.stem + "_part_period.nc")
        edge_part = raw.with_name(raw.stem + "_part_edge.nc")
        _retrieve(build_request(cfg), period_part, f"ERA5 for {cfg.period_label}")
        _retrieve(edge_request, edge_part,
                  f"ERA5 hours completing the local days (UTC{cfg.utc_offset_hours:+d})")
        merge_hourly_files([edge_part, period_part], raw)
        period_part.unlink()
        edge_part.unlink()
        print(f"Merged into {raw}")

    _retrieve(build_land_sea_mask_request(cfg), cfg.paths.raw_land_sea_mask, "ERA5 land-sea mask")

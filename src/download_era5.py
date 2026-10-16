"""
download_era5.py

Download from the Copernicus Climate Data Store (CDS), for the configured
region:
  - ERA5 hourly single-level 2m temperature and 2m dewpoint temperature;
  - the ERA5 land-sea mask (time-invariant, a single time step).

Single-period config: the configured period, plus the UTC hours of the
neighbouring day needed to complete the first (or last) local day (config
`utc_offset_hours`); both requests are merged into one raw file.

Multi-year config: one CDS request per raw file (src/config.py RawFile):
one per year and month, plus one small edge request per season for local
days east (or west) of UTC. The download is resumable: complete files
are skipped, each file is written as `<name>.part`, checked (variables,
every expected hour, region) and only then renamed, so a partial file is
never taken for a complete one. A failed file is reported and the next
one tried; re-running the download fetches what is still missing.

Before running:
    1. Register at https://cds.climate.copernicus.eu/
    2. Set up your CDS API key (see https://cds.climate.copernicus.eu/how-to-api)
    3. Accept the ERA5 licence on the dataset page.
"""

import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import xarray as xr

from src.config import RawFile, RunConfig
from src.era5_utils import (
    detect_d2m_var,
    detect_lat_coord,
    detect_lon_coord,
    detect_t2m_var,
    detect_time_coord,
    load_dataset,
)


DATASET = "reanalysis-era5-single-levels"

VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
]

# Stop a multi-year download after this many failed requests in a row:
# that usually means a problem on the CDS side or with the account
# (credentials, licence), not with one file.
MAX_CONSECUTIVE_FAILURES = 3

# Grid coordinates must match the region bounds to within this (degrees).
GRID_TOLERANCE_DEG = 1e-6


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
    Build the CDS API request for every UTC hour of a single-period config.

    Only periods within a single calendar month are supported: the CDS
    combines the year/month/day lists, so a multi-month request would also
    fetch days outside the period. Longer periods use a multi-year config
    (`period: {years, months}`), downloaded one month per request.
    """
    if cfg.is_multi_year:
        raise ValueError("Multi-year configs are downloaded per raw file: use build_raw_file_request.")
    if (cfg.start_date.year, cfg.start_date.month) != (cfg.end_date.year, cfg.end_date.month):
        raise NotImplementedError(
            f"Period {cfg.period_label} spans several months; use a multi-year config "
            "(period: {years, months}) for longer periods."
        )

    n_days = (cfg.end_date - cfg.start_date).days + 1
    days = [cfg.start_date + timedelta(days=i) for i in range(n_days)]
    return _request(cfg, VARIABLES, days, range(24))  # all 24 UTC hours


def build_edge_request(cfg: RunConfig):
    """
    Build the CDS API request for the UTC hours outside a single period
    that belong to its first or last local day, or None for UTC days.

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


def build_raw_file_request(cfg: RunConfig, raw_file: RawFile) -> dict:
    """CDS request for one raw file of a multi-year config: one year, one month."""
    days = [date(raw_file.year, raw_file.month, d) for d in raw_file.days]
    return _request(cfg, VARIABLES, days, raw_file.hours)


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

    # The parts may have been packed with different scale factors, so drop
    # the packing (values are written exactly, as decoded) but keep the
    # file compressed.
    for var in merged.variables.values():
        for key in ("scale_factor", "add_offset", "dtype"):
            var.encoding.pop(key, None)
    for name in merged.data_vars:
        merged[name].encoding = {"zlib": True, "complevel": 4}

    # Write to a temporary name and rename only on success, so an
    # interrupted write never leaves a truncated raw file that a later run
    # would skip as "already exists".
    partial_file = output_file.with_name(output_file.name + ".part")
    partial_file.unlink(missing_ok=True)
    merged.to_netcdf(partial_file)
    partial_file.replace(output_file)


def check_raw_file(path, raw_file: RawFile, cfg: RunConfig) -> None:
    """
    Raise ValueError unless `path` is a readable NetCDF file with 2m
    temperature and dewpoint at exactly the UTC hours of `raw_file`, each
    once and in order, on a grid spanning the configured region.
    """
    try:
        ds = xr.open_dataset(path)
    except Exception as err:  # e.g. a truncated download
        raise ValueError(f"{path.name}: not a readable NetCDF file ({err})") from err

    with ds:
        try:
            detect_t2m_var(ds)
            detect_d2m_var(ds)
            times = pd.DatetimeIndex(ds[detect_time_coord(ds)].values)
            lat = ds[detect_lat_coord(ds)].values
            lon = ds[detect_lon_coord(ds)].values
        except KeyError as err:
            raise ValueError(f"{path.name}: {err}") from err

    expected = raw_file.utc_times
    if not times.equals(expected):
        missing = expected.difference(times)
        extra = times.difference(expected)
        raise ValueError(
            f"{path.name}: time steps do not match the request ({len(times)} found, "
            f"{len(expected)} expected: {len(missing)} missing, {len(extra)} unexpected, "
            f"{int(times.duplicated().sum())} duplicated"
            + (f"; first missing {missing[0]}" if len(missing) else "")
            + ")."
        )

    bounds = (lat.min(), lat.max(), lon.min(), lon.max())
    if not np.allclose(bounds, (cfg.south, cfg.north, cfg.west, cfg.east), rtol=0.0, atol=GRID_TOLERANCE_DEG):
        raise ValueError(
            f"{path.name}: grid spans {bounds[0]:g}-{bounds[1]:g}°N, {bounds[2]:g}-{bounds[3]:g}°E, "
            f"not the configured region {cfg.south:g}-{cfg.north:g}°N, {cfg.west:g}-{cfg.east:g}°E."
        )


def _print_request(what: str, request: dict, output_file) -> None:
    print(f"Downloading {what} ...")
    print(f"  Variables: {request['variable']}")
    print(f"  Days:      {request['year'][0]}-{request['month'][0]}-{{{','.join(request['day'])}}}")
    print(f"  UTC hours: {request['time'][0]} to {request['time'][-1]}")
    print(f"  Output:    {output_file}")


def _retrieve_to_part(request: dict, output_file):
    """Retrieve one CDS request into `<output_file>.part` and return that path."""
    import cdsapi  # imported here so the other steps run without cdsapi

    output_file.parent.mkdir(parents=True, exist_ok=True)
    partial_file = output_file.with_name(output_file.name + ".part")
    partial_file.unlink(missing_ok=True)  # left over from an interrupted run

    client = cdsapi.Client()
    client.retrieve(DATASET, request, str(partial_file))
    return partial_file


def _retrieve(request: dict, output_file, what: str) -> None:
    """Retrieve one CDS request into `output_file`, unless the file already exists."""
    if output_file.exists():
        print(f"{what}: file already exists: {output_file}")
        print("Delete it first if you want to download again.")
        return

    _print_request(what, request, output_file)
    # Download to a temporary name and rename only on success, so an
    # interrupted download never leaves a truncated file at the final path.
    _retrieve_to_part(request, output_file).replace(output_file)
    print("Download completed.")


def download_raw_files(cfg: RunConfig) -> None:
    """
    Download every raw file of a multi-year config that is not complete yet.

    Existing files are checked (check_raw_file) and skipped when complete.
    A missing file is downloaded to `<name>.part`, checked, and renamed.
    Failures (a CDS error, a file that fails the check, an existing file
    that is incomplete) are reported and the next file is tried; existing
    files are never deleted or overwritten. Raises RuntimeError at the end
    if any file failed, or earlier after MAX_CONSECUTIVE_FAILURES failed
    requests in a row.
    """
    n_files = len(cfg.raw_files)
    downloaded, skipped, failed = 0, 0, []
    consecutive_failures = 0

    for i, raw_file in enumerate(cfg.raw_files, start=1):
        path = raw_file.path
        label = f"[{i}/{n_files}] {path.name}"

        if path.exists():
            try:
                check_raw_file(path, raw_file, cfg)
            except ValueError as err:
                print(f"{label}: existing file is not complete, not overwritten: {err}")
                failed.append((path.name, f"existing file is not complete ({err}); delete it to download again"))
            else:
                print(f"{label}: complete, skipped")
                skipped += 1
            continue

        request = build_raw_file_request(cfg, raw_file)
        _print_request(label, request, path)
        started = time.monotonic()
        try:
            partial_file = _retrieve_to_part(request, path)
            check_raw_file(partial_file, raw_file, cfg)
        except Exception as err:  # report, then try the next file (Ctrl+C still stops)
            consecutive_failures += 1
            print(f"{label}: FAILED after {time.monotonic() - started:.0f} s: {err}")
            failed.append((path.name, str(err)))
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(
                    f"Stopped after {consecutive_failures} failed downloads in a row "
                    "(CDS, network, credentials or licence problem?). "
                    f"Last error: {err}. Run the download again to resume."
                ) from err
            continue

        partial_file.replace(path)
        consecutive_failures = 0
        downloaded += 1
        print(f"{label}: done in {time.monotonic() - started:.0f} s")

    print(f"\nRaw files: {n_files} expected, {downloaded} downloaded, "
          f"{skipped} already complete, {len(failed)} failed.")
    if failed:
        raise RuntimeError(
            f"{len(failed)} raw file(s) failed; run the download again to resume:\n"
            + "\n".join(f"  {name}: {reason}" for name, reason in failed)
        )


def report_raw_files(cfg: RunConfig) -> bool:
    """
    Print which raw ERA5 files of the config are complete, missing or
    incomplete (multi-year configs: check_raw_file on every file).
    Returns True if all are complete.
    """
    if not cfg.is_multi_year:
        exists = cfg.paths.raw_era5.exists()
        print(f"{cfg.paths.raw_era5}: {'exists' if exists else 'missing'}")
        return exists

    missing, incomplete = [], []
    for raw_file in cfg.raw_files:
        if not raw_file.path.exists():
            missing.append(raw_file.path.name)
            continue
        try:
            check_raw_file(raw_file.path, raw_file, cfg)
        except ValueError as err:
            incomplete.append(str(err))

    n_files = len(cfg.raw_files)
    print(f"Raw files of {cfg.name}: {n_files - len(missing) - len(incomplete)} of {n_files} complete, "
          f"{len(missing)} missing, {len(incomplete)} incomplete.")
    for name in missing:
        print(f"  missing:    {name}")
    for reason in incomplete:
        print(f"  incomplete: {reason}")
    return not missing and not incomplete


def run(cfg: RunConfig) -> None:
    """Download the raw ERA5 file(s) and the land-sea mask, unless they already exist."""
    print(f"Region: N={cfg.north}°, W={cfg.west}°, S={cfg.south}°, E={cfg.east}°")

    if cfg.is_multi_year:
        # The small land-sea mask first: a quick check of the CDS account
        # before the long monthly downloads.
        _retrieve(build_land_sea_mask_request(cfg), cfg.paths.raw_land_sea_mask, "ERA5 land-sea mask")
        download_raw_files(cfg)
        return

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

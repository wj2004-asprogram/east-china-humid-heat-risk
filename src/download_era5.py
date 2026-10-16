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
are skipped, CDS request IDs survive process exit in data/cds_state/,
each file is written as `<name>.part`, checked (variables,
every expected hour, region) and only then renamed, so a partial file is
never taken for a complete one. A failed file is reported and the next
one tried; re-running queries the original request for each missing file.
An uncertain submission is reconciled, never blindly submitted again.
Several requests can wait in the CDS queue at once (`workers`, CLI
`--download-workers`, 1-4): each file is fetched in its own thread with
its own CDS client, and every log line starts with the file it is about.

Before running:
    1. Register at https://cds.climate.copernicus.eu/
    2. Set up your CDS API key (see https://cds.climate.copernicus.eu/how-to-api)
    3. Accept the ERA5 licence on the dataset page.
"""

import queue
import threading
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import xarray as xr

from src.config import RawFile, RunConfig, cds_state_path
from src.cds_download import CDSClient, Downloader, StateStore
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

# Requests of a multi-year download that may wait in the CDS queue at
# once. The CLI default is DEFAULT_DOWNLOAD_WORKERS; 1 fetches one file
# after the other.
MAX_DOWNLOAD_WORKERS = 4
DEFAULT_DOWNLOAD_WORKERS = 3

# Grid coordinates must match the region bounds to within this (degrees).
GRID_TOLERANCE_DEG = 1e-6

# Plausible range of ERA5 2m temperature and dewpoint (K), -93 to +67 °C,
# wider than any surface value on record; used to reject broken files.
PLAUSIBLE_KELVIN = (180.0, 340.0)


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
    once and in order, on a grid spanning the configured region, and with
    plausible values: no missing value (NaN) and every value within
    PLAUSIBLE_KELVIN. A downloaded file is renamed to its final name only if
    it passes, so a bad file is never skipped as complete later on.
    """
    try:
        ds = xr.open_dataset(path)
    except Exception as err:  # e.g. a truncated download
        raise ValueError(f"{path.name}: not a readable NetCDF file ({err})") from err

    with ds:
        try:
            variables = [detect_t2m_var(ds), detect_d2m_var(ds)]
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

        # Values (read last: the checks above need no data). ERA5 t2m/d2m
        # have no gaps, so a NaN or a value far outside Earth's surface
        # range means a broken or wrong file (e.g. one in °C).
        low_k, high_k = PLAUSIBLE_KELVIN
        for name in variables:
            values = ds[name].values
            n_missing = int(np.isnan(values).sum())
            if n_missing:
                raise ValueError(
                    f"{path.name}: {name} has {n_missing} missing (NaN) values; complete ERA5 files have none."
                )
            low, high = float(values.min()), float(values.max())
            if low < low_k or high > high_k:
                raise ValueError(
                    f"{path.name}: {name} ranges from {low:.1f} to {high:.1f}, outside the plausible "
                    f"{low_k:g}-{high_k:g} K (not in Kelvin?)."
                )


def _print_request(what: str, request: dict, output_file) -> None:
    print(f"Downloading {what} ...")
    print(f"  Variables: {request['variable']}")
    print(f"  Days:      {request['year'][0]}-{request['month'][0]}-{{{','.join(request['day'])}}}")
    print(f"  UTC hours: {request['time'][0]} to {request['time'][-1]}")
    print(f"  Output:    {output_file}")


def _download_file(request: dict, output_file, client_options: dict | None = None, validator=None):
    """
    Recover/submit a durable CDS job, transfer, validate, and publish safely.

    A new CDS client is made for every call, so threads never share one.
    """
    downloader = Downloader(StateStore(cds_state_path(output_file)), client_factory=CDSClient,
                            log=lambda message: _say("CDS", message))
    return downloader.retrieve(DATASET, request, output_file, client_options, validator)


def _retrieve(request: dict, output_file, what: str) -> None:
    """Retrieve one CDS request into `output_file`, unless the file already exists."""
    if output_file.exists():
        print(f"{what}: file already exists: {output_file}")
        print("Delete it first if you want to download again.")
        return

    _print_request(what, request, output_file)
    # Download to a temporary name and rename only on success, so an
    # interrupted download never leaves a truncated file at the final path.
    _download_file(request, output_file)
    print("Download completed.")


_print_lock = threading.Lock()


def _say(label: str, message: str) -> None:
    """Print one log line starting with `label` (the file), whole even when threads interleave."""
    with _print_lock:
        print(f"{label}: {message}", flush=True)


def _client_options(label: str) -> dict:
    """
    CDS client options for one file: no progress bar (it would garble the
    interleaved output) and the client's messages prefixed with `label`.
    """
    def report(*args, **kwargs):
        message = str(args[0]) if args else ""
        if len(args) > 1:
            try:
                message = message % args[1:]
            except (TypeError, ValueError):
                message = " ".join(str(a) for a in args)
        _say(label, f"CDS: {message}")

    return {"progress": False, "info_callback": report, "warning_callback": report, "error_callback": report}


def _fetch_raw_file(cfg: RunConfig, raw_file: RawFile, label: str) -> tuple:
    """
    Make one raw file of a multi-year config complete; runs in its own thread.

    Returns (outcome, reason) with outcome "skipped" (complete file already
    there), "downloaded", "incomplete" (an existing file fails the check; it
    is left alone) or "failed" (the request or the check of the download
    failed; only `<name>.part` is left).
    """
    path = raw_file.path
    if path.exists():
        try:
            check_raw_file(path, raw_file, cfg)
        except ValueError as err:
            _say(label, f"existing file is not complete, not overwritten: {err}")
            return "incomplete", f"existing file is not complete ({err}); delete it to download again"
        _say(label, "complete, skipped")
        return "skipped", ""

    request = build_raw_file_request(cfg, raw_file)
    _say(label, f"requesting {request['year'][0]}-{request['month'][0]}, days {request['day'][0]}-"
                f"{request['day'][-1]}, UTC hours {request['time'][0]}-{request['time'][-1]}")
    started = time.monotonic()
    try:
        _download_file(request, path, _client_options(label),
                       validator=lambda partial: check_raw_file(partial, raw_file, cfg))
    except Exception as err:  # reported by the caller, which goes on with other files
        _say(label, f"FAILED after {time.monotonic() - started:.0f} s: {err}")
        return "failed", str(err)

    # Published under the task lock only after the existing scientific checks.
    _say(label, f"done in {time.monotonic() - started:.0f} s")
    return "downloaded", ""


def download_raw_files(cfg: RunConfig, workers: int = 1) -> None:
    """
    Download every raw file of a multi-year config that is not complete yet,
    with up to `workers` requests waiting in the CDS queue at once.

    Files are started in the config's order (workers=1: one after the
    other). Existing files are checked (check_raw_file) and skipped when
    complete. A missing file is downloaded to `<name>.part`, checked, and
    renamed. Failures (a CDS error, a file that fails the check, an existing
    file that is incomplete) are reported and other files are tried;
    existing files are never deleted or overwritten. After
    MAX_CONSECUTIVE_FAILURES failed downloads in a row (in the order they
    finish) no new file is started, and the requests still running are
    allowed to finish. Raises RuntimeError at the end if any file failed.

    Each file runs in a daemon thread. Ctrl+C or killing the process stops
    local work; the durable journal retains each CDS ID/submission intent,
    and `.part` files remain available for transfer recovery.
    """
    if not 1 <= workers <= MAX_DOWNLOAD_WORKERS:
        raise ValueError(f"workers must be from 1 to {MAX_DOWNLOAD_WORKERS}, got {workers}.")

    n_files = len(cfg.raw_files)
    results = queue.Queue()  # (index, outcome, reason) from the file threads

    def fetch(index: int, raw_file: RawFile) -> None:
        label = f"[{index + 1}/{n_files}] {raw_file.path.name}"
        try:
            outcome, reason = _fetch_raw_file(cfg, raw_file, label)
        except BaseException as err:  # e.g. KeyboardInterrupt: re-raised by the main thread
            results.put((index, "interrupted", err))
        else:
            results.put((index, outcome, reason))

    downloaded, skipped, failed = 0, 0, []
    consecutive_failures, stop_error = 0, None
    next_index, running = 0, 0

    while True:
        # Start files in order while there is a free worker.
        while stop_error is None and running < workers and next_index < n_files:
            threading.Thread(
                target=fetch, args=(next_index, cfg.raw_files[next_index]),
                daemon=True, name=f"download-{cfg.raw_files[next_index].path.name}",
            ).start()
            next_index += 1
            running += 1
        if running == 0:
            break

        # Wait with a timeout, so that Ctrl+C reaches this thread promptly.
        try:
            index, outcome, reason = results.get(timeout=1.0)
        except queue.Empty:
            continue
        running -= 1
        name = cfg.raw_files[index].path.name

        if outcome == "interrupted":
            raise reason
        if outcome == "skipped":
            skipped += 1
        elif outcome == "downloaded":
            downloaded += 1
            consecutive_failures = 0
        elif outcome == "incomplete":
            failed.append((name, reason))
        else:  # "failed"
            failed.append((name, reason))
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES and stop_error is None:
                stop_error = reason
                _say("download", f"{consecutive_failures} failed downloads in a row: no new requests; "
                                 f"waiting for the {running} still running")

    print(f"\nRaw files: {n_files} expected, {downloaded} downloaded, "
          f"{skipped} already complete, {len(failed)} failed"
          + (f", {n_files - next_index} not started." if next_index < n_files else "."))
    failures = "\n".join(f"  {name}: {reason}" for name, reason in failed)
    if stop_error is not None:
        raise RuntimeError(
            f"Stopped after {MAX_CONSECUTIVE_FAILURES} failed downloads in a row "
            "(CDS, network, credentials or licence problem?). "
            f"Last error: {stop_error}. Run the download again to resume.\n{failures}"
        )
    if failed:
        raise RuntimeError(f"{len(failed)} raw file(s) failed; run the download again to resume:\n{failures}")


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


def run(cfg: RunConfig, workers: int = 1) -> None:
    """
    Download the raw ERA5 file(s) and the land-sea mask, unless they already
    exist. `workers`: requests at once for multi-year configs (see
    download_raw_files); single-period configs ignore it.
    """
    print(f"Region: N={cfg.north}°, W={cfg.west}°, S={cfg.south}°, E={cfg.east}°")

    if cfg.is_multi_year:
        # The small land-sea mask first: a quick check of the CDS account
        # before the long monthly downloads.
        _retrieve(build_land_sea_mask_request(cfg), cfg.paths.raw_land_sea_mask, "ERA5 land-sea mask")
        download_raw_files(cfg, workers)
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

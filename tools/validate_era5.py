"""Read-only ERA5 input checks (T-C3); reports never repair input files."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.config import load_config


@dataclass(frozen=True)
class Options:
    grid_step: float = 0.25
    coordinate_atol: float = 1e-6
    dewpoint_tolerance_k: float = 0.5
    temperature_min_k: float = 180.0
    temperature_max_k: float = 340.0
    hours_per_chunk: int = 24
    max_examples: int = 20


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check(checks, name, passed, message, **details):
    checks.append({'id': name, 'status': 'PASS' if passed else 'FAIL',
                   'message': message, 'details': details})


def choose(available, names):
    for name in names:
        if name in available:
            return name
    raise ValueError(f'Required name not found: {names}; available: {list(available)}')


def examples(values, options):
    values = list(values)
    return {'count': len(values), 'examples': [str(v) for v in values[:options.max_examples]],
            'examples_truncated': len(values) > options.max_examples}


def expected_hours(cfg):
    required = pd.date_range(pd.Timestamp(cfg.start_date) - pd.Timedelta(hours=cfg.utc_offset_hours),
                             pd.Timestamp(cfg.end_date) + pd.Timedelta(hours=23 - cfg.utc_offset_hours), freq='h')
    # Current producer downloads all UTC hours of the configured dates too.
    downloaded = pd.date_range(min(required[0], pd.Timestamp(cfg.start_date)),
                               max(required[-1], pd.Timestamp(cfg.end_date) + pd.Timedelta(hours=23)), freq='h')
    return required, downloaded


def grid(ds, cfg, checks, options):
    lat_name = choose(ds.coords, ('latitude', 'lat'))
    lon_name = choose(ds.coords, ('longitude', 'lon'))
    lat = np.asarray(ds[lat_name].values, dtype=float)
    lon = np.asarray(ds[lon_name].values, dtype=float)
    if lat.ndim != 1 or lon.ndim != 1 or not len(lat) or not len(lon):
        raise ValueError('Only nonempty, one-dimensional regular lat/lon coordinates are supported')
    finite = bool(np.isfinite(lat).all() and np.isfinite(lon).all())
    check(checks, 'grid.finite', finite, 'Spatial coordinates must be finite')
    if not finite:
        raise ValueError('Nonfinite spatial coordinates prevent grid/value interpretation')
    check(checks, 'grid.bounds', finite and np.allclose(
        [lat.min(), lat.max(), lon.min(), lon.max()],
        [cfg.south, cfg.north, cfg.west, cfg.east], rtol=0, atol=options.coordinate_atol),
        'Bounds equal the configured region', actual=[float(lat.min()), float(lat.max()), float(lon.min()), float(lon.max())])
    check(checks, 'grid.order', bool(np.all(np.diff(lat) < 0) and np.all(np.diff(lon) > 0)),
          'Latitude descending and longitude ascending')
    step_ok = len(lat) > 1 and len(lon) > 1 and np.allclose(np.diff(lat), -options.grid_step, rtol=0, atol=options.coordinate_atol) and np.allclose(np.diff(lon), options.grid_step, rtol=0, atol=options.coordinate_atol)
    check(checks, 'grid.spacing', bool(step_ok), 'Regular configured grid spacing', expected_degrees=options.grid_step)
    return lat_name, lon_name, lat, lon


def spatial_variable(ds, name, spatial_dims, time_name=None):
    da = ds[name]
    wanted = ([time_name] if time_name else []) + list(spatial_dims)
    extra = [dim for dim in da.dims if dim not in wanted]
    if any(da.sizes[dim] != 1 for dim in extra):
        raise ValueError(f'{name}: unsupported non-singleton dimensions {extra}')
    da = da.isel({dim: 0 for dim in extra}, drop=True)
    if set(da.dims) != set(wanted):
        raise ValueError(f'{name}: expected dimensions {wanted}, got {da.dims}')
    return da.transpose(*wanted)


def inspect_hourly(ds, cfg, entry, options, expected_times=None):
    checks = entry['checks']
    lat_name, lon_name, lat, lon = grid(ds, cfg, checks, options)
    time_name = choose(ds.coords, ('valid_time', 'time'))
    if not np.issubdtype(ds[time_name].dtype, np.datetime64):
        raise ValueError('Time must decode to datetime64 UTC values; numeric times are not inferred')
    times = pd.DatetimeIndex(ds[time_name].values)
    if not len(times):
        raise ValueError('Empty hourly time axis')
    required, downloaded = expected_hours(cfg) if expected_times is None else expected_times
    valid = times[~times.isna()]
    duplicates = valid[valid.duplicated(keep=False)].unique()
    off_hour = valid[valid != valid.floor('h')]
    missing = required.difference(valid)
    missing_layout = downloaded.difference(valid)
    outside_layout = valid.difference(downloaded)
    check(checks, 'time.datetime', not times.isna().any(), 'No NaT timestamps')
    check(checks, 'time.unique', len(duplicates) == 0, 'No repeated timestamps', **examples(duplicates, options))
    check(checks, 'time.on_hour', len(off_hour) == 0, 'Every timestamp is on a full UTC hour', **examples(off_hour, options))
    check(checks, 'time.sorted', bool(times.is_monotonic_increasing), 'Timestamp order is increasing')
    check(checks, 'time.required_coverage', len(missing) == 0,
          'All hours needed by local calendar days are present' if expected_times is None else 'All explicit required UTC hours are present',
          **examples(missing, options))
    check(checks, 'time.download_layout', len(missing_layout) == 0 and len(outside_layout) == 0,
          'Current producer layout, including permitted edge hours' if expected_times is None else 'Explicit physical UTC file layout',
          missing=examples(missing_layout, options), unexpected=examples(outside_layout, options))
    entry['time'] = {'observed_hours': len(times), 'required_hours': len(required),
                     'producer_hours': len(downloaded), 'producer_extra_hours': len(downloaded.difference(required)),
                     'first_utc': str(times[0]), 'last_utc': str(times[-1]), 'utc_offset_hours': cfg.utc_offset_hours}
    if expected_times is not None:
        entry['time']['observed_utc'] = [str(t) for t in times]
        entry['time']['expectation_mode'] = 'explicit_utc'
    entry['variables'] = {}
    arrays = {}
    for canonical, aliases in [('t2m', ('t2m', '2m_temperature')), ('d2m', ('d2m', '2m_dewpoint_temperature'))]:
        name = choose(ds.data_vars, aliases)
        da = spatial_variable(ds, name, (lat_name, lon_name), time_name)
        units = str(da.attrs.get('units', '')).strip().lower()
        check(checks, f'{canonical}.units', units in {'k', 'kelvin', 'kelvins'}, 'Temperature units must explicitly be Kelvin', actual=units)
        nan_time = np.zeros(len(times), dtype=np.int64)
        nan_grid = np.zeros((len(lat), len(lon)), dtype=np.int64)
        minimum, maximum, infinities = float('inf'), float('-inf'), 0
        for first in range(0, len(times), options.hours_per_chunk):
            values = np.asarray(da.isel({time_name: slice(first, first + options.hours_per_chunk)}).values, dtype=float)
            mask = np.isnan(values)
            nan_time[first:first + len(values)] = mask.sum(axis=(1, 2))
            nan_grid += mask.sum(axis=0)
            infinities += int(np.isinf(values).sum())
            finite = values[np.isfinite(values)]
            if finite.size:
                minimum = min(minimum, float(finite.min()))
                maximum = max(maximum, float(finite.max()))
        locations = np.argwhere(nan_grid > 0)
        summary = {'name': name, 'units': units, 'nan_count': int(nan_time.sum()), 'infinite_count': infinities,
                   'minimum_k': minimum if np.isfinite(minimum) else None,
                   'maximum_k': maximum if np.isfinite(maximum) else None,
                   'by_time': [{'utc': str(times[i]), 'nan_cells': int(nan_time[i])} for i in np.flatnonzero(nan_time)],
                   'by_grid': [{'latitude': float(lat[i]), 'longitude': float(lon[j]), 'nan_hours': int(nan_grid[i, j])} for i, j in locations],
                   'coordinate_names': {'time': time_name, 'latitude': lat_name, 'longitude': lon_name}}
        entry['variables'][canonical] = summary
        check(checks, f'{canonical}.missing', summary['nan_count'] == 0, 'Missing-value counts with per-time and per-grid locations', nan_count=summary['nan_count'])
        check(checks, f'{canonical}.finite', infinities == 0, 'No infinite values', count=infinities)
        check(checks, f'{canonical}.range', np.isfinite(minimum) and options.temperature_min_k <= minimum <= maximum <= options.temperature_max_k,
              'Configurable engineering sanity limits, not a climatological claim', minimum_k=summary['minimum_k'], maximum_k=summary['maximum_k'],
              lower_k=options.temperature_min_k, upper_k=options.temperature_max_k)
        arrays[canonical] = da
    exceed_time = np.zeros(len(times), dtype=np.int64)
    exceed_grid = np.zeros((len(lat), len(lon)), dtype=np.int64)
    max_difference = None
    for first in range(0, len(times), options.hours_per_chunk):
        selection = {time_name: slice(first, first + options.hours_per_chunk)}
        difference = np.asarray(arrays['d2m'].isel(selection).values, dtype=float) - np.asarray(arrays['t2m'].isel(selection).values, dtype=float)
        exceed = np.isfinite(difference) & (difference > options.dewpoint_tolerance_k)
        exceed_time[first:first + len(difference)] = exceed.sum(axis=(1, 2))
        exceed_grid += exceed.sum(axis=0)
        finite = difference[np.isfinite(difference)]
        if finite.size:
            max_difference = max(float(finite.max()), max_difference if max_difference is not None else -np.inf)
    bad_cells = np.argwhere(exceed_grid > 0)
    check(checks, 'dewpoint.relationship', not exceed_time.any(), 'Dewpoint <= air temperature plus explicit tolerance',
          tolerance_k=options.dewpoint_tolerance_k, max_difference_k=max_difference,
          offending_samples=int(exceed_time.sum()),
          by_time=[{'utc': str(times[i]), 'cells': int(exceed_time[i])} for i in np.flatnonzero(exceed_time)],
          by_grid=[{'latitude': float(lat[i]), 'longitude': float(lon[j]), 'hours': int(exceed_grid[i, j])} for i, j in bad_cells])
    return lat, lon


def inspect_mask(ds, cfg, entry, options):
    checks = entry['checks']
    lat_name, lon_name, lat, lon = grid(ds, cfg, checks, options)
    name = choose(ds.data_vars, ('lsm', 'land_sea_mask'))
    source = ds[name]
    extra_dims = {dim: source.sizes[dim] for dim in source.dims if dim not in (lat_name, lon_name)}
    check(checks, 'mask.single_time', all(size == 1 for size in extra_dims.values()), 'Static mask or a single time step', non_spatial_dimensions=extra_dims)
    da = spatial_variable(ds, name, (lat_name, lon_name))
    values = np.asarray(da.values, dtype=float)
    finite = np.isfinite(values)
    bounded = bool(finite.all() and ((values >= 0) & (values <= 1)).all())
    check(checks, 'mask.values', bounded, 'All mask fractions finite and in [0, 1]', invalid_cells=int((~finite | (values < 0) | (values > 1)).sum()))
    weights = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))
    entry['mask'] = {'threshold': cfg.land_sea_mask_threshold,
                     'land_percent': float(100 * ((values >= cfg.land_sea_mask_threshold) * weights).sum() / weights.sum()),
                     'minimum': float(values[finite].min()) if finite.any() else None,
                     'maximum': float(values[finite].max()) if finite.any() else None}
    return lat, lon


def inspect_file(path, cfg, role, options, expected_times=None):
    path = Path(path).resolve()
    entry = {'path': str(path), 'role': role, 'checks': [], 'sha256_before': None, 'sha256_after': None}
    coordinates = None
    if not path.is_file():
        entry['checks'].append({'id': 'file.available', 'status': 'NOT_RUN', 'message': 'Required final file is unavailable', 'details': {}})
        return entry, coordinates
    before = path.stat()
    try:
        entry['sha256_before'] = sha256(path)
        with xr.open_dataset(path) as ds:
            inspector = inspect_hourly if role == 'hourly' else inspect_mask
            coordinates = (inspector(ds, cfg, entry, options, expected_times=expected_times)
                           if role == 'hourly' else inspector(ds, cfg, entry, options))
    except Exception as error:
        entry['checks'].append({'id': 'file.inspect', 'status': 'ERROR', 'message': f'{type(error).__name__}: {error}', 'details': {}})
    finally:
        try:
            entry['sha256_after'] = sha256(path)
            after = path.stat()
            unchanged = entry['sha256_before'] is not None and entry['sha256_before'] == entry['sha256_after'] and before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
            check(entry['checks'], 'file.unchanged', unchanged, 'SHA-256, size and modification time unchanged while reading', size_bytes=after.st_size)
        except OSError as error:
            entry['checks'].append({'id': 'file.unchanged', 'status': 'ERROR', 'message': str(error), 'details': {}})
    return entry, coordinates


def inventory(configs, options):
    groups = defaultdict(Counter)
    for cfg in configs:
        region = (cfg.north, cfg.west, cfg.south, cfg.east, cfg.utc_offset_hours)
        for day in pd.date_range(cfg.start_date, cfg.end_date, freq='D'):
            groups[(region, day.year)][day] += 1
    checks = []
    for (region, year), dates in groups.items():
        # Continuity within each configured year; no invented winter coverage.
        expected = pd.date_range(min(dates), max(dates), freq='D')
        missing = expected.difference(pd.DatetimeIndex(dates))
        duplicates = [day for day, count in dates.items() if count > 1]
        check(checks, f'inventory.{year}.{region}', len(missing) == 0 and len(duplicates) == 0,
              'Configured local dates continuous within each year, each once; different years independent',
              year=year, region_and_offset=list(region), local_days=len(dates), missing=examples(missing, options), duplicate=examples(duplicates, options))
    return checks


def validate(config_paths, raw_dir=None, options=None):
    options = options or Options()
    numeric = (options.grid_step, options.coordinate_atol, options.dewpoint_tolerance_k,
               options.temperature_min_k, options.temperature_max_k)
    if not all(np.isfinite(value) for value in numeric) or not isinstance(options.hours_per_chunk, int) or not isinstance(options.max_examples, int) or options.hours_per_chunk < 1 or options.max_examples < 1 or options.grid_step <= 0 or options.coordinate_atol < 0 or options.dewpoint_tolerance_k < 0 or options.temperature_min_k >= options.temperature_max_k:
        raise ValueError('Invalid validation options')
    configs = [load_config(path) for path in config_paths]
    report = {'schema_version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'mode': 'read_only', 'config_paths': [str(Path(p).resolve()) for p in config_paths],
              'options': asdict(options), 'files': [], 'checks': inventory(configs, options)}
    masks = {}
    for cfg in configs:
        folder = Path(raw_dir) if raw_dir is not None else cfg.paths.raw_era5.parent
        hourly_path = folder / cfg.paths.raw_era5.name
        mask_path = folder / cfg.paths.raw_land_sea_mask.name
        entry, coords = inspect_file(hourly_path, cfg, 'hourly', options)
        entry['config_name'] = cfg.name
        report['files'].append(entry)
        mask_key = (str(mask_path.resolve()), cfg.land_sea_mask_threshold)
        if mask_key not in masks:
            mask_entry, mask_coords = inspect_file(mask_path, cfg, 'land_sea_mask', options)
            masks[mask_key] = mask_entry, mask_coords
            report['files'].append(mask_entry)
        _, mask_coords = masks[mask_key]
        if coords is None or mask_coords is None:
            entry['checks'].append({'id': 'grid.mask_match', 'status': 'NOT_RUN', 'message': 'Raw or mask grid unavailable; alignment not checked', 'details': {}})
        else:
            same = all(a.shape == b.shape and np.allclose(a, b, rtol=0, atol=options.coordinate_atol) for a, b in zip(coords, mask_coords))
            check(entry['checks'], 'grid.mask_match', same, 'Every raw grid coordinate matches mask without resampling')
    statuses = Counter(item['status'] for item in report['checks'] + [c for f in report['files'] for c in f['checks']])
    overall = 'FAIL' if statuses['FAIL'] else 'ERROR' if statuses['ERROR'] else 'INCOMPLETE' if statuses['NOT_RUN'] else 'PASS'
    report['summary'] = {'status': overall, 'counts': {status: statuses[status] for status in ('PASS', 'FAIL', 'ERROR', 'NOT_RUN')}}
    return report


def markdown(report):
    lines = ['# ERA5 input validation', '', f"Result: **{report['summary']['status']}**", '',
             f"Created (UTC): {report['created_at_utc']}", '', '## Check counts', '',
             json.dumps(report['summary']['counts']), '', '## Inputs', '']
    for entry in report['files']:
        lines += [f"### {Path(entry['path']).name}", '', f"Path: `{entry['path']}`", '',
                  f"SHA-256 before: `{entry['sha256_before']}`", '', f"SHA-256 after: `{entry['sha256_after']}`", '']
        if 'time' in entry:
            lines += [f"Hours observed / required / producer: {entry['time']['observed_hours']} / {entry['time']['required_hours']} / {entry['time']['producer_hours']}", '']
        if 'mask' in entry:
            lines += [f"Land fraction (%): {entry['mask']['land_percent']:.6f}", '']
        lines += ['| Check | Status | Description |', '|---|---|---|']
        lines += [f"| {c['id']} | {c['status']} | {c['message'].replace('|', '/')} |" for c in entry['checks']]
        lines += ['']
        for c in entry['checks']:
            if c['status'] != 'PASS':
                lines += [f"{c['id']}: `{json.dumps(c['details'], ensure_ascii=False)}`", '']
    lines += ['## Coverage inventory', '']
    lines += [f"- {c['status']} {c['id']}: {json.dumps(c['details'], ensure_ascii=False)}" for c in report['checks']]
    lines += ['', 'Detailed per-variable missing-time/grid summaries and options are in the JSON report.', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', nargs='+', required=True, help='One or more existing YAML configs')
    parser.add_argument('--raw-dir', type=Path, help='Read-only raw input directory; no discovery/download')
    parser.add_argument('--output', type=Path, required=True, help='New JSON report path; Markdown companion also written')
    parser.add_argument('--grid-step', type=float, default=0.25)
    parser.add_argument('--dewpoint-tolerance-k', type=float, default=0.5)
    parser.add_argument('--temperature-min-k', type=float, default=180.)
    parser.add_argument('--temperature-max-k', type=float, default=340.)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.suffix.lower() != '.json':
        parser.error('--output must end with .json')
    # Guard both report files before reading/writing; never overwrite a report.
    targets = [output, output.with_suffix('.md')]
    try:
        configs = [load_config(path) for path in args.config]
    except (OSError, ValueError, KeyError) as error:
        parser.error(f'Invalid configuration: {error}')
    folders = [(args.raw_dir or cfg.paths.raw_era5.parent).resolve() for cfg in configs]
    if any(target.exists() or any(target.is_relative_to(folder) for folder in folders) for target in targets):
        parser.error('Reports must be new files outside every raw input directory')
    options = Options(grid_step=args.grid_step, dewpoint_tolerance_k=args.dewpoint_tolerance_k,
                      temperature_min_k=args.temperature_min_k, temperature_max_k=args.temperature_max_k)
    try:
        report = validate(args.config, args.raw_dir, options)
    except ValueError as error:
        parser.error(str(error))
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create also prevents an accidental concurrent overwrite.
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    with output.with_suffix('.md').open('x', encoding='utf-8') as stream:
        stream.write(markdown(report))
    print(json.dumps(report['summary']))
    return {'PASS': 0, 'FAIL': 1, 'ERROR': 2, 'INCOMPLETE': 2}[report['summary']['status']]


if __name__ == '__main__':
    raise SystemExit(main())

"""Reproduce four M1 audit states using only new synthetic input files."""

import argparse
import calendar
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.config import load_config
from tools.validate_batch import audit, create_manifest, markdown, write_json_new


def make_inputs(root, month):
    config = root / 'configs' / f'2022-{month:02d}.yaml'
    if not config.exists():
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(f'''name: synthetic_2022_{month:02d}
period: {{start: "2022-{month:02d}-01", end: "2022-{month:02d}-{calendar.monthrange(2022, month)[1]}"}}
region: {{north: 30.5, west: 118, south: 30, east: 118.5}}
cities: []
utc_offset_hours: 8
land_sea_mask_threshold: 0.5
file_tag: synthetic_2022_{month:02d}
figures_dir: figures/synthetic
''', encoding='utf-8')
    return config


def write_month(root, month):
    cfg = load_config(make_inputs(root, month))
    raw, daily = root / 'raw', root / 'daily'
    raw.mkdir(exist_ok=True)
    daily.mkdir(exist_ok=True)
    times = pd.date_range(pd.Timestamp(cfg.start_date) - pd.Timedelta(hours=8),
                          pd.Timestamp(cfg.end_date) + pd.Timedelta(hours=23), freq='h')
    coords = {'time': times, 'latitude': [30.5, 30.25, 30.], 'longitude': [118., 118.25, 118.5]}
    shape = (len(times), 3, 3)
    xr.Dataset({name: (('time', 'latitude', 'longitude'), np.full(shape, value, dtype='float32'), {'units': 'K'})
                for name, value in [('t2m', 300.), ('d2m', 295.)]}, coords=coords).to_netcdf(raw / cfg.paths.raw_era5.name)
    mask = raw / cfg.paths.raw_land_sea_mask.name
    if not mask.exists():
        xr.Dataset({'lsm': (('latitude', 'longitude'), np.ones((3, 3), dtype='float32'))},
                   coords={k: coords[k] for k in ['latitude', 'longitude']}).to_netcdf(mask)
    days = pd.date_range(cfg.start_date, cfg.end_date, freq='D')
    xr.Dataset({'tmax': (('day', 'latitude', 'longitude'), np.full((len(days), 3, 3), 300.), {'units': 'K'})},
               coords={'day': days, 'latitude': coords['latitude'], 'longitude': coords['longitude']}).to_netcdf(daily / f'2022-{month:02d}.nc')


def run(output_dir):
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    configs = {month: make_inputs(root, month) for month in [6, 7, 8]}
    plan = create_manifest(configs[6], [2022], [6, 7, 8])
    plan['bindings'] = [{'period': f'2022-{m:02d}', 'config': f'configs/2022-{m:02d}.yaml', 'raw_dir': 'raw',
                         'daily': {'path': f'daily/2022-{m:02d}.nc', 'time_coord': 'day', 'variables': {'tmax': 'K'}}}
                        for m in [6, 7, 8]]
    manifest = root / 'manifest.json'
    write_json_new(manifest, plan)
    write_month(root, 6)
    write_month(root, 8)
    cfg = load_config(configs[7])
    (root / 'raw' / (cfg.paths.raw_era5.name + '.part')).write_bytes(b'incomplete synthetic download')
    reports = root / 'reports'
    previous = None
    results = []
    for index, name in enumerate(['missing-july', 'completed', 'unchanged', 'changed-accepted-input']):
        if index == 1:
            write_month(root, 7)
        if index == 3:
            raw_path = root / 'raw' / load_config(configs[6]).paths.raw_era5.name
            with xr.open_dataset(raw_path) as source:
                changed = source.load()
            changed.t2m.values[0, 0, 0] += .25
            changed.to_netcdf(raw_path)  # Deliberately change this demo's own synthetic fixture.
        report = audit(manifest, previous)
        path = reports / f'{index + 1}-{name}.json'
        write_json_new(path, report)
        path.with_suffix('.md').write_text(markdown(report), encoding='utf-8')
        results.append({'case': name, 'summary': report['summary'],
                        'resume_states': {r['period']: r['resume_state'] for r in report['matrix']}})
        previous = path
    statuses = [r['summary']['status'] for r in results]
    if statuses != ['INCOMPLETE', 'PASS', 'PASS', 'FAIL']:
        raise AssertionError(statuses)
    write_json_new(root / 'demo-summary.json', {'synthetic_only': True, 'cases': results})
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True, help='Must not already exist')
    args = parser.parse_args(argv)
    try:
        for result in run(args.output_dir):
            print(f"{result['case']}: {result['summary']['status']}")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

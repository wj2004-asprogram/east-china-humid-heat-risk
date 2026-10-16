"""Small independent physical-file fixtures; no credentials or network."""

import calendar
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.config import load_config


def make_config(root, offset=8, first=2003, last=2004, months=(6, 7, 8)):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    config = root / f'multi_{offset}.yaml'
    config.write_text(f'''name: independent_{offset}
period:
  years: {{first: {first}, last: {last}}}
  months: {list(months)}
region: {{north: 30.5, west: 118, south: 30, east: 118.5}}
cities:
  - {{name: TestCity, lat: 30.5, lon: 118, color: blue}}
utc_offset_hours: {offset}
land_sea_mask_threshold: 0.5
file_tag: independent_{offset}
figures_dir: figures/independent_{offset}
''', encoding='utf-8')
    return config, load_config(config, project_root=root)


def hourly(times):
    times = pd.DatetimeIndex(times)
    values = np.broadcast_to((300 + times.hour.to_numpy() / 10)[:, None, None], (len(times), 3, 3)).copy()
    # Literal markers at UTC hours that straddle Beijing month/season edges.
    for month, day, hour, value in [(5, 31, 16, 320), (6, 30, 15, 321),
                                     (6, 30, 16, 322), (7, 31, 16, 323), (8, 31, 16, 324)]:
        selected = (times.month == month) & (times.day == day) & (times.hour == hour)
        values[selected, 0, 0] = value
    return xr.Dataset({'t2m': (('valid_time', 'latitude', 'longitude'), values.astype('float32'), {'units': 'K'}),
                       'd2m': (('valid_time', 'latitude', 'longitude'), np.full(values.shape, 295., dtype='float32'), {'units': 'K'})},
                      coords={'valid_time': times, 'latitude': [30.5, 30.25, 30.], 'longitude': [118., 118.25, 118.5]})


def write_mask(cfg):
    cfg.paths.raw_land_sea_mask.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({'lsm': (('latitude', 'longitude'), np.ones((3, 3), dtype='float32'))},
               coords={'latitude': [30.5, 30.25, 30.], 'longitude': [118., 118.25, 118.5]}).to_netcdf(cfg.paths.raw_land_sea_mask)


def write_inputs(cfg, omit=()):
    """Generate calendar files without using production raw-file hour helpers."""
    folder = cfg.paths.raw_land_sea_mask.parent
    folder.mkdir(parents=True, exist_ok=True)
    covered = {(y, m) for y in cfg.years for m in cfg.months}
    for year in cfg.years:
        for month in cfg.months:
            name = f'era5_east_china_{year:04d}_{month:02d}.nc'
            if name not in omit:
                times = pd.date_range(f'{year}-{month:02d}-01',
                                      f'{year}-{month:02d}-{calendar.monthrange(year, month)[1]} 23:00', freq='h')
                hourly(times).to_netcdf(folder / name)
        if cfg.utc_offset_hours == 8:
            day = date(year, cfg.months[0], 1) - timedelta(days=1)
            name = f'era5_east_china_{day:%Y_%m_%d}_h16-23.nc'
            if (day.year, day.month) not in covered and name not in omit:
                hourly(pd.date_range(f'{day} 16:00', f'{day} 23:00', freq='h')).to_netcdf(folder / name)
    write_mask(cfg)


def rewrite(path, transform):
    with xr.open_dataset(path) as source:
        ds = source.load()
    transform(ds).to_netcdf(path)

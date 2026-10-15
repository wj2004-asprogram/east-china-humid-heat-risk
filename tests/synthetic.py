"""Test helper: a small synthetic ERA5-like dataset."""

import numpy as np
import pandas as pd
import xarray as xr


def make_synthetic_era5(start: str, end: str, step: float = 0.5, seed: int = 0) -> xr.Dataset:
    """
    Build a small hourly dataset laid out like a new-CDS ERA5 download.

    Dimensions (valid_time, latitude, longitude), latitude descending,
    t2m/d2m in Kelvin as float32. Temperatures (~10-45 °C) and dewpoint
    depressions (1-18 K) are spread wide enough to reach every heat index
    branch and most risk categories.
    """
    times = pd.date_range(start, f"{end} 23:00", freq="h")
    lats = np.arange(36.0, 25.0 - 1e-9, -step)
    lons = np.arange(112.0, 123.0 + 1e-9, step)
    shape = (len(times), len(lats), len(lons))

    rng = np.random.default_rng(seed)
    diurnal = np.sin(2 * np.pi * times.hour.values / 24.0)  # peak at 06 UTC (14 Beijing time)
    cell_offset = rng.uniform(-10.0, 5.0, shape[1:])  # cool and hot grid cells
    t2m_c = 28.0 + 6.0 * diurnal[:, None, None] + cell_offset + rng.normal(0.0, 2.0, shape)
    d2m_c = t2m_c - rng.uniform(1.0, 18.0, shape)

    dims = ("valid_time", "latitude", "longitude")
    return xr.Dataset(
        {
            "t2m": (dims, (t2m_c + 273.15).astype(np.float32), {"units": "K"}),
            "d2m": (dims, (d2m_c + 273.15).astype(np.float32), {"units": "K"}),
        },
        coords={
            "valid_time": times,
            "latitude": lats,
            "longitude": lons,
            "number": 0,
            "expver": ("valid_time", np.full(len(times), "0001")),
        },
    )

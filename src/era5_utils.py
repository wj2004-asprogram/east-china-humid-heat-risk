"""
era5_utils.py

Detect coordinate and variable names in ERA5 NetCDF files.

ERA5 files from the CDS do not always use the same names: the classic CDS
API delivers "time", the new CDS delivers "valid_time"; variables may use
short names ("t2m") or long names ("2m_temperature").
"""

import xarray as xr


def _first_present(candidates, available, what: str) -> str:
    """Return the first candidate found in `available`, or raise a clear error."""
    for candidate in candidates:
        if candidate in available:
            return candidate
    raise KeyError(
        f"Cannot find {what} (tried {', '.join(repr(c) for c in candidates)}). "
        f"Available: {list(available)}"
    )


def detect_time_coord(ds: xr.Dataset) -> str:
    """Detect the hourly time coordinate name ('time' or 'valid_time')."""
    return _first_present(["time", "valid_time"], ds.coords, "a time coordinate")


def detect_lat_coord(ds) -> str:
    """Detect the latitude coordinate name ('latitude' or 'lat')."""
    return _first_present(["latitude", "lat"], ds.coords, "a latitude coordinate")


def detect_lon_coord(ds) -> str:
    """Detect the longitude coordinate name ('longitude' or 'lon')."""
    return _first_present(["longitude", "lon"], ds.coords, "a longitude coordinate")


def detect_date_dim(da) -> str:
    """
    Detect the daily time dimension name.

    Prefers 'date' (used for daily variables in this workflow) over 'time'.
    """
    return _first_present(["date", "time"], da.dims, "a daily time dimension")


def detect_t2m_var(ds: xr.Dataset) -> str:
    """Detect the 2m temperature variable name ('t2m' or '2m_temperature')."""
    return _first_present(
        ["t2m", "2m_temperature"], ds.data_vars, "a 2m temperature variable"
    )


def detect_d2m_var(ds: xr.Dataset) -> str:
    """Detect the 2m dewpoint variable name ('d2m' or '2m_dewpoint_temperature')."""
    return _first_present(
        ["d2m", "2m_dewpoint_temperature"], ds.data_vars, "a 2m dewpoint variable"
    )


def load_dataset(path) -> xr.Dataset:
    """
    Read a NetCDF file fully into memory and close it.

    Closing straight away matters on Windows: an open handle would block a
    later step (or a re-run) from overwriting the same file.
    """
    with xr.open_dataset(path) as ds:
        return ds.load()


def select_city_points(data, cities, lat_name: str, lon_name: str):
    """
    Select the nearest grid point for each city.

    Parameters
    ----------
    data : xr.Dataset or xr.DataArray
        Gridded data with latitude/longitude coordinates.
    cities : iterable of City
        Cities from the run configuration.

    Returns
    -------
    list of (City, xr.Dataset or xr.DataArray, matched_lat, matched_lon)
    """
    points = []
    for city in cities:
        point = data.sel({lat_name: city.lat, lon_name: city.lon}, method="nearest")
        actual_lat = float(point[lat_name].values)
        actual_lon = float(point[lon_name].values)
        print(f"{city.name:>10s}: requested ({city.lat:.2f}°N, {city.lon:.2f}°E)  "
              f"→ matched ({actual_lat:.2f}°N, {actual_lon:.2f}°E)")
        points.append((city, point, actual_lat, actual_lon))
    return points


def require_var(ds: xr.Dataset, var_name: str) -> xr.DataArray:
    """Return `ds[var_name]`, with a clear error if the variable is missing."""
    if var_name not in ds.data_vars:
        raise KeyError(
            f"Variable '{var_name}' not found. Available: {list(ds.data_vars)}"
        )
    return ds[var_name]

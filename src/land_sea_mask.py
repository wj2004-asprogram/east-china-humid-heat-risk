"""
land_sea_mask.py

Read the ERA5 land-sea mask and turn it into a boolean land mask on the
grid of the processed data.

ERA5 "land_sea_mask" (short name "lsm") is the fraction of land in each
grid cell, from 0 (sea) to 1 (land). It does not change in time, so the
download holds a single time step.
"""

import xarray as xr

from src.era5_utils import detect_lat_coord, detect_lon_coord, detect_lsm_var


def load_land_mask(path, threshold: float, like: xr.DataArray, lat_name: str, lon_name: str) -> xr.DataArray:
    """
    Return a boolean land mask (True = land) on the grid of `like`.

    A grid cell is land where the ERA5 land-sea mask is >= `threshold`.

    Parameters
    ----------
    path : str or Path
        ERA5 land-sea mask NetCDF file (one time step).
    threshold : float
        Land fraction above which a cell counts as land, e.g. 0.5.
    like : xr.DataArray
        Data whose latitude/longitude grid the mask must match.
    lat_name, lon_name : str
        Latitude/longitude coordinate names of `like`.

    Raises
    ------
    KeyError
        If a grid point of `like` is not on the land-sea mask grid
        (e.g. the mask was downloaded for another region).
    """
    with xr.open_dataset(path) as ds:
        lsm = ds[detect_lsm_var(ds)].load()
        mask_lat = detect_lat_coord(ds)
        mask_lon = detect_lon_coord(ds)

    # Drop the single time step and any other non-spatial dimension.
    extra_dims = [d for d in lsm.dims if d not in (mask_lat, mask_lon)]
    lsm = lsm.isel({d: 0 for d in extra_dims}, drop=True)
    renames = {old: new for old, new in [(mask_lat, lat_name), (mask_lon, lon_name)] if old != new}
    lsm = lsm.rename(renames)

    # Same grid as the data: match every grid point to within 1e-6 degrees.
    lsm = lsm.sel(
        {lat_name: like[lat_name], lon_name: like[lon_name]},
        method="nearest",
        tolerance=1e-6,
    )

    land = (lsm >= threshold).reset_coords(drop=True)
    land.name = "land_mask"
    return land

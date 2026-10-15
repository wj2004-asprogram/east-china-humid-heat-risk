"""Tests for reading the ERA5 land-sea mask onto the data grid."""

import numpy as np
import pytest

from src.land_sea_mask import load_land_mask
from tests.synthetic import make_synthetic_era5, make_synthetic_land_sea_mask


@pytest.fixture
def lsm_file(tmp_path):
    path = tmp_path / "lsm.nc"
    make_synthetic_land_sea_mask("2022-07-01").to_netcdf(path)
    return path


@pytest.fixture
def grid():
    return make_synthetic_era5("2022-07-01", "2022-07-01")["t2m"]


def test_land_mask_threshold_and_grid(lsm_file, grid):
    land = load_land_mask(lsm_file, 0.5, grid, "latitude", "longitude")

    assert land.dims == ("latitude", "longitude")
    assert land.dtype == bool
    np.testing.assert_array_equal(land["latitude"], grid["latitude"])
    np.testing.assert_array_equal(land["longitude"], grid["longitude"])
    # lsm = clip(122 - lon, 0, 1): land up to 121.5 °E (lsm = 0.5 >= 0.5), sea east of it.
    is_land_column = land.all("latitude")
    assert is_land_column.sel(longitude=121.5)
    assert not is_land_column.sel(longitude=[122.0, 122.5, 123.0]).any()
    assert int(land.sum()) == land.sizes["latitude"] * (land.sizes["longitude"] - 3)


def test_land_mask_stricter_threshold(lsm_file, grid):
    land = load_land_mask(lsm_file, 0.9, grid, "latitude", "longitude")
    assert not land.sel(longitude=121.5).any()  # lsm = 0.5 < 0.9
    assert land.sel(longitude=121.0).all()      # lsm = 1.0


def test_land_mask_on_another_grid_is_an_error(lsm_file, grid):
    shifted = grid.assign_coords(longitude=grid["longitude"] + 0.25)
    with pytest.raises(KeyError):
        load_land_mask(lsm_file, 0.5, shifted, "latitude", "longitude")

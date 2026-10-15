"""Tests for the CDS requests and the merge of downloaded parts (no network access)."""

import dataclasses
from datetime import date
from pathlib import Path

import pytest
import xarray as xr

from src.config import PROJECT_ROOT, load_config
from src.download_era5 import VARIABLES, build_edge_request, merge_hourly_files
from tests.synthetic import make_synthetic_era5


JULY_BJT = load_config(PROJECT_ROOT / "configs" / "july_2022_bjt.yaml")


def test_edge_request_for_beijing_time():
    # Beijing day 2022-07-01 starts at 16:00 UTC on 2022-06-30.
    request = build_edge_request(JULY_BJT)
    assert request["variable"] == VARIABLES
    assert (request["year"], request["month"], request["day"]) == (["2022"], ["06"], ["30"])
    assert request["time"] == [f"{h}:00" for h in range(16, 24)]
    assert request["area"] == [36, 112, 25, 123]


def test_edge_request_across_a_year_boundary():
    # Beijing days from 2023-01-01 need 16:00-23:00 UTC on 2022-12-31.
    january = dataclasses.replace(JULY_BJT, start_date=date(2023, 1, 1), end_date=date(2023, 1, 7))
    request = build_edge_request(january)
    assert (request["year"], request["month"], request["day"]) == (["2022"], ["12"], ["31"])
    assert request["time"][0] == "16:00" and request["time"][-1] == "23:00"


def test_edge_request_west_of_utc_and_utc():
    west = build_edge_request(dataclasses.replace(JULY_BJT, utc_offset_hours=-5))
    assert (west["year"], west["month"], west["day"]) == (["2022"], ["08"], ["01"])
    assert west["time"] == ["00:00", "01:00", "02:00", "03:00", "04:00"]

    assert build_edge_request(dataclasses.replace(JULY_BJT, utc_offset_hours=0)) is None


def test_merge_hourly_files(tmp_path):
    full = make_synthetic_era5("2022-07-01", "2022-07-01", hours_before=8)
    edge, period = tmp_path / "edge.nc", tmp_path / "period.nc"
    full.isel(valid_time=slice(0, 8)).to_netcdf(edge)
    full.isel(valid_time=slice(8, None)).to_netcdf(period)

    merged_file = tmp_path / "merged.nc"
    merge_hourly_files([period, edge], merged_file)  # order of parts does not matter

    with xr.open_dataset(merged_file) as merged:
        assert merged.sizes["valid_time"] == 8 + 24
        xr.testing.assert_equal(merged[["t2m", "d2m"]], full[["t2m", "d2m"]])
        for name in ("t2m", "d2m"):
            # Exact values (same dtype, no packing), compressed on disk.
            assert merged[name].dtype == full[name].dtype
            assert "scale_factor" not in merged[name].encoding
            assert merged[name].encoding.get("zlib") is True


def test_interrupted_merge_leaves_no_raw_file(tmp_path, monkeypatch):
    # A failed write must not leave a (truncated) file at the final path,
    # otherwise the next download run would skip it as "already exists".
    full = make_synthetic_era5("2022-07-01", "2022-07-01", hours_before=8)
    edge, period = tmp_path / "edge.nc", tmp_path / "period.nc"
    full.isel(valid_time=slice(0, 8)).to_netcdf(edge)
    full.isel(valid_time=slice(8, None)).to_netcdf(period)

    def interrupted_write(self, path, *args, **kwargs):
        Path(path).write_bytes(b"truncated")
        raise OSError("write interrupted")

    monkeypatch.setattr(xr.Dataset, "to_netcdf", interrupted_write)
    merged_file = tmp_path / "merged.nc"
    with pytest.raises(OSError):
        merge_hourly_files([edge, period], merged_file)
    assert not merged_file.exists()

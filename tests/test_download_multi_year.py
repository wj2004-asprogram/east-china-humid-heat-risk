"""
Resumable multi-year download (no network): a fake `cdsapi` module writes
synthetic NetCDF files for each request, or fails on chosen files.
"""

import sys
import types

import pandas as pd
import pytest

from src import download_era5
from src.config import load_config
from src.download_era5 import check_raw_file, download_raw_files, report_raw_files
from tests.synthetic import make_synthetic_era5_at, make_synthetic_land_sea_mask


CONFIG = """
name: multi
period:
  years: {first: 2003, last: 2004}
  months: [6, 7]
region: {north: 36, west: 112, south: 25, east: 123}
cities:
  - {name: Nanjing,  lat: 32.06, lon: 118.79, color: "#d62728"}
utc_offset_hours: 8
land_sea_mask_threshold: 0.5
file_tag: multi
figures_dir: figures/multi
"""


@pytest.fixture
def cfg(tmp_path):
    config_file = tmp_path / "multi.yaml"
    config_file.write_text(CONFIG, encoding="utf-8")
    cfg = load_config(config_file, project_root=tmp_path)
    # 4 monthly files and 2 edge files (31 May, 16-23 UTC).
    assert len(cfg.raw_files) == 6
    return cfg


def _times(request) -> pd.DatetimeIndex:
    """The hours a CDS request returns: the product of its days and times."""
    y, m = request["year"][0], request["month"][0]
    return pd.DatetimeIndex([f"{y}-{m}-{d} {t}" for d in request["day"] for t in request["time"]])


class FakeCDS:
    """
    Fake `cdsapi`. Client().retrieve writes a synthetic file for the
    request, or, for target names in `fail`, writes a partial file and
    raises the given exception, as an interrupted transfer would.
    """

    def __init__(self, monkeypatch, fail=None, drop_hour=None):
        self.calls = []  # target file names, in order
        self.fail = fail or {}
        self.drop_hour = drop_hour  # (target name, timestamp) to leave out of a file
        fake = self

        class Client:
            def retrieve(self, dataset, request, target):
                name = target.replace("\\", "/").rsplit("/", 1)[-1]
                fake.calls.append(name)
                if request["variable"] == ["land_sea_mask"]:
                    make_synthetic_land_sea_mask("2003-06-01", step=1.0).to_netcdf(target)
                    return
                for prefix, error in fake.fail.items():
                    if name.startswith(prefix):
                        with open(target, "wb") as f:
                            f.write(b"CDF partial")
                        raise error
                times = _times(request)
                if fake.drop_hour and name.startswith(fake.drop_hour[0]):
                    times = times.drop(pd.Timestamp(fake.drop_hour[1]))
                make_synthetic_era5_at(times, step=1.0).to_netcdf(target)

        monkeypatch.setitem(sys.modules, "cdsapi", types.SimpleNamespace(Client=Client))


def _all_complete(cfg) -> bool:
    for raw_file in cfg.raw_files:
        check_raw_file(raw_file.path, raw_file, cfg)
    return not list(cfg.raw_files[0].path.parent.glob("*.part"))


def test_download_fetches_every_file_once(cfg, monkeypatch):
    fake = FakeCDS(monkeypatch)
    download_era5.run(cfg)

    assert fake.calls == [cfg.paths.raw_land_sea_mask.name + ".part"] + [f.path.name + ".part" for f in cfg.raw_files]
    assert _all_complete(cfg)
    assert report_raw_files(cfg)

    # A second run finds every file complete and requests nothing.
    fake.calls.clear()
    download_era5.run(cfg)
    assert fake.calls == []


@pytest.mark.parametrize("interruption", [KeyboardInterrupt(), ConnectionError("connection reset")])
def test_resume_after_an_interrupted_download(cfg, monkeypatch, interruption):
    july_2004 = cfg.raw_files[3]
    assert july_2004.path.name == "era5_east_china_2004_07.nc"

    FakeCDS(monkeypatch, fail={july_2004.path.name: interruption})
    # Ctrl+C stops the run at once; a CDS/network error is reported and the
    # remaining files are still fetched, then the run fails.
    with pytest.raises(type(interruption) if isinstance(interruption, KeyboardInterrupt) else RuntimeError):
        download_raw_files(cfg)

    # The partial file is never at the final path.
    assert not july_2004.path.exists()
    assert july_2004.path.with_name(july_2004.path.name + ".part").exists()
    done_before = [f.path.name for f in cfg.raw_files if f.path.exists()]
    assert not report_raw_files(cfg)

    # Resume: only what is missing is requested; the stale .part is replaced.
    fake = FakeCDS(monkeypatch)
    download_raw_files(cfg)
    assert fake.calls == [f.path.name + ".part" for f in cfg.raw_files if f.path.name not in done_before]
    assert july_2004.path.name + ".part" in fake.calls
    assert _all_complete(cfg)


def test_leftover_part_file_is_not_taken_as_complete(cfg, monkeypatch):
    june_2003 = cfg.raw_files[0]
    june_2003.path.parent.mkdir(parents=True)
    june_2003.path.with_name(june_2003.path.name + ".part").write_bytes(b"CDF partial")

    fake = FakeCDS(monkeypatch)
    download_raw_files(cfg)
    assert june_2003.path.name + ".part" in fake.calls
    assert _all_complete(cfg)


def test_incomplete_existing_file_is_reported_not_overwritten(cfg, monkeypatch):
    june_2003 = cfg.raw_files[0]
    june_2003.path.parent.mkdir(parents=True)
    june_2003.path.write_bytes(b"truncated")  # e.g. copied by hand, not by the downloader

    fake = FakeCDS(monkeypatch)
    with pytest.raises(RuntimeError, match="(?s)era5_east_china_2003_06.nc.*delete it"):
        download_raw_files(cfg)
    assert june_2003.path.read_bytes() == b"truncated"
    assert june_2003.path.name + ".part" not in fake.calls
    # Every other file was still downloaded.
    for raw_file in cfg.raw_files[1:]:
        check_raw_file(raw_file.path, raw_file, cfg)


def test_downloaded_file_with_a_missing_hour_is_not_renamed(cfg, monkeypatch):
    edge = cfg.raw_files[4]
    assert edge.path.name == "era5_east_china_2003_05_31_h16-23.nc"
    FakeCDS(monkeypatch, drop_hour=(edge.path.name, "2003-05-31 20:00"))

    with pytest.raises(RuntimeError, match="h16-23.*time steps do not match"):
        download_raw_files(cfg)
    assert not edge.path.exists()


def test_stops_after_repeated_failures(cfg, monkeypatch):
    # Every request fails (e.g. licence not accepted): stop after 3 tries.
    fake = FakeCDS(monkeypatch, fail={"era5_east_china_": PermissionError("licence not accepted")})
    with pytest.raises(RuntimeError, match="3 failed downloads in a row.*licence"):
        download_raw_files(cfg)
    assert len(fake.calls) == download_era5.MAX_CONSECUTIVE_FAILURES
    assert not any(f.path.exists() for f in cfg.raw_files)

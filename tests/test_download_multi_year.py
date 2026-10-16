"""
Resumable multi-year download (no network): a fake `cdsapi` module writes
synthetic NetCDF files for each request, or fails on chosen files; with
several workers, requests run in parallel threads.
"""

import sys
import threading
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
    Fake `cdsapi`. Client(**options).retrieve writes a synthetic file for
    the request, or, for target names in `fail`, writes a partial file and
    raises the given exception, as an interrupted transfer would. `hold`
    (called with the target name at the start of each request) can make
    requests wait, e.g. for each other.
    """

    def __init__(self, monkeypatch, fail=None, drop_hour=None, hold=None, alter=None):
        self.calls = []  # target file names, in the order the requests started
        self.options = []  # keyword arguments of each Client()
        self.fail = fail or {}
        self.drop_hour = drop_hour  # (target name, timestamp) to leave out of a file
        self.hold = hold
        self.alter = alter  # (target name, function(Dataset) -> Dataset) for bad values
        self.lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0
        fake = self

        class Client:
            def __init__(self, **options):
                with fake.lock:
                    fake.options.append(options)

            def retrieve(self, dataset, request, target):
                name = target.replace("\\", "/").rsplit("/", 1)[-1]
                with fake.lock:
                    fake.calls.append(name)
                    fake.in_flight += 1
                    fake.max_in_flight = max(fake.max_in_flight, fake.in_flight)
                try:
                    if fake.hold is not None:
                        fake.hold(name)
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
                    ds = make_synthetic_era5_at(times, step=1.0)
                    if fake.alter and name.startswith(fake.alter[0]):
                        ds = fake.alter[1](ds)
                    ds.to_netcdf(target)
                finally:
                    with fake.lock:
                        fake.in_flight -= 1

        monkeypatch.setitem(sys.modules, "cdsapi", types.SimpleNamespace(Client=Client))


def _join_download_threads(timeout: float = 30.0) -> None:
    """Wait for file threads still running after a download stopped early."""
    for thread in threading.enumerate():
        if thread.name.startswith("download-"):
            thread.join(timeout)


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


# ---------------------------------------------------------------------------
# Value check: a file with missing or implausible values is never accepted
# ---------------------------------------------------------------------------

def _with_nan_block(ds):
    ds = ds.copy(deep=True)
    ds["t2m"][10:20, 2:5, 3:6] = float("nan")
    return ds


def _in_celsius(ds):
    ds = ds.copy(deep=True)
    ds["d2m"] = ds["d2m"] - 273.15
    return ds


@pytest.mark.parametrize("alter, message", [
    (_with_nan_block, "t2m has 90 missing \\(NaN\\) values"),
    (_in_celsius, "d2m ranges from .* outside the plausible 180-340 K \\(not in Kelvin\\?\\)"),
])
def test_check_rejects_implausible_values(cfg, alter, message):
    june_2003 = cfg.raw_files[0]
    june_2003.path.parent.mkdir(parents=True)
    alter(make_synthetic_era5_at(june_2003.utc_times, step=1.0)).to_netcdf(june_2003.path)

    with pytest.raises(ValueError, match=message):
        check_raw_file(june_2003.path, june_2003, cfg)
    # --check-raw reports it as incomplete.
    assert not report_raw_files(cfg)


def test_downloaded_file_with_bad_values_is_not_renamed(cfg, monkeypatch):
    # A bad CDS month must not get its final name, or a resumed download
    # would skip it as complete for ever.
    july_2003 = cfg.raw_files[1]
    FakeCDS(monkeypatch, alter=(july_2003.path.name, _in_celsius))

    with pytest.raises(RuntimeError, match="(?s)era5_east_china_2003_07.nc: .*not in Kelvin"):
        download_raw_files(cfg, workers=3)
    assert not july_2003.path.exists()
    assert july_2003.path.with_name(july_2003.path.name + ".part").exists()

    # The next run fetches it again.
    fake = FakeCDS(monkeypatch)
    download_raw_files(cfg, workers=3)
    assert fake.calls == [july_2003.path.name + ".part"]
    assert _all_complete(cfg)


# ---------------------------------------------------------------------------
# Several requests at once (--download-workers)
# ---------------------------------------------------------------------------

def test_one_worker_keeps_the_config_order(cfg, monkeypatch):
    # workers=1 (the function's default): one file after the other, in the
    # config's order; a complete file in between is skipped.
    july_2003 = cfg.raw_files[1]
    july_2003.path.parent.mkdir(parents=True)
    make_synthetic_era5_at(july_2003.utc_times, step=1.0).to_netcdf(july_2003.path)

    fake = FakeCDS(monkeypatch)
    download_raw_files(cfg, workers=1)
    assert fake.calls == [f.path.name + ".part" for f in cfg.raw_files if f is not july_2003]
    assert fake.max_in_flight == 1
    assert _all_complete(cfg)


def test_three_workers_run_requests_at_the_same_time(cfg, monkeypatch):
    # Each request waits until 3 requests are in flight: this only finishes
    # if 3 requests really run at once (else the barrier times out and the
    # requests fail).
    barrier = threading.Barrier(3, timeout=20)
    fake = FakeCDS(monkeypatch, hold=lambda name: barrier.wait())

    download_raw_files(cfg, workers=3)
    assert fake.max_in_flight == 3
    assert sorted(fake.calls) == sorted(f.path.name + ".part" for f in cfg.raw_files)
    assert _all_complete(cfg)
    # One client per request, without progress bars, CDS messages prefixed.
    assert len(fake.options) == len(cfg.raw_files)
    assert all(o["progress"] is False and callable(o["info_callback"]) for o in fake.options)


def test_a_failure_does_not_affect_other_files(cfg, monkeypatch):
    july_2004 = cfg.raw_files[3]
    FakeCDS(monkeypatch, fail={july_2004.path.name: ConnectionError("connection reset")})

    with pytest.raises(RuntimeError, match="(?s)1 raw file.*era5_east_china_2004_07.nc: connection reset"):
        download_raw_files(cfg, workers=3)
    assert not july_2004.path.exists()
    assert july_2004.path.with_name(july_2004.path.name + ".part").exists()
    for raw_file in cfg.raw_files:
        if raw_file is not july_2004:
            check_raw_file(raw_file.path, raw_file, cfg)


def test_stop_after_repeated_failures_lets_running_requests_finish(cfg, monkeypatch):
    """
    With 3 workers: the first file is held back, every other request
    fails. After 3 failures in a row (in the order they finish) no new file
    is started; the requests still running finish, the held one with
    success. Files 1-5 are started, file 6 never.
    """
    stopped = threading.Event()
    say = download_era5._say

    def watch(label, message):
        say(label, message)
        if "no new requests" in message:
            stopped.set()

    monkeypatch.setattr(download_era5, "_say", watch)
    first = cfg.raw_files[0]

    def hold(name):
        if name.startswith(first.path.name):
            assert stopped.wait(20), "the download did not stop starting new files"

    fake = FakeCDS(monkeypatch, hold=hold,
                   fail={f.path.name: ConnectionError("CDS down") for f in cfg.raw_files[1:]})

    with pytest.raises(RuntimeError, match="(?s)3 failed downloads in a row.*CDS down") as excinfo:
        download_raw_files(cfg, workers=3)
    assert sorted(fake.calls) == sorted(f.path.name + ".part" for f in cfg.raw_files[:5])
    check_raw_file(first.path, first, cfg)  # the running request finished
    assert not cfg.raw_files[5].path.exists()
    # Every failure is reported, with its reason.
    assert str(excinfo.value).count("CDS down") == 1 + 4


def test_resume_with_three_workers(cfg, monkeypatch):
    june_2004, edge_2003 = cfg.raw_files[2], cfg.raw_files[4]
    FakeCDS(monkeypatch, fail={june_2004.path.name: ConnectionError("reset"),
                               edge_2003.path.name: ConnectionError("reset")})
    with pytest.raises(RuntimeError, match="2 raw file"):
        download_raw_files(cfg, workers=3)
    assert {p.name for p in june_2004.path.parent.glob("*.part")} == {
        june_2004.path.name + ".part", edge_2003.path.name + ".part",
    }

    # Resume: complete files are skipped, the .part files are fetched again.
    fake = FakeCDS(monkeypatch)
    download_raw_files(cfg, workers=3)
    assert sorted(fake.calls) == sorted([june_2004.path.name + ".part", edge_2003.path.name + ".part"])
    assert _all_complete(cfg)


def test_ctrl_c_with_three_workers_leaves_only_part_files(cfg, monkeypatch):
    # Ctrl+C (here raised inside one request) stops the download; the
    # interrupted file exists only as .part.
    july_2003 = cfg.raw_files[1]
    FakeCDS(monkeypatch, fail={july_2003.path.name: KeyboardInterrupt()})
    try:
        with pytest.raises(KeyboardInterrupt):
            download_raw_files(cfg, workers=3)
    finally:
        _join_download_threads()
    assert not july_2003.path.exists()
    assert july_2003.path.with_name(july_2003.path.name + ".part").exists()
    for raw_file in cfg.raw_files:  # whatever was renamed is complete
        if raw_file.path.exists():
            check_raw_file(raw_file.path, raw_file, cfg)


@pytest.mark.parametrize("workers", [0, 5])
def test_number_of_workers_is_checked(cfg, workers):
    with pytest.raises(ValueError, match="workers"):
        download_raw_files(cfg, workers=workers)

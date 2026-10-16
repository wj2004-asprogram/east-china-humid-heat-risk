"""
Multi-year seasonal configs (phase 2): config format, raw file layout and
month-by-month processing on synthetic data split into monthly raw files
exactly as the CDS delivers them.
"""

import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402

from src import pipeline  # noqa: E402
from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.download_era5 import build_raw_file_request, check_raw_file  # noqa: E402
from src.hourly_input import iter_hourly_blocks, utc_window  # noqa: E402
from src.indices import heat_index_rothfusz, kelvin_to_celsius, relative_humidity  # noqa: E402
from src.pipeline import STEPS  # noqa: E402
from src.risk import CATEGORY_LABELS, MISSING_CATEGORY, VALID_COVERAGE_LABEL  # noqa: E402
from tests.synthetic import make_synthetic_era5, make_synthetic_land_sea_mask  # noqa: E402


CONFIG_DIR = PROJECT_ROOT / "configs"

# Two summers, 2004 a leap year. Grid step 1° keeps the files small.
CONFIG = """
name: multi
period:
  years: {first: 2003, last: 2004}
  months: [6, 7, 8]
region: {north: 36, west: 112, south: 25, east: 123}
cities:
  - {name: Nanjing,  lat: 32.06, lon: 118.79, color: "#d62728"}
  - {name: Shanghai, lat: 31.23, lon: 121.47, color: "#1f77b4"}
utc_offset_hours: 8
land_sea_mask_threshold: 0.5
file_tag: multi
figures_dir: figures/multi
"""
STEP_DEG = 1.0
SUMMER_DAYS = 30 + 31 + 31


def _config(tmp_path: Path, utc_offset_hours: int = 8, text: str = CONFIG):
    config_file = tmp_path / f"multi_{utc_offset_hours}.yaml"
    config_file.write_text(
        text.replace("utc_offset_hours: 8", f"utc_offset_hours: {utc_offset_hours}"), encoding="utf-8",
    )
    return load_config(config_file, project_root=tmp_path)


def _continuous_hourly() -> xr.Dataset:
    """One continuous synthetic hourly series, 16:00 UTC on 31 May 2003 to 23:00 UTC on 31 Aug 2004."""
    return make_synthetic_era5("2003-06-01", "2004-08-31", step=STEP_DEG, hours_before=8)


def _write_raw_files(cfg, hourly: xr.Dataset) -> None:
    """Cut every raw file of the config from the continuous series, as downloaded."""
    for raw_file in cfg.raw_files:
        raw_file.path.parent.mkdir(parents=True, exist_ok=True)
        hourly.sel(valid_time=raw_file.utc_times).to_netcdf(raw_file.path)
    make_synthetic_land_sea_mask("2003-06-01", step=STEP_DEG).to_netcdf(cfg.paths.raw_land_sea_mask)


def _hourly_heat_index(hourly: xr.Dataset) -> xr.DataArray:
    t2m_c = kelvin_to_celsius(hourly["t2m"])
    return heat_index_rothfusz(t2m_c, relative_humidity(t2m_c, kelvin_to_celsius(hourly["d2m"])))


def _reference_daily_max(hourly_values: xr.DataArray, days, utc_offset_hours: int) -> np.ndarray:
    """
    Independent reference: the maximum over the 24 UTC hours of each local
    day, cut from the continuous series with plain timestamps.
    """
    out = []
    for day in days:
        first_hour = pd.Timestamp(day) - pd.Timedelta(hours=utc_offset_hours)
        window = hourly_values.sel(valid_time=slice(first_hour, first_hour + pd.Timedelta(hours=23)))
        assert window.sizes["valid_time"] == 24
        out.append(window.max("valid_time").values)
    return np.stack(out)


# ---------------------------------------------------------------------------
# Config format and raw file layout
# ---------------------------------------------------------------------------

def test_multi_year_configs():
    utc = load_config(CONFIG_DIR / "jja_2000_2024.yaml")
    bjt = load_config(CONFIG_DIR / "jja_2000_2024_bjt.yaml")

    for cfg in (utc, bjt):
        assert cfg.is_multi_year
        assert cfg.years == tuple(range(2000, 2025)) and cfg.months == (6, 7, 8)
        assert cfg.period_label == "June to August, 2000 to 2024"
        assert (cfg.start_date, cfg.end_date) == (date(2000, 6, 1), date(2024, 8, 31))
        assert len(cfg.seasons) == 25 and len(cfg.month_blocks) == 75
        assert len(cfg.local_days) == 25 * SUMMER_DAYS
        assert cfg.paths.raw_era5 is None
        assert cfg.cds_area == [36, 112, 25, 123]

    assert (utc.utc_offset_hours, bjt.utc_offset_hours) == (0, 8)
    assert (utc.day_definition, bjt.day_definition) == ("UTC", "UTC+8")
    assert bjt.day_window_utc == "16:00 UTC on D-1 to 15:00 UTC on D"

    # 75 monthly files (one CDS request per year and month), shared by both
    # configs; Beijing time adds one 31 May 16-23 UTC file per summer.
    monthly = [f"era5_east_china_{y}_{m:02d}.nc" for y in range(2000, 2025) for m in (6, 7, 8)]
    edges = [f"era5_east_china_{y}_05_31_h16-23.nc" for y in range(2000, 2025)]
    assert [f.path.name for f in utc.raw_files] == monthly
    assert [f.path.name for f in bjt.raw_files] == monthly + edges
    assert all(f.path.parent == PROJECT_ROOT / "data" / "raw" for f in bjt.raw_files)

    # The July 2022 monthly file is the raw file of the v0.2 UTC config.
    july = next(f for f in utc.raw_files if (f.year, f.month) == (2022, 7))
    assert july.path == load_config(CONFIG_DIR / "july_2022.yaml").paths.raw_era5

    # Processed files are never shared between UTC and Beijing time.
    shared = set(vars(utc.paths).values()) & set(vars(bjt.paths).values())
    assert shared == {utc.paths.raw_land_sea_mask, None}


def test_every_request_is_a_single_month():
    cfg = load_config(CONFIG_DIR / "jja_2000_2024_bjt.yaml")
    for raw_file in cfg.raw_files:
        request = build_raw_file_request(cfg, raw_file)
        assert len(request["year"]) == 1 and len(request["month"]) == 1
        # The CDS returns the product of days and times: exactly the hours
        # the file must hold.
        product = pd.DatetimeIndex([
            f"{request['year'][0]}-{request['month'][0]}-{d} {t}" for d in request["day"] for t in request["time"]
        ])
        assert product.equals(raw_file.utc_times)
        assert request["area"] == [36, 112, 25, 123]

    june_2000, edge_2000 = cfg.raw_files[0], cfg.raw_files[75]
    assert len(june_2000.utc_times) == 30 * 24
    assert list(edge_2000.utc_times.strftime("%Y-%m-%d %H")) == [f"2000-05-31 {h}" for h in range(16, 24)]


def test_edge_files_only_where_no_monthly_file_covers_them(tmp_path):
    # West of UTC (UTC-5): the first 5 UTC hours of the day after each season.
    west = _config(tmp_path, utc_offset_hours=-5)
    assert [f.path.name for f in west.raw_files[6:]] == [
        "era5_east_china_2003_09_01_h00-04.nc", "era5_east_china_2004_09_01_h00-04.nc",
    ]
    # UTC days: monthly files only.
    assert len(_config(tmp_path, utc_offset_hours=0).raw_files) == 6

    # Whole years: the 31 December hours of 2003 are in the December 2003
    # file, so only the first year needs an edge file.
    all_year = _config(tmp_path, text=CONFIG.replace("months: [6, 7, 8]", "months: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]"))
    assert [f.path.name for f in all_year.raw_files[24:]] == ["era5_east_china_2002_12_31_h16-23.nc"]


@pytest.mark.parametrize(
    "period",
    [
        "{years: {first: 2003, last: 2004}, months: [8, 6]}",       # not increasing
        "{years: {first: 2003, last: 2004}, months: [6, 8]}",       # not consecutive
        "{years: {first: 2003, last: 2004}, months: [12, 1]}",      # across a year boundary
        "{years: {first: 2003, last: 2004}, months: []}",
        "{years: {first: 2003, last: 2004}, months: [0, 1]}",
        "{years: {first: 2003, last: 2004}, months: [true]}",
        "{years: {first: 2004, last: 2003}, months: [6]}",           # reversed
        "{years: {first: 1939, last: 2003}, months: [6]}",           # before ERA5
        "{years: [2003, 2004], months: [6]}",
        "{years: {first: '2003', last: 2004}, months: [6]}",
        '{years: {first: 2003, last: 2004}, months: [6], start: "2003-06-01"}',  # both kinds
        "{months: [6, 7, 8]}",
    ],
)
def test_bad_multi_year_period_is_rejected(tmp_path, period):
    lines = CONFIG.splitlines()
    i = lines.index("period:")
    text = "\n".join(lines[:i] + [f"period: {period}"] + lines[i + 3:])
    with pytest.raises(ValueError):
        _config(tmp_path, text=text)


# ---------------------------------------------------------------------------
# Month-by-month processing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("utc_offset_hours", [8, 0])
def test_multi_year_pipeline_matches_continuous_reference(tmp_path, utc_offset_hours):
    """
    All steps on two summers split into monthly files (and, for Beijing
    time, the 31 May edge files). Every daily value equals the reference
    cut from one continuous hourly series: month boundaries are seamless,
    1 June uses 16-23 UTC of 31 May, and the gap between the summers is
    neither filled nor counted as missing.
    """
    hourly = _continuous_hourly()
    cfg = _config(tmp_path, utc_offset_hours)
    _write_raw_files(cfg, hourly)

    for step in STEPS.values():
        step(cfg)

    days = cfg.local_days
    assert len(days) == 2 * SUMMER_DAYS
    assert days[SUMMER_DAYS - 1] == pd.Timestamp("2003-08-31") and days[SUMMER_DAYS] == pd.Timestamp("2004-06-01")

    with xr.open_dataset(cfg.paths.humid_heat_nc) as hh, xr.open_dataset(cfg.paths.daily_tmax_nc) as tmax:
        assert pd.DatetimeIndex(hh["date"].values).equals(days)
        assert pd.DatetimeIndex(tmax["valid_time"].values).equals(days)
        # Multi-year runs keep the daily variables only.
        assert set(hh.data_vars) == {"daily_max_heat_index_c", "daily_mean_rh"}
        assert not hh["daily_max_heat_index_c"].isnull().any()

        np.testing.assert_allclose(
            hh["daily_max_heat_index_c"].values,
            _reference_daily_max(_hourly_heat_index(hourly), days, utc_offset_hours),
            rtol=0, atol=1e-5,
        )
        np.testing.assert_allclose(
            tmax["t2m"].values,
            _reference_daily_max(hourly["t2m"], days, utc_offset_hours) - 273.15,
            rtol=0, atol=1e-4,
        )

    fractions = pd.read_csv(cfg.paths.area_fraction_csv)
    assert list(fractions["date"]) == list(days.strftime("%Y-%m-%d"))
    assert fractions[VALID_COVERAGE_LABEL].tolist() == [100.0] * len(days)
    assert fractions[list(CATEGORY_LABELS.values())].sum(axis=1).tolist() == pytest.approx([100.0] * len(days))

    city_hi = pd.read_csv(cfg.paths.city_heat_index_csv)
    assert list(city_hi["date"]) == list(fractions["date"])
    assert not city_hi.isnull().any().any()

    # Same CSV schema as single-period runs: local dates in a `date` column.
    city_tmax = pd.read_csv(cfg.paths.city_tmax_csv)
    assert list(city_tmax.columns) == ["city", "date", "tmax_C", "era5_lat", "era5_lon"]
    assert len(city_tmax) == 2 * len(days)
    for _, rows in city_tmax.groupby("city"):
        assert list(rows["date"]) == list(fractions["date"])

    # run_info records the day definition and the seasonal period.
    info = json.loads(cfg.paths.run_info_json.read_text(encoding="utf-8"))
    assert info["day_definition"] == ("UTC+8" if utc_offset_hours == 8 else "UTC")
    assert info["period"]["start"] == "2003-06-01" and info["period"]["end"] == "2004-08-31"
    assert info["period"]["years"] == {"first": 2003, "last": 2004}
    assert info["period"]["months"] == [6, 7, 8]
    assert info["period"]["days"] == len(days)

    with xr.open_dataset(cfg.paths.climatology_nc) as clim:
        assert clim["daily_max_heat_index_c"].dims == ("year", "day_of_year", "latitude", "longitude")
        assert clim["year"].values.tolist() == [2003, 2004]
        assert clim["day_of_year"].values.tolist() == list(range(152, 152 + SUMMER_DAYS))

    for name, path in vars(cfg.paths).items():
        if name.startswith("fig_"):
            assert path.exists(), f"missing figure {path.name}"


def test_beijing_day_1_june_uses_the_edge_file(tmp_path):
    """
    Spikes in 2m temperature at the hours around the edges of a Beijing
    summer: 16:00 UTC on 31 May (edge file) is the first hour of Beijing
    1 June; 16:00 UTC on 30 June (June file) the first hour of 1 July;
    16:00 UTC on 31 August (August file) belongs to 1 September, outside
    the run.
    """
    hourly = _continuous_hourly()
    cell = {"latitude": 30.0, "longitude": 115.0}
    spikes = {"2003-05-31T16:00": 331.15, "2003-06-30T16:00": 332.15, "2003-08-31T16:00": 333.15,
              "2003-06-30T15:00": 334.15}
    for utc_time, kelvin in spikes.items():
        hourly["t2m"].loc[{"valid_time": utc_time, **cell}] = kelvin

    cfg = _config(tmp_path)
    _write_raw_files(cfg, hourly)
    STEPS["process_temperature"](cfg)

    with xr.open_dataset(cfg.paths.daily_tmax_nc) as tmax:
        point = tmax["t2m"].sel(**cell)
        assert float(point.sel(valid_time="2003-06-01")) == pytest.approx(58.0, abs=1e-4)
        assert float(point.sel(valid_time="2003-06-30")) == pytest.approx(61.0, abs=1e-4)  # 15:00 UTC = 23:00 Beijing
        assert float(point.sel(valid_time="2003-07-01")) == pytest.approx(59.0, abs=1e-4)
        assert float(point.sel(valid_time="2003-08-31")) < 55.0
        assert pd.Timestamp("2003-09-01") not in pd.DatetimeIndex(tmax["valid_time"].values)


@pytest.mark.parametrize("utc_offset_hours", [0, 8])
def test_blocks_own_their_data(tmp_path, utc_offset_hours):
    """
    Each month block is a deep copy. Changing a block in place (here: all
    NaN) changes neither the raw files kept in memory nor a later block.
    UTC blocks are cut from a single file (without the deep copy they would
    be views of it); Beijing-time blocks join two files.
    """
    hourly = _continuous_hourly()
    cfg = _config(tmp_path, utc_offset_hours)
    _write_raw_files(cfg, hourly)

    n_blocks = 0
    for block in iter_hourly_blocks(cfg):
        start, end = utc_window(block.first_day, block.last_day, utc_offset_hours)
        for name in ("t2m", "d2m"):
            values = block.data[name].values
            np.testing.assert_array_equal(values, hourly[name].sel(valid_time=slice(start, end)).values)
            assert values.flags.owndata, f"{name} of block {block.first_day} is a view"
            values[...] = np.nan  # in place
        n_blocks += 1
    assert n_blocks == len(cfg.month_blocks)


def test_missing_edge_file_is_reported(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    _write_raw_files(cfg, _continuous_hourly())
    edge = cfg.raw_files[-1].path
    assert edge.name == "era5_east_china_2004_05_31_h16-23.nc"
    edge.unlink()

    monkeypatch.setattr(pipeline, "load_config", lambda path: load_config(path, project_root=tmp_path))
    with pytest.raises(SystemExit, match="(?s)1 of 8.*2004_05_31_h16-23"):
        pipeline.main(["--config", str(tmp_path / "multi_8.yaml"), "--steps", "process_humid_heat"])


def test_missing_hour_in_an_edge_file_is_an_error(tmp_path):
    # F2: Beijing 1 June 2004 without 20:00 UTC on 31 May has 23 hours: stop.
    hourly = _continuous_hourly()
    cfg = _config(tmp_path)
    _write_raw_files(cfg, hourly)
    edge = cfg.raw_files[-1]
    hourly.sel(valid_time=edge.utc_times.drop(pd.Timestamp("2004-05-31 20:00"))).to_netcdf(edge.path)

    with pytest.raises(ValueError, match="2004-06-01"):
        STEPS["process_humid_heat"](cfg)
    with pytest.raises(ValueError, match="time steps do not match"):
        check_raw_file(edge.path, edge, cfg)


def test_missing_value_across_a_month_boundary_is_missing(tmp_path):
    # F1: a NaN at 20:00 UTC on 31 July 2004 (04:00 Beijing on 1 August)
    # makes that cell Missing on Beijing 1 August only.
    hourly = _continuous_hourly()
    cell = {"latitude": 30.0, "longitude": 115.0}
    hourly["d2m"].loc[{"valid_time": "2004-07-31T20:00", **cell}] = np.nan

    cfg = _config(tmp_path)
    _write_raw_files(cfg, hourly)
    for step in ("process_humid_heat", "classify_heat_risk"):
        STEPS[step](cfg)

    with xr.open_dataset(cfg.paths.heat_risk_nc) as ds:
        category = ds["heat_risk_category"].sel(**cell)
        missing_days = pd.DatetimeIndex(category["date"].values[category.values == MISSING_CATEGORY])
        assert list(missing_days.strftime("%Y-%m-%d")) == ["2004-08-01"]
        assert int((ds["heat_risk_category"] == MISSING_CATEGORY).sum()) == 1


def test_monthly_files_must_share_one_grid(tmp_path):
    # A raw file on another grid must not be joined with NaN padding (which
    # would show up as Missing): concatenation uses join="exact".
    hourly = _continuous_hourly()
    cfg = _config(tmp_path)
    _write_raw_files(cfg, hourly)
    july = cfg.raw_files[1]
    hourly.sel(valid_time=july.utc_times).isel(longitude=slice(0, -1)).to_netcdf(july.path)

    with pytest.raises(ValueError):
        STEPS["process_humid_heat"](cfg)
    with pytest.raises(ValueError, match="grid spans"):
        check_raw_file(july.path, july, cfg)

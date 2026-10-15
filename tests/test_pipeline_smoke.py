"""End-to-end tests: run the pipeline steps on small synthetic ERA5 files."""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402

from src.config import PROJECT_ROOT, load_config  # noqa: E402
from src.pipeline import STEPS  # noqa: E402
from src.risk import CATEGORY_LABELS  # noqa: E402
from tests.synthetic import make_synthetic_era5, make_synthetic_land_sea_mask  # noqa: E402


CONFIG = """
name: smoke
period: {start: "2022-07-01", end: "2022-07-03"}
region: {north: 36, west: 112, south: 25, east: 123}
cities:
  - {name: Nanjing,  lat: 32.06, lon: 118.79, color: "#d62728"}
  - {name: Shanghai, lat: 31.23, lon: 121.47, color: "#1f77b4"}
utc_offset_hours: 8
land_sea_mask_threshold: 0.5
file_tag: smoke
figures_dir: figures/smoke
"""


@pytest.mark.parametrize("utc_offset_hours", [0, 8])
def test_pipeline_runs_end_to_end(tmp_path, utc_offset_hours):
    config_file = tmp_path / "smoke.yaml"
    config_file.write_text(
        CONFIG.replace("utc_offset_hours: 8", f"utc_offset_hours: {utc_offset_hours}"),
        encoding="utf-8",
    )
    cfg = load_config(config_file, project_root=tmp_path)

    cfg.paths.raw_era5.parent.mkdir(parents=True)
    # As downloaded: for UTC+8 also 16:00-23:00 UTC on 2022-06-30.
    make_synthetic_era5(
        "2022-07-01", "2022-07-03", hours_before=utc_offset_hours,
    ).to_netcdf(cfg.paths.raw_era5)
    make_synthetic_land_sea_mask("2022-07-01").to_netcdf(cfg.paths.raw_land_sea_mask)

    for step in STEPS.values():
        step(cfg)

    fractions = pd.read_csv(cfg.paths.area_fraction_csv)
    assert list(fractions.columns) == ["date", *CATEGORY_LABELS.values()]
    assert list(fractions["date"]) == ["2022-07-01", "2022-07-02", "2022-07-03"]
    assert fractions[list(CATEGORY_LABELS.values())].sum(axis=1).tolist() == pytest.approx([100.0] * 3)

    # Land mask saved with the categories: the 3 columns east of 121.5 °E are sea.
    with xr.open_dataset(cfg.paths.heat_risk_nc) as ds:
        land = ds["land_mask"].astype(bool)
        assert not land.sel(longitude=[122.0, 122.5, 123.0]).any()
        assert land.sel(longitude=slice(112.0, 121.5)).all()

    city_hi = pd.read_csv(cfg.paths.city_heat_index_csv)
    assert list(city_hi.columns) == ["date", "Nanjing", "Shanghai"]
    assert len(city_hi) == 3

    city_tmax = pd.read_csv(cfg.paths.city_tmax_csv)
    assert len(city_tmax) == 2 * 3
    assert city_tmax["tmax_C"].between(15, 60).all()  # Celsius, not Kelvin

    for name, path in vars(cfg.paths).items():
        if name.startswith("fig_"):
            assert path.exists(), f"missing figure {path.name}"


def test_beijing_sample_config_day_boundaries(tmp_path):
    """
    configs/sample_2022_07_01_07_bjt.yaml on synthetic data: each Beijing
    day D (2022-07-01 ... 2022-07-07) is computed from 16:00 UTC on D-1 to
    15:00 UTC on D, so day 1 uses 16:00-23:00 UTC on 2022-06-30.
    """
    cfg = load_config(PROJECT_ROOT / "configs" / "sample_2022_07_01_07_bjt.yaml", project_root=tmp_path)
    raw = make_synthetic_era5("2022-07-01", "2022-07-07", hours_before=8)
    assert str(raw["valid_time"].values[0])[:16] == "2022-06-30T16:00"
    cfg.paths.raw_era5.parent.mkdir(parents=True)
    raw.to_netcdf(cfg.paths.raw_era5)

    STEPS["process_temperature"](cfg)
    STEPS["process_humid_heat"](cfg)

    days = pd.date_range("2022-07-01", "2022-07-07")
    with xr.open_dataset(cfg.paths.daily_tmax_nc) as tmax, xr.open_dataset(cfg.paths.humid_heat_nc) as hh:
        assert list(tmax["valid_time"].values) == list(days.values)
        assert list(hh["date"].values) == list(days.values)

        for day in days:
            window = slice(day - pd.Timedelta(hours=8), day + pd.Timedelta(hours=15))
            hourly_t2m_k = raw["t2m"].sel(valid_time=window)
            assert hourly_t2m_k.sizes["valid_time"] == 24

            np.testing.assert_allclose(
                tmax["t2m"].sel(valid_time=day), hourly_t2m_k.max("valid_time") - 273.15, atol=1e-4,
            )
            np.testing.assert_array_equal(
                hh["daily_max_heat_index_c"].sel(date=day), hh["heat_index_c"].sel(time=window).max("time"),
            )

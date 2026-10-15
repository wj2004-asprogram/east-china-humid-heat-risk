"""End-to-end smoke test: run every step on a small synthetic ERA5 file."""

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402

from src.config import load_config  # noqa: E402
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
land_sea_mask_threshold: 0.5
file_tag: smoke
figures_dir: figures/smoke
"""


def test_pipeline_runs_end_to_end(tmp_path):
    config_file = tmp_path / "smoke.yaml"
    config_file.write_text(CONFIG, encoding="utf-8")
    cfg = load_config(config_file, project_root=tmp_path)

    cfg.paths.raw_era5.parent.mkdir(parents=True)
    make_synthetic_era5("2022-07-01", "2022-07-03").to_netcdf(cfg.paths.raw_era5)
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

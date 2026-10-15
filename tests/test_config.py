"""Configuration tests: every config must reproduce the v0.1/v0.2 file names."""

from pathlib import Path

import pytest

from src.config import PROJECT_ROOT, load_config
from src.download_era5 import build_request


CONFIG_DIR = PROJECT_ROOT / "configs"


@pytest.mark.parametrize(
    "config_name, tag, period",
    [
        ("sample_2022_07_01_07.yaml", "2022_07_01_07", "2022-07-01 to 2022-07-07"),
        ("july_2022.yaml", "2022_07", "2022-07-01 to 2022-07-31"),
    ],
)
def test_config_reproduces_existing_outputs(config_name, tag, period):
    cfg = load_config(CONFIG_DIR / config_name)

    assert cfg.period_label == period
    assert cfg.cds_area == [36, 112, 25, 123]
    assert [c.name for c in cfg.cities] == ["Nanjing", "Shanghai", "Hangzhou", "Huai'an"]
    assert cfg.paths.raw_era5 == PROJECT_ROOT / "data" / "raw" / f"era5_east_china_{tag}.nc"

    # The CSVs tracked in git are exactly the files this config writes,
    # so a re-run shows any change in results as a git diff.
    for csv in (cfg.paths.city_tmax_csv, cfg.paths.city_heat_index_csv, cfg.paths.area_fraction_csv):
        assert csv.exists(), f"{csv.relative_to(PROJECT_ROOT)} is not a tracked output"

    # Figures go to the version folder that already holds this run's figures.
    assert cfg.paths.fig_area_fraction.exists()


def test_config_rejects_reversed_period(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        (CONFIG_DIR / "july_2022.yaml").read_text(encoding="utf-8")
        .replace('start: "2022-07-01"', 'start: "2022-08-01"'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_config(bad)


@pytest.mark.parametrize(
    "old, new",
    [
        # City outside the region would silently snap to an edge grid cell.
        ('{name: Nanjing,  lat: 32.06', '{name: Nanjing,  lat: 39.90'),
        # Region with south >= north.
        ("south: 25", "south: 40"),
    ],
)
def test_config_rejects_invalid_region_or_city(tmp_path: Path, old, new):
    text = (CONFIG_DIR / "july_2022.yaml").read_text(encoding="utf-8")
    assert old in text
    bad = tmp_path / "bad.yaml"
    bad.write_text(text.replace(old, new), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(bad)


def test_download_request_days():
    sample = build_request(load_config(CONFIG_DIR / "sample_2022_07_01_07.yaml"))
    july = build_request(load_config(CONFIG_DIR / "july_2022.yaml"))

    assert sample["day"] == [f"{d:02d}" for d in range(1, 8)]
    assert july["day"] == [f"{d:02d}" for d in range(1, 32)]
    assert july["year"] == ["2022"] and july["month"] == ["07"]
    assert len(july["time"]) == 24
    assert july["area"] == [36, 112, 25, 123]

"""
Configuration tests: the UTC configs reproduce the v0.1/v0.2 (phase 1a)
file names; the Beijing-time configs share no raw or processed file with them.
"""

from pathlib import Path

import pytest

from src.config import PROJECT_ROOT, load_config
from src.download_era5 import build_edge_request, build_land_sea_mask_request, build_request


CONFIG_DIR = PROJECT_ROOT / "configs"

# (UTC config, Beijing-time config) pairs over the same period.
CONFIG_PAIRS = [
    ("sample_2022_07_01_07.yaml", "sample_2022_07_01_07_bjt.yaml"),
    ("july_2022.yaml", "july_2022_bjt.yaml"),
]


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
    assert cfg.utc_offset_hours == 0  # UTC days, same raw file as phase 1a
    assert build_edge_request(cfg) is None
    assert cfg.land_sea_mask_threshold == 0.5
    # One land-sea mask file per region, shared by every config.
    assert cfg.paths.raw_land_sea_mask == (
        PROJECT_ROOT / "data" / "raw" / "era5_land_sea_mask_36N_112E_25N_123E.nc"
    )

    # The CSVs tracked in git are exactly the files this config writes,
    # so a re-run shows any change in results as a git diff.
    for csv in (cfg.paths.city_tmax_csv, cfg.paths.city_heat_index_csv, cfg.paths.area_fraction_csv):
        assert csv.exists(), f"{csv.relative_to(PROJECT_ROOT)} is not a tracked output"

    # Figures go to the version folder that already holds this run's figures.
    assert cfg.paths.fig_area_fraction.exists()


@pytest.mark.parametrize("utc_name, bjt_name", CONFIG_PAIRS)
def test_beijing_config_has_its_own_files(utc_name, bjt_name):
    utc = load_config(CONFIG_DIR / utc_name)
    bjt = load_config(CONFIG_DIR / bjt_name)

    assert bjt.utc_offset_hours == 8
    assert (bjt.start_date, bjt.end_date, bjt.cds_area) == (utc.start_date, utc.end_date, utc.cds_area)
    assert bjt.cities == utc.cities
    assert bjt.land_sea_mask_threshold == utc.land_sea_mask_threshold

    # Never mix Beijing-time and UTC files: the only shared file is the
    # time-invariant land-sea mask.
    utc_files = set(vars(utc.paths).values())
    shared = {path for path in vars(bjt.paths).values() if path in utc_files}
    assert shared == {utc.paths.raw_land_sea_mask}
    assert bjt.paths.raw_era5.name == f"era5_east_china_{utc.file_tag}_bjt.nc"


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


@pytest.mark.parametrize("threshold", ["0", "1.5"])
def test_config_rejects_bad_land_sea_mask_threshold(tmp_path: Path, threshold):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        (CONFIG_DIR / "july_2022.yaml").read_text(encoding="utf-8")
        .replace("land_sea_mask_threshold: 0.5", f"land_sea_mask_threshold: {threshold}"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_config(bad)


@pytest.mark.parametrize("offset", ["8.5", "15", "'8'", "true", "false"])
def test_config_rejects_bad_utc_offset(tmp_path: Path, offset):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        (CONFIG_DIR / "july_2022_bjt.yaml").read_text(encoding="utf-8")
        .replace("utc_offset_hours: 8", f"utc_offset_hours: {offset}"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_config(bad)


def test_land_sea_mask_request():
    request = build_land_sea_mask_request(load_config(CONFIG_DIR / "july_2022.yaml"))

    assert request["variable"] == ["land_sea_mask"]
    # Time-invariant field: a single time step is enough.
    assert (request["year"], request["month"], request["day"], request["time"]) == (
        ["2022"], ["07"], ["01"], ["00:00"]
    )
    assert request["area"] == [36, 112, 25, 123]


def test_download_request_days():
    sample = build_request(load_config(CONFIG_DIR / "sample_2022_07_01_07.yaml"))
    july = build_request(load_config(CONFIG_DIR / "july_2022.yaml"))

    assert sample["day"] == [f"{d:02d}" for d in range(1, 8)]
    assert july["day"] == [f"{d:02d}" for d in range(1, 32)]
    assert july["year"] == ["2022"] and july["month"] == ["07"]
    assert len(july["time"]) == 24
    assert july["area"] == [36, 112, 25, 123]


@pytest.mark.parametrize(
    "key, line, hint",
    [
        ("utc_offset_hours", "utc_offset_hours: 8", "0 for UTC days"),
        ("land_sea_mask_threshold", "land_sea_mask_threshold: 0.5", "0.5"),
    ],
)
def test_config_missing_phase_1b_key_is_explained(tmp_path: Path, key, line, hint):
    # An older (v0.2) config without the key gets a ValueError that names
    # the key and the value to use, not a bare KeyError and no silent default.
    text = (CONFIG_DIR / "july_2022_bjt.yaml").read_text(encoding="utf-8")
    assert line in text
    old = tmp_path / "old.yaml"
    old.write_text(text.replace(line, ""), encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        load_config(old)
    message = str(excinfo.value)
    assert key in message and hint in message

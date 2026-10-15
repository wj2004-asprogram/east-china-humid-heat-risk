"""
config.py

Load a run configuration (period, region, cities) from a YAML file and
derive every input/output path of the workflow from it.

All file names are built here, so no other module hard-codes a path,
a date, or a region.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Keys added in phase 1b. They have no default: an older config must say
# explicitly which behaviour it wants. (meaning, value to use)
PHASE_1B_KEYS = {
    "utc_offset_hours": (
        "daily values are calendar days in UTC + this many hours",
        "0 for UTC days as in v0.2, 8 for Beijing time",
    ),
    "land_sea_mask_threshold": (
        "area fractions count land cells only, i.e. where the ERA5 land-sea mask is >= this",
        "0.5 as in every config of this project (counting sea cells as in v0.2 is no longer possible)",
    ),
}


@dataclass(frozen=True)
class City:
    """A city point compared at the nearest ERA5 grid point."""

    name: str
    lat: float  # degrees north
    lon: float  # degrees east
    color: str  # matplotlib colour used in city comparison plots


@dataclass(frozen=True)
class Paths:
    """All input/output files of one run."""

    # Data
    raw_era5: Path
    raw_land_sea_mask: Path
    daily_tmax_nc: Path
    humid_heat_nc: Path
    heat_risk_nc: Path
    city_tmax_csv: Path
    city_heat_index_csv: Path
    area_fraction_csv: Path

    # Figures
    fig_mean_tmax: Path
    fig_city_tmax: Path
    fig_mean_heat_index: Path
    fig_city_heat_index: Path
    fig_area_fraction: Path
    fig_heat_index_map: Path
    fig_category_map: Path


@dataclass(frozen=True)
class RunConfig:
    """Settings for one workflow run, e.g. the July 2022 case study."""

    name: str
    start_date: date
    end_date: date
    north: float
    west: float
    south: float
    east: float
    cities: tuple
    utc_offset_hours: int  # daily values are calendar days in UTC + this offset
    land_sea_mask_threshold: float  # land where ERA5 land_sea_mask >= this
    file_tag: str
    paths: Paths

    @property
    def period_label(self) -> str:
        """Period as used in titles and metadata, e.g. '2022-07-01 to 2022-07-31'."""
        return f"{self.start_date.isoformat()} to {self.end_date.isoformat()}"

    @property
    def cds_area(self) -> list:
        """Region in the order expected by the CDS API: [North, West, South, East]."""
        return [self.north, self.west, self.south, self.east]


def build_paths(file_tag: str, region: dict, figures_dir: Path, project_root: Path) -> Paths:
    """
    Derive every file path of a run from its file tag.

    The land-sea mask does not depend on the period, so its file is named
    after the region and shared by every run over the same region.
    """
    raw = project_root / "data" / "raw"
    processed = project_root / "data" / "processed"
    t = file_tag
    area = "{north:g}N_{west:g}E_{south:g}N_{east:g}E".format(**region)

    return Paths(
        raw_era5=raw / f"era5_east_china_{t}.nc",
        raw_land_sea_mask=raw / f"era5_land_sea_mask_{area}.nc",
        daily_tmax_nc=processed / f"daily_tmax_{t}.nc",
        humid_heat_nc=processed / f"humid_heat_{t}.nc",
        heat_risk_nc=processed / f"heat_risk_categories_{t}.nc",
        city_tmax_csv=processed / f"city_daily_tmax_{t}.csv",
        city_heat_index_csv=processed / f"city_daily_heat_index_{t}.csv",
        area_fraction_csv=processed / f"daily_heat_risk_area_fraction_{t}.csv",
        fig_mean_tmax=figures_dir / f"east_china_mean_daily_tmax_{t}.png",
        fig_city_tmax=figures_dir / f"east_china_city_daily_tmax_{t}.png",
        fig_mean_heat_index=figures_dir / f"east_china_mean_daily_heat_index_{t}.png",
        fig_city_heat_index=figures_dir / f"east_china_city_daily_heat_index_{t}.png",
        fig_area_fraction=figures_dir / f"east_china_heat_risk_area_fraction_{t}.png",
        fig_heat_index_map=figures_dir / f"east_china_heat_index_map_hottest_day_{t}.png",
        fig_category_map=figures_dir / f"east_china_heat_risk_category_map_hottest_day_{t}.png",
    )


def _require_phase_1b_key(raw: dict, key: str):
    """Return raw[key], or raise a ValueError that explains a missing phase 1b key."""
    if key not in raw:
        meaning, value = PHASE_1B_KEYS[key]
        raise ValueError(
            f"Config key '{key}' is missing ({meaning}). "
            f"Add '{key}: <value>' to the config; use {value}."
        )
    return raw[key]


def load_config(config_path, project_root: Path = PROJECT_ROOT) -> RunConfig:
    """
    Read a YAML run configuration.

    Parameters
    ----------
    config_path : str or Path
        Path to a file in `configs/`.
    project_root : Path
        Base directory for data/ and figures/ (defaults to the repository root).

    Returns
    -------
    RunConfig
    """
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    start_date = date.fromisoformat(str(raw["period"]["start"]))
    end_date = date.fromisoformat(str(raw["period"]["end"]))
    if end_date < start_date:
        raise ValueError(f"Period end {end_date} is before start {start_date}.")

    region = {k: float(raw["region"][k]) for k in ("north", "west", "south", "east")}
    if region["south"] >= region["north"] or region["west"] >= region["east"]:
        raise ValueError(f"Invalid region {region}: need south < north and west < east.")

    cities = tuple(
        City(name=c["name"], lat=float(c["lat"]), lon=float(c["lon"]), color=c["color"])
        for c in raw["cities"]
    )
    # A city outside the region would silently snap to an edge grid cell.
    for city in cities:
        if not (region["south"] <= city.lat <= region["north"]
                and region["west"] <= city.lon <= region["east"]):
            raise ValueError(
                f"City {city.name} ({city.lat}°N, {city.lon}°E) is outside the region {region}."
            )

    utc_offset_hours = _require_phase_1b_key(raw, "utc_offset_hours")
    # bool is a subclass of int in Python: YAML `true` would become UTC+1.
    if (isinstance(utc_offset_hours, bool) or not isinstance(utc_offset_hours, int)
            or not -12 <= utc_offset_hours <= 14):
        raise ValueError(
            f"utc_offset_hours must be a whole number of hours from -12 to 14, got {utc_offset_hours!r}."
        )

    lsm_threshold = float(_require_phase_1b_key(raw, "land_sea_mask_threshold"))
    if not 0.0 < lsm_threshold <= 1.0:
        raise ValueError(f"land_sea_mask_threshold must be in (0, 1], got {lsm_threshold}.")

    file_tag = str(raw["file_tag"])
    figures_dir = project_root / raw["figures_dir"]

    return RunConfig(
        name=raw["name"],
        start_date=start_date,
        end_date=end_date,
        north=region["north"],
        west=region["west"],
        south=region["south"],
        east=region["east"],
        cities=cities,
        utc_offset_hours=utc_offset_hours,
        land_sea_mask_threshold=lsm_threshold,
        file_tag=file_tag,
        paths=build_paths(file_tag, region, figures_dir, project_root),
    )

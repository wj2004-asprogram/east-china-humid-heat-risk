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


def build_paths(file_tag: str, figures_dir: Path, project_root: Path) -> Paths:
    """Derive every file path of a run from its file tag."""
    raw = project_root / "data" / "raw"
    processed = project_root / "data" / "processed"
    t = file_tag

    return Paths(
        raw_era5=raw / f"era5_east_china_{t}.nc",
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

    region = raw["region"]
    cities = tuple(
        City(name=c["name"], lat=float(c["lat"]), lon=float(c["lon"]), color=c["color"])
        for c in raw["cities"]
    )

    file_tag = str(raw["file_tag"])
    figures_dir = project_root / raw["figures_dir"]

    return RunConfig(
        name=raw["name"],
        start_date=start_date,
        end_date=end_date,
        north=float(region["north"]),
        west=float(region["west"]),
        south=float(region["south"]),
        east=float(region["east"]),
        cities=cities,
        file_tag=file_tag,
        paths=build_paths(file_tag, figures_dir, project_root),
    )

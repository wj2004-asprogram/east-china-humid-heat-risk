"""
config.py

Load a run configuration (period, region, cities) from a YAML file and
derive every input/output path of the workflow from it.

All file names are built here, so no other module hard-codes a path,
a date, or a region.

Two kinds of period:
  - a single period within one month (`period: {start, end}`), read from
    one raw file named after the config's `file_tag` (v0.1/v0.2 layout);
  - multi-year seasons (`period: {years: {first, last}, months: [...]}`),
    e.g. June-August 2000-2024, read from one raw file per year and month
    (see `RawFile`). Those names do not depend on the config, so every
    multi-year config over the region shares them.
"""

import calendar
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def cds_state_path(target: Path) -> Path:
    """Keep the CDS journal beside raw/, isolated within each project root."""
    return target.parent.parent / "cds_state" / "requests.sqlite3"

# ERA5 hourly data start in 1940.
FIRST_ERA5_YEAR = 1940

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
class RawFile:
    """
    One raw ERA5 file of a multi-year config: hourly 2m temperature and 2m
    dewpoint over the region from a single CDS request for one year and
    one month, on the given days of the month and UTC hours.

    The CDS combines the year, month, day and time lists of a request, so
    the file holds every (day, hour) pair: `utc_times`.
    """

    path: Path
    year: int
    month: int
    days: tuple  # days of the month
    hours: tuple  # UTC hours

    @property
    def utc_times(self) -> pd.DatetimeIndex:
        """Every hourly time step (UTC) the file must hold, in order."""
        return pd.DatetimeIndex(
            [datetime(self.year, self.month, d, h) for d in self.days for h in self.hours]
        )


@dataclass(frozen=True)
class Paths:
    """All input/output files of one run."""

    # Data
    raw_era5: Path | None  # single-period configs only; multi-year: RunConfig.raw_files
    raw_land_sea_mask: Path
    daily_tmax_nc: Path
    humid_heat_nc: Path
    heat_risk_nc: Path
    climatology_nc: Path
    city_tmax_csv: Path
    city_heat_index_csv: Path
    area_fraction_csv: Path
    run_info_json: Path

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
    start_date: date  # first local day of the run
    end_date: date  # last local day of the run
    north: float
    west: float
    south: float
    east: float
    cities: tuple
    utc_offset_hours: int  # daily values are calendar days in UTC + this offset
    land_sea_mask_threshold: float  # land where ERA5 land_sea_mask >= this
    file_tag: str
    paths: Paths
    # Multi-year configs only (None / empty for a single period):
    years: tuple | None = None  # every year of the run
    months: tuple | None = None  # consecutive months of each year's season
    raw_files: tuple = ()  # RawFile: monthly files, then edge files

    @property
    def is_multi_year(self) -> bool:
        """True for a multi-year seasonal config (one raw file per month)."""
        return self.months is not None

    @property
    def seasons(self) -> tuple:
        """(first day, last day) of each contiguous block of local days."""
        if not self.is_multi_year:
            return ((self.start_date, self.end_date),)
        first_month, last_month = self.months[0], self.months[-1]
        return tuple(
            (date(y, first_month, 1), date(y, last_month, calendar.monthrange(y, last_month)[1]))
            for y in self.years
        )

    @property
    def month_blocks(self) -> tuple:
        """
        (first day, last day) of the local days processed together: one
        block per calendar month of a multi-year config, the whole period
        otherwise.
        """
        if not self.is_multi_year:
            return ((self.start_date, self.end_date),)
        return tuple(
            (date(y, m, 1), date(y, m, calendar.monthrange(y, m)[1]))
            for y in self.years for m in self.months
        )

    @property
    def local_days(self) -> pd.DatetimeIndex:
        """Every local day of the run, in order (gaps between seasons excluded)."""
        return pd.DatetimeIndex(
            [day for first, last in self.seasons for day in pd.date_range(first, last, freq="D")]
        )

    @property
    def raw_input_files(self) -> list:
        """The raw ERA5 files (hourly t2m/d2m) the processing steps read."""
        if self.is_multi_year:
            return [raw_file.path for raw_file in self.raw_files]
        return [self.paths.raw_era5]

    @property
    def period_label(self) -> str:
        """
        Period as used in titles and metadata, e.g. '2022-07-01 to 2022-07-31'
        or 'June to August, 2000 to 2024'.
        """
        if self.is_multi_year:
            first, last = (calendar.month_name[m] for m in (self.months[0], self.months[-1]))
            months = first if first == last else f"{first} to {last}"
            return f"{months}, {self.years[0]} to {self.years[-1]}"
        return f"{self.start_date.isoformat()} to {self.end_date.isoformat()}"

    @property
    def day_definition(self) -> str:
        """Time zone of the daily values' calendar days: 'UTC', or e.g. 'UTC+8' (Beijing time)."""
        if self.utc_offset_hours == 0:
            return "UTC"
        return f"UTC{self.utc_offset_hours:+d}"

    @property
    def day_window_utc(self) -> str:
        """The UTC hours of local day D, e.g. '16:00 UTC on D-1 to 15:00 UTC on D' for UTC+8."""

        def utc_hour(local_hour: int) -> str:
            # Hours relative to 00:00 UTC on day D; floor division gives the day.
            hours = local_hour - self.utc_offset_hours
            day = "D" if hours // 24 == 0 else f"D{hours // 24:+d}"
            return f"{hours % 24:02d}:00 UTC on {day}"

        return f"{utc_hour(0)} to {utc_hour(23)}"

    @property
    def cds_area(self) -> list:
        """Region in the order expected by the CDS API: [North, West, South, East]."""
        return [self.north, self.west, self.south, self.east]


def build_paths(file_tag: str, region: dict, figures_dir: Path, project_root: Path,
                multi_year: bool = False) -> Paths:
    """
    Derive every file path of a run from its file tag.

    The land-sea mask does not depend on the period, so its file is named
    after the region and shared by every run over the same region. A
    multi-year run has no single raw ERA5 file (see build_raw_files).
    """
    raw = project_root / "data" / "raw"
    processed = project_root / "data" / "processed"
    t = file_tag
    area = "{north:g}N_{west:g}E_{south:g}N_{east:g}E".format(**region)

    return Paths(
        raw_era5=None if multi_year else raw / f"era5_east_china_{t}.nc",
        raw_land_sea_mask=raw / f"era5_land_sea_mask_{area}.nc",
        daily_tmax_nc=processed / f"daily_tmax_{t}.nc",
        humid_heat_nc=processed / f"humid_heat_{t}.nc",
        heat_risk_nc=processed / f"heat_risk_categories_{t}.nc",
        climatology_nc=processed / f"heat_index_climatology_{t}.nc",
        city_tmax_csv=processed / f"city_daily_tmax_{t}.csv",
        city_heat_index_csv=processed / f"city_daily_heat_index_{t}.csv",
        area_fraction_csv=processed / f"daily_heat_risk_area_fraction_{t}.csv",
        run_info_json=processed / f"run_info_{t}.json",
        fig_mean_tmax=figures_dir / f"east_china_mean_daily_tmax_{t}.png",
        fig_city_tmax=figures_dir / f"east_china_city_daily_tmax_{t}.png",
        fig_mean_heat_index=figures_dir / f"east_china_mean_daily_heat_index_{t}.png",
        fig_city_heat_index=figures_dir / f"east_china_city_daily_heat_index_{t}.png",
        fig_area_fraction=figures_dir / f"east_china_heat_risk_area_fraction_{t}.png",
        fig_heat_index_map=figures_dir / f"east_china_heat_index_map_hottest_day_{t}.png",
        fig_category_map=figures_dir / f"east_china_heat_risk_category_map_hottest_day_{t}.png",
    )


def monthly_raw_file(year: int, month: int, raw_dir: Path) -> RawFile:
    """
    Raw file with every UTC hour of one month: `era5_east_china_YYYY_MM.nc`.

    Same name and content as the raw file of a single-period UTC config
    over that whole month (configs/july_2022.yaml: era5_east_china_2022_07.nc).
    """
    n_days = calendar.monthrange(year, month)[1]
    return RawFile(
        path=raw_dir / f"era5_east_china_{year:04d}_{month:02d}.nc",
        year=year, month=month, days=tuple(range(1, n_days + 1)), hours=tuple(range(24)),
    )


def edge_raw_file(day: date, hours: range, raw_dir: Path) -> RawFile:
    """
    Raw file with some UTC hours of one day, which complete the first (or
    last) local day of a season: `era5_east_china_YYYY_MM_DD_hHH-HH.nc`
    with the first and last UTC hour, e.g. era5_east_china_2000_05_31_h16-23.nc.
    """
    return RawFile(
        path=raw_dir / f"era5_east_china_{day:%Y_%m_%d}_h{hours[0]:02d}-{hours[-1]:02d}.nc",
        year=day.year, month=day.month, days=(day.day,), hours=tuple(hours),
    )


def build_raw_files(years: tuple, months: tuple, utc_offset_hours: int, raw_dir: Path) -> tuple:
    """
    Raw files of a multi-year config: one per year and month (all UTC
    hours), then the edge hours that complete the local days at the edges
    of each season.

    East of UTC (offset h > 0), local day D starts at (24 - h):00 UTC on
    D-1. Inside a season those hours are in the previous month's file, so
    only the first day of each season needs an edge file: for Beijing time
    16:00-23:00 UTC on the day before (31 May for June-August). West of
    UTC (h < 0), the first |h| UTC hours of the day after each season are
    needed instead. There is no edge file for UTC days, or where that day
    lies in a month of the run (adjacent seasons).
    """
    monthly = [monthly_raw_file(y, m, raw_dir) for y in years for m in months]
    covered = {(f.year, f.month) for f in monthly}

    h = utc_offset_hours
    edges = []
    for y in years:
        if h > 0:
            day = date(y, months[0], 1) - timedelta(days=1)
            hours = range(24 - h, 24)
        elif h < 0:
            day = date(y, months[-1], calendar.monthrange(y, months[-1])[1]) + timedelta(days=1)
            hours = range(0, -h)
        else:
            continue
        if (day.year, day.month) not in covered:
            edges.append(edge_raw_file(day, hours, raw_dir))
    return tuple(monthly + edges)


def _parse_seasons(period: dict) -> tuple:
    """Return the (years, months) of a multi-year period, after checking them."""
    if "start" in period or "end" in period:
        raise ValueError("A period has either 'start'/'end' or 'years'/'months', not both.")

    years = period.get("years")
    if not isinstance(years, dict) or set(years) != {"first", "last"}:
        raise ValueError(f"period.years must be {{first: <year>, last: <year>}}, got {years!r}.")
    first, last = years["first"], years["last"]
    for year in (first, last):
        # bool is a subclass of int in Python: reject YAML `true`.
        if isinstance(year, bool) or not isinstance(year, int):
            raise ValueError(f"period.years: years must be whole numbers, got {year!r}.")
    if not FIRST_ERA5_YEAR <= first <= last:
        raise ValueError(
            f"period.years: need {FIRST_ERA5_YEAR} <= first <= last, got first={first}, last={last}."
        )

    months = period.get("months")
    if (not isinstance(months, list) or not months
            or any(isinstance(m, bool) or not isinstance(m, int) or not 1 <= m <= 12 for m in months)):
        raise ValueError(f"period.months must be a list of month numbers 1-12, got {months!r}.")
    # Consecutive months within one calendar year, so that each year's
    # season is one contiguous block of days.
    if months != list(range(months[0], months[0] + len(months))):
        raise ValueError(
            "period.months must be consecutive months within one calendar year, "
            f"e.g. [6, 7, 8]; got {months}."
        )
    return tuple(range(first, last + 1)), tuple(months)


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

    period = raw["period"]
    if "years" in period or "months" in period:
        years, months = _parse_seasons(period)
        start_date = date(years[0], months[0], 1)
        end_date = date(years[-1], months[-1], calendar.monthrange(years[-1], months[-1])[1])
    else:
        years = months = None
        start_date = date.fromisoformat(str(period["start"]))
        end_date = date.fromisoformat(str(period["end"]))
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
    multi_year = months is not None
    raw_files = (
        build_raw_files(years, months, utc_offset_hours, project_root / "data" / "raw")
        if multi_year else ()
    )

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
        paths=build_paths(file_tag, region, figures_dir, project_root, multi_year),
        years=years,
        months=months,
        raw_files=raw_files,
    )

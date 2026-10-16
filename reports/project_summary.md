# Project Summary: East China Humid Heat Risk

## Overview

East China Humid Heat Risk is an atmospheric data science project that uses ERA5 hourly single-level reanalysis data to analyze humid heat risk over East China.

The first version, v0.1, completed a 7-day workflow test from 2022-07-01 to 2022-07-07. v0.2 extended the workflow to the full month of July 2022. v0.3 is in progress: its first phase rebuilt the scripts as a single config-driven, tested pipeline (1a) and corrected the methods (1b): NOAA/WPC heat index, risk area fractions over land only, Beijing-time calendar days, and explicit handling of missing data. The results below are the v0.3 outputs. Multi-year summers and a machine learning module follow (see `docs/ROADMAP.md`).

The goal of this project is to demonstrate a reproducible and extensible workflow for converting ERA5 temperature and dewpoint data into humid heat risk indicators, city-level comparisons, area-fraction statistics, and spatial maps.

This project should currently be interpreted as a monthly case study and workflow demonstration. It does not make long-term climate trend claims.

## Study Area and Period

The current analysis covers an East China sample domain:

- North: 36°N
- South: 25°N
- West: 112°E
- East: 123°E

Current period:

- 2022-07-01 to 2022-07-31
- Hourly ERA5 data, aggregated to Beijing calendar days (UTC+8); the same month with UTC days is kept for comparison

Selected city points:

- Nanjing: 32.06°N, 118.79°E
- Shanghai: 31.23°N, 121.47°E
- Hangzhou: 30.27°N, 120.15°E
- Huai'an: 33.61°N, 119.02°E

## Data

The project uses ERA5 hourly single-level reanalysis data from the Copernicus Climate Data Store.

Variables used:

- 2m temperature
- 2m dewpoint temperature
- land-sea mask (time-invariant, used to restrict area fractions to land)

Large NetCDF files are excluded from Git tracking through `.gitignore`. Scripts, reports, figures, and processed CSV outputs are tracked.

## Workflow

The v0.3 workflow follows this sequence, run by `python -m src.pipeline --config configs/<run>.yaml`:

```text
ERA5 hourly data (UTC)
→ Kelvin to Celsius conversion
→ relative humidity calculation
→ hourly NOAA/WPC heat index
→ daily maxima over local calendar days (Beijing time, UTC+8, or UTC)
→ city-level comparison (nearest ERA5 grid point)
→ humid heat risk classification (missing data kept as "Missing")
→ cos(latitude)-weighted area fractions over land only
→ spatial maps of the hottest risk day (sea greyed out)
→ figures and CSV outputs
```

Each run is one YAML file in `configs/`: the 7-day sample and July 2022, each with UTC days (`sample_2022_07_01_07`, `july_2022`) and Beijing days (`*_bjt`).

## Key Results: July 2022 (Beijing days)

- On average 48.0 % of the land area was in Danger (daily maximum heat index 40.6–54.4 °C) and 45.6 % in Extreme Caution; No Risk covered 1.4 % and Caution 5.0 %. Extreme Danger was reached by a single grid cell on a single day.
- Part of the land was in Danger on every day, from 19.4 % (07-05) to 80.9 % (07-09). From 07-08 to 07-16 more than 60 % of the land was in Danger on nine consecutive days.
- The hottest risk day was 2022-07-09 (80.9 % of the land in Danger).
- City daily maximum heat index, mean / highest: Nanjing 42.5 / 49.9 °C, Shanghai 43.4 / 51.8 °C, Hangzhou 45.6 / 49.0 °C, Huai'an 39.6 / 47.7 °C. Hangzhou was in Danger on 28 of 31 days.
- With UTC days the period means and the hottest risk day are the same; daily Danger shares differ by at most 2.2 percentage points.
- Compared with v0.2, the mean Danger share rose from 37.3 % to 48.0 %, mainly because the sea is no longer counted in the area fractions. Details: `reports/phase1b_method_changes.md`.

These numbers describe one month of one year; they are not a climatology or a trend.

## Outputs

- `data/processed/daily_heat_risk_area_fraction_<tag>.csv`: daily land area share of each risk category, plus `Valid Land Coverage`.
- `data/processed/city_daily_heat_index_<tag>.csv` and `city_daily_tmax_<tag>.csv`: city daily maximum heat index and 2m temperature.
- `data/processed/run_info_<tag>.json`: the day definition (UTC or UTC+8) of the run's CSV dates.
- `figures/v0.3_july_2022_bjt/` (headline), `figures/v0.2_july_2022/` (UTC days), and the 7-day runs in `figures/v0.3_sample_7day_bjt/` and `figures/v0.1_sample_7day/`. All are produced with the v0.3 methods.

## Limitations

- One month of one year: no climatology, no trend analysis.
- ERA5 is a 0.25° reanalysis; city values are grid-cell values, not station observations.
- The NOAA heat index assumes shade and light wind, and its risk thresholds were defined for the United States.
- The regional-mean time series are plain means over all grid cells, sea included and not area-weighted.

# Project Summary: East China Humid Heat Risk

## Overview

East China Humid Heat Risk is an atmospheric data science project that uses ERA5 hourly single-level reanalysis data to analyze humid heat risk over East China.

The project has reached v0.2. The first version, v0.1, completed a 7-day workflow test from 2022-07-01 to 2022-07-07. The current version, v0.2, extends the workflow to the full month of July 2022.

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
- Hourly ERA5 data

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

Large NetCDF files are excluded from Git tracking through `.gitignore`. Scripts, reports, figures, and processed CSV outputs are tracked.

## Workflow

The v0.2 workflow follows this sequence:

```text
ERA5 hourly data
→ temperature and dewpoint preprocessing
→ Kelvin to Celsius conversion
→ relative humidity calculation
→ heat index calculation
→ daily maximum heat index extraction
→ city-level comparison
→ humid heat risk classification
→ latitude-weighted area-fraction analysis
→ spatial maps
→ figures and CSV outputs

# East China Humid Heat Risk

**Humid heat risk analysis over East China using ERA5 hourly reanalysis data.**

[![Python](https://img.shields.io/badge/python-3.x-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A beginner-friendly atmospheric data science project that implements the full pipeline — from downloading ERA5 reanalysis data to calculating humid-heat indices, classifying risk levels, and producing publication-quality visualisations.

> ⚠️ **Status:** Workflow test with a 7-day small sample (2022-07-01 to 2022-07-07). Results are illustrative only and do **not** represent long-term climate trends.

---

## Table of Contents

- [Project Description](#project-description)
- [Data Source](#data-source)
- [Study Area & Period](#study-area--period)
- [Variables](#variables)
- [Workflow](#workflow)
- [Scripts](#scripts)
- [Output Figures](#output-figures)
- [Processed Data](#processed-data)
- [How to Run](#how-to-run)
- [Key Preliminary Findings](#key-preliminary-findings)
- [Next Steps](#next-steps)

---

## Project Description

Summer humid heat — the combination of high temperature and high humidity — poses a serious health risk in East China. This project uses **ERA5 hourly single-level reanalysis data** to:

1. Calculate hourly **heat index** (NOAA / Rothfusz formula) from 2 m temperature and dewpoint.
2. Derive daily maximum heat index and classify grid cells into **five heat-risk categories**.
3. Compute **daily area fractions** of each risk category using latitude-weighted area.
4. Visualise results as regional-mean time series, city-level comparisons, spatial maps, and stacked bar charts.

## Data Source

- **Dataset:** ERA5 hourly data on single levels
- **Provider:** Copernicus Climate Data Store (CDS)
- **Access:** [`cdsapi`](https://cds.climate.copernicus.eu/api-how-to) Python client

## Study Area & Period

| Parameter | Value |
|---|---|
| **Region** | East China |
| **Latitude** | 25°N – 36°N |
| **Longitude** | 112°E – 123°E |
| **Current test period** | 2022-07-01 to 2022-07-07 (7 days) |
| **Temporal resolution** | Hourly (24 steps per day) |
| **Target period (future)** | June–August, multiple years |

### Reference Cities

| City | Latitude | Longitude |
|---|---|---|
| Nanjing | 32.06°N | 118.79°E |
| Shanghai | 31.23°N | 121.47°E |
| Hangzhou | 30.27°N | 120.15°E |
| Huai'an | 33.61°N | 119.02°E |

## Variables

| Variable | ERA5 Name | Unit (raw) | Unit (processed) |
|---|---|---|---|
| 2 m air temperature | `t2m` / `2m_temperature` | Kelvin | °C |
| 2 m dewpoint temperature | `d2m` / `2m_dewpoint_temperature` | Kelvin | °C |
| Relative humidity | *(derived)* | — | % |
| Heat index | *(derived)* | — | °C |

## Workflow

```
ERA5 NetCDF (raw, Kelvin)
        │
        ▼
[1] Convert K → °C
        │
        ▼
[2] Calculate relative humidity (August–Roche–Magnus)
        │
        ▼
[3] Calculate hourly heat index (NOAA / Rothfusz)
        │
        ▼
[4] Resample to daily maximum heat index
        │
        ▼
[5] Classify into 5 risk categories
        │
        ▼
[6] Compute latitude-weighted area fractions
        │
        ▼
[7] Generate figures & export CSV / NetCDF
```

### Heat Risk Categories

| Level | Label | Heat Index Threshold (°C) |
|---|---|---|
| 0 | No Risk | HI < 26.7 |
| 1 | Caution | 26.7 ≤ HI < 32.2 |
| 2 | Extreme Caution | 32.2 ≤ HI < 40.6 |
| 3 | Danger | 40.6 ≤ HI < 54.4 |
| 4 | Extreme Danger | HI ≥ 54.4 |

## Scripts

All scripts live in `src/` and are designed to be run in sequence.

| Script | Purpose |
|---|---|
| `download_era5.py` | Download ERA5 hourly data from CDS for the defined region and period |
| `process_temperature.py` | Convert 2 m temperature from Kelvin to °C; compute daily maximum T |
| `process_humid_heat.py` | Full pipeline: K → °C, relative humidity, hourly heat index, daily max HI |
| `calculate_indices.py` | Alternative humid-heat indices (heat index, Humidex, simplified WBGT) |
| `classify_heat_risk.py` | Classify daily max HI into risk categories; compute area-weighted fractions; stacked bar chart |
| `plot_figures.py` | Regional-mean daily maximum temperature time series |
| `plot_city_tmax.py` | Four-city comparison of daily maximum temperature + CSV export |
| `plot_heat_index_figures.py` | Regional-mean daily max heat index + four-city comparison + CSV export |
| `plot_heat_risk_maps.py` | Spatial maps: continuous heat index + discrete risk categories for the hottest-risk day |

## Output Figures

All figures are saved to `figures/`.

| Figure | Description |
|---|---|
| `east_china_mean_daily_tmax_2022_07_01_07.png` | Regional-mean daily maximum 2 m temperature (°C) |
| `east_china_city_daily_tmax_2022_07_01_07.png` | Daily maximum temperature at Nanjing, Shanghai, Hangzhou, Huai'an |
| `east_china_mean_daily_heat_index_2022_07_01_07.png` | Regional-mean daily maximum heat index (°C) |
| `east_china_city_daily_heat_index_2022_07_01_07.png` | Daily maximum heat index at the four cities |
| `east_china_heat_risk_area_fraction_2022_07_01_07.png` | Stacked bar chart of daily area fractions by risk category |
| `east_china_heat_index_map_hottest_day_2022_07_01_07.png` | Spatial map of daily max heat index on the hottest-risk day |
| `east_china_heat_risk_category_map_hottest_day_2022_07_01_07.png` | Spatial map of risk categories on the hottest-risk day |

## Processed Data

All processed outputs are saved to `data/processed/`.

| File | Content |
|---|---|
| `daily_tmax_2022_07_01_07.nc` | Daily maximum 2 m temperature (°C) |
| `humid_heat_2022_07_01_07.nc` | Hourly t2m, d2m, RH, heat index + daily max HI + daily mean RH |
| `heat_risk_categories_2022_07_01_07.nc` | Daily max heat index + integer risk category (0–4) |
| `daily_heat_risk_area_fraction_2022_07_01_07.csv` | Daily area fraction (%) per risk category |
| `city_daily_tmax_2022_07_01_07.csv` | Daily max T per city (long format) |
| `city_daily_heat_index_2022_07_01_07.csv` | Daily max HI per city (wide format) |

## How to Run

### 1. Clone and set up the environment

```bash
git clone <repo-url>
cd east-china-humid-heat-risk
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set up CDS API credentials

Follow the [CDS API setup guide](https://cds.climate.copernicus.eu/api-how-to). Place your `.cdsapirc` file in your home directory (it is git-ignored).

### 3. Download ERA5 data

```bash
python src/download_era5.py
```

This downloads a NetCDF file to `data/raw/era5_east_china_2022_07_01_07.nc`.

### 4. Process step by step

```bash
# Step 1: Temperature conversion + daily max
python src/process_temperature.py

# Step 2: Full humid-heat pipeline (RH, heat index, daily max HI)
python src/process_humid_heat.py

# Step 3: Heat risk classification + area fractions
python src/classify_heat_risk.py
```

### 5. Generate all figures

```bash
# Regional-mean daily Tmax
python src/plot_figures.py

# City comparison of daily Tmax
python src/plot_city_tmax.py

# Regional-mean + city comparison of daily max heat index
python src/plot_heat_index_figures.py

# Spatial maps: heat index + risk categories (hottest-risk day)
python src/plot_heat_risk_maps.py
```

Alternatively, explore the data interactively with Jupyter:

```bash
jupyter notebook
```

## Key Preliminary Findings

> These findings are based on **only 7 days** of data (2022-07-01 to 2022-07-07) and should not be interpreted as climatological conclusions.

- **Heat index amplification:** Adding humidity raises the apparent temperature substantially above the air temperature. On the hottest day in this sample, the daily maximum heat index exceeded 45 °C across large portions of the study area.
- **Spatial pattern:** The southern and inland areas of the domain (southern Jiangsu, Zhejiang, and inland Anhui) show generally higher heat index values, consistent with the combined effect of high temperature and high humidity during the East Asian summer.
- **Risk classification:** On peak days, a significant fraction of the domain falls into the "Extreme Caution" (32.2–40.6 °C) and "Danger" (40.6–54.4 °C) categories. The "Extreme Danger" category (≥54.4 °C) was not reached in this 7-day sample.
- **City differences:** Nanjing, Shanghai, and Hangzhou show broadly similar heat index levels in this period, while Huai'an (further north) is slightly cooler — consistent with its more northern latitude.
- **Area fraction evolution:** The daily risk profile varies from day to day, reflecting the passage of synoptic weather systems; some days are dominated by "No Risk" / "Caution" while others show a large "Danger" area.

## Next Steps

- [ ] **Extend temporal coverage:** Download and process full JJA seasons across multiple years (e.g., 1991–2020) to establish a climatological baseline.
- [ ] **Trend analysis:** Examine long-term trends in heat index and risk-category frequencies.
- [ ] **Diurnal cycle:** Analyse the sub-daily evolution of heat stress (morning vs. afternoon).
- [ ] **Population exposure:** Overlay gridded population data to estimate the number of people exposed to each risk category.
- [ ] **Additional indices:** Compare the NOAA heat index with Humidex, Wet-Bulb Globe Temperature (WBGT), and other physiological heat-stress metrics.
- [ ] **Synoptic classification:** Link high-risk days to specific weather patterns (e.g., West Pacific Subtropical High position).
- [ ] **Uncertainty quantification:** Compare results across multiple reanalysis products (e.g., ERA5 vs. JRA-55).

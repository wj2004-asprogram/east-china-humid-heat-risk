# CLAUDE.md — East China Humid Heat Risk

## Project context
- Atmospheric data science portfolio project: humid heat risk over East China from ERA5 hourly single-level reanalysis.
- Released: v0.1 (7-day test) and v0.2 (July 2022 case study). In progress: v0.3 — phases, scope and acceptance criteria are in `docs/ROADMAP.md`. Work on one phase at a time, one branch per phase.
- The README deliberately makes no long-term climate trend claims. Keep it that way unless multi-decade data and a proper trend analysis are added.

## Current setup (keep in sync with README)
- Region: 112–123°E, 25–36°N (CDS `area = [36, 112, 25, 123]`).
- Periods: one YAML file per run in `configs/`: `sample_2022_07_01_07.yaml`, `july_2022.yaml` (UTC days, phase 1a raw files) and `sample_2022_07_01_07_bjt.yaml`, `july_2022_bjt.yaml` (Beijing days, own `_bjt` raw/processed/figure files).
- Variables: ERA5 2m temperature and 2m dewpoint temperature; ERA5 land-sea mask (one file per region, `data/raw/era5_land_sea_mask_<region>.nc`).
- Daily values: local calendar days, `utc_offset_hours` (0 = UTC, 8 = Beijing). A Beijing-time raw file also holds 16:00–23:00 UTC of the day before the period. Never give a UTC raw file to a Beijing-time config (the run stops with an error) and never reuse a UTC config's `file_tag` for Beijing time.
- CSV dates: every CSV has a `date` column (`YYYY-MM-DD`, local calendar day). The `run_info` step writes `data/processed/run_info_<tag>.json` with the day definition (`UTC` or `UTC+8`); figure date axes and map titles name it.
- Area fractions: land cells only, land = `land_sea_mask >= land_sea_mask_threshold` (0.5 in both configs). Maps grey out the sea.
- Missing data: a cell-day with any missing hour is NaN (daily aggregation uses `skipna=False`) and gets category −1 "Missing", never "No Risk". Area fractions are shares of the valid land area (sum to 100 %); the CSV column `Valid Land Coverage` is the valid share of the land area (100 when nothing is missing; 0 and NaN categories when nothing is valid).
- Plotting: matplotlib only (no cartopy).

## Commands
```bash
python -m src.pipeline --config configs/july_2022_bjt.yaml         # all processing steps (Beijing days)
python -m src.pipeline --config configs/july_2022.yaml             # same, UTC days (v0.2 files)
python -m src.pipeline --config configs/july_2022.yaml --download  # fetch raw ERA5 first (needs ~/.cdsapirc)
python -m src.pipeline --config configs/july_2022.yaml --steps classify_heat_risk plot_heat_risk_maps
python -m pytest
```

## Code layout
- `src/pipeline.py` — CLI; runs the steps in order.
- `src/config.py` — loads a YAML config; **every file path is derived here**.
- `src/indices.py` — unit conversions, relative humidity, heat index (pure functions).
- `src/risk.py` — risk categories, thresholds, colours, cos(latitude) area fractions (optionally land only).
- `src/land_sea_mask.py` — reads the ERA5 land-sea mask onto the data grid as a boolean land mask.
- `src/time_utils.py` — hourly UTC data to local calendar days (drops incomplete days).
- `src/era5_utils.py` — coordinate/variable name detection (`time` vs `valid_time`, etc.).
- `src/<step>.py` — one pipeline step each, exposing `run(cfg)`.
- `tests/` — pytest; `tests/synthetic.py` builds a small ERA5-like file for end-to-end tests.

## Rules
- One code path for every period: never copy a script to change dates, region or cities — add a config file.
- No hard-coded paths, dates or regions in `src/`; they come from the config.
- Refactors must not change results. Re-run both configs and check `git diff --stat data/processed` is empty. A change that is meant to alter results gets its own commit, with a before/after note.
- Formula changes need tests against an independent reference (NWS heat index chart, MetPy, published values), never against our own output.
- Time: ERA5 timestamps are UTC. Daily aggregation is by local calendar day via `src/time_utils.local_daily` (config `utc_offset_hours`; 8 = Beijing time, day D = 16:00 UTC on D−1 to 15:00 UTC on D). Never resample hourly data to days directly. A day is complete only with exactly the 24 local hours, each once; duplicate or non-hourly timestamps raise (never de-duplicate silently). Hourly fields keep UTC timestamps.
- Machine learning (phase 3): split train/test by year, never by random shuffle.
- Keep code simple and readable: English docstrings and comments, units in variable names or attributes.
- Python virtual environment in `.venv/`; dependencies in `requirements.txt`.
- Large NetCDF files are not tracked; CSV outputs and figures are.
- Commit after each meaningful step.

## Known issues (phase 1b — see docs/ROADMAP.md)
- Tracked CSVs (`data/processed/`) and figures are still the v0.2 outputs, on purpose: phase 1b was validated on real July 2022 data in a separate worktree (results in `reports/phase1b_method_changes.md`), but how the v0.3 outputs are published is not decided yet. Do not regenerate or commit them until then.
- Regional-mean time series (`plot_figures`, `plot_heat_index_figures`) are plain means over all grid cells: no cos(latitude) weights, sea included.
- The Beijing-time raw files and the land-sea mask were downloaded in the validation worktree (`projects/ehhr-1a-verify`), not in this checkout. Do not run CDS downloads without the user's confirmation.

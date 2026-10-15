# CLAUDE.md — East China Humid Heat Risk

## Project context
- Atmospheric data science portfolio project: humid heat risk over East China from ERA5 hourly single-level reanalysis.
- Released: v0.1 (7-day test) and v0.2 (July 2022 case study). In progress: v0.3 — phases, scope and acceptance criteria are in `docs/ROADMAP.md`. Work on one phase at a time, one branch per phase.
- The README deliberately makes no long-term climate trend claims. Keep it that way unless multi-decade data and a proper trend analysis are added.

## Current setup (keep in sync with README)
- Region: 112–123°E, 25–36°N (CDS `area = [36, 112, 25, 123]`).
- Periods: one YAML file per run in `configs/` (`sample_2022_07_01_07.yaml`, `july_2022.yaml`).
- Variables: ERA5 2m temperature and 2m dewpoint temperature; ERA5 land-sea mask (one file per region, `data/raw/era5_land_sea_mask_<region>.nc`).
- Area fractions: land cells only, land = `land_sea_mask >= land_sea_mask_threshold` (0.5 in both configs). Maps grey out the sea.
- Plotting: matplotlib only (no cartopy).

## Commands
```bash
python -m src.pipeline --config configs/july_2022.yaml             # all processing steps
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
- `src/era5_utils.py` — coordinate/variable name detection (`time` vs `valid_time`, etc.).
- `src/<step>.py` — one pipeline step each, exposing `run(cfg)`.
- `tests/` — pytest; `tests/synthetic.py` builds a small ERA5-like file for end-to-end tests.

## Rules
- One code path for every period: never copy a script to change dates, region or cities — add a config file.
- No hard-coded paths, dates or regions in `src/`; they come from the config.
- Refactors must not change results. Re-run both configs and check `git diff --stat data/processed` is empty. A change that is meant to alter results gets its own commit, with a before/after note.
- Formula changes need tests against an independent reference (NWS heat index chart, MetPy, published values), never against our own output.
- Time: ERA5 timestamps are UTC. Daily aggregation is currently by UTC day (switch to Beijing time, UTC+8, is planned in phase 1b).
- Machine learning (phase 3): split train/test by year, never by random shuffle.
- Keep code simple and readable: English docstrings and comments, units in variable names or attributes.
- Python virtual environment in `.venv/`; dependencies in `requirements.txt`.
- Large NetCDF files are not tracked; CSV outputs and figures are.
- Commit after each meaningful step.

## Known issues (phase 1b — see docs/ROADMAP.md)
- Tracked CSVs (`data/processed/`) and figures are still the v0.2 outputs: phase 1b changes the methods, but no raw ERA5 data was available to re-run them. Re-run both configs and fill in `reports/phase1b_method_changes.md` (currently a TODO).
- Regional-mean time series (`plot_figures`, `plot_heat_index_figures`) are plain means over all grid cells: no cos(latitude) weights, sea included.
- Daily values use UTC days.

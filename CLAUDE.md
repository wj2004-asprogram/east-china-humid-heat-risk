# CLAUDE.md — East China Humid Heat Risk

## Project context
- Atmospheric data science portfolio project: humid heat risk over East China from ERA5 hourly single-level reanalysis.
- Released: v0.1 (7-day test) and v0.2 (July 2022 case study). v0.3 methods and phase 2 multi-year outputs are complete — phases, scope and acceptance criteria are in `docs/ROADMAP.md`. Work on one phase at a time, one branch per phase.
- The README deliberately makes no long-term climate trend claims. Keep it that way unless multi-decade data and a proper trend analysis are added.

## Current setup (keep in sync with README)
- Region: 112–123°E, 25–36°N (CDS `area = [36, 112, 25, 123]`).
- Periods: one YAML file per run in `configs/`: `sample_2022_07_01_07.yaml`, `july_2022.yaml` (UTC days, phase 1a raw files) and `sample_2022_07_01_07_bjt.yaml`, `july_2022_bjt.yaml` (Beijing days, own `_bjt` raw/processed/figure files). The README headline results are the Beijing-time July run (`figures/v0.3_july_2022_bjt/`); the UTC runs are the comparison. Multi-year (phase 2): `jja_2000_2024.yaml` (UTC days) and `jja_2000_2024_bjt.yaml` (Beijing days), `period: {years: {first, last}, months: [6, 7, 8]}`.
- Raw files: layout in `docs/raw_files.md` (keep it stable; an external acceptance check relies on it). Multi-year configs share one file per year and month, `data/raw/era5_east_china_YYYY_MM.nc` (one CDS request each, all 24 UTC hours); Beijing time adds `era5_east_china_YYYY_05_31_h16-23.nc` per summer. Single-period configs: one file per `file_tag`.
- Variables: ERA5 2m temperature and 2m dewpoint temperature; ERA5 land-sea mask (one file per region, `data/raw/era5_land_sea_mask_<region>.nc`).
- Daily values: local calendar days, `utc_offset_hours` (0 = UTC, 8 = Beijing). A single-period Beijing-time raw file also holds 16:00–23:00 UTC of the day before the period. Never give a UTC raw file to a single-period Beijing-time config (the run stops with an error) and never reuse a UTC config's `file_tag` for Beijing time. Multi-year runs are processed one month at a time (`src/hourly_input.py`), each month cut to the UTC hours of its local days, and the daily results concatenated; they save daily variables only.
- CSV dates: every CSV has a `date` column (`YYYY-MM-DD`, local calendar day). The `run_info` step writes `data/processed/run_info_<tag>.json` with the day definition (`UTC` or `UTC+8`) and, for multi-year configs, the years and months; figure date axes and map titles name it.
- Climatology (phase 3 baseline): `heat_index_climatology_<tag>.nc` holds the daily maximum heat index per year and day of year (365-day calendar). No all-years climatology is stored; phase 3 must call `src.climatology.climatology_for_years` with the training years only.
- Area fractions: land cells only, land = `land_sea_mask >= land_sea_mask_threshold` (0.5 in both configs). Maps grey out the sea.
- Missing data: a cell-day with any missing hour is NaN (daily aggregation uses `skipna=False`) and gets category −1 "Missing", never "No Risk". Area fractions are shares of the valid land area (sum to 100 %); the CSV column `Valid Land Coverage` is the valid share of the land area (100 when nothing is missing; 0 and NaN categories when nothing is valid).
- Plotting: matplotlib only (no cartopy).

## Commands
```bash
python -m src.pipeline --config configs/july_2022_bjt.yaml         # all processing steps (Beijing days)
python -m src.pipeline --config configs/july_2022.yaml             # same, UTC days (v0.2 files)
python -m src.pipeline --config configs/july_2022.yaml --download  # fetch raw ERA5 first (needs ~/.cdsapirc)
python -m src.pipeline --config configs/july_2022.yaml --steps classify_heat_risk plot_heat_risk_maps
python -m src.pipeline --config configs/jja_2000_2024_bjt.yaml --check-raw     # which raw files are complete (no download)
python -m tools.validate_multiyear --config configs/jja_2000_2024_bjt.yaml --raw-dir data/raw --output <report.json>  # independent read-only acceptance
python -m pytest
```

## Code layout
- `src/pipeline.py` — CLI; runs the steps in order.
- `src/config.py` — loads a YAML config; **every file path is derived here**.
- `src/indices.py` — unit conversions, relative humidity, heat index (pure functions).
- `src/risk.py` — risk categories, thresholds, colours, cos(latitude) area fractions (optionally land only).
- `src/land_sea_mask.py` — reads the ERA5 land-sea mask onto the data grid as a boolean land mask.
- `src/time_utils.py` — hourly UTC data to local calendar days (drops incomplete days).
- `src/hourly_input.py` — serves raw hourly data in blocks (one month for multi-year runs) and concatenates daily results.
- `src/climatology.py` — step `heat_index_climatology` and `climatology_for_years` (phase 3 baseline).
- `src/plot_utils.py` — compressed season axis for multi-year time series (seasons side by side); single-period plots keep calendar dates.
- `src/era5_utils.py` — coordinate/variable name detection (`time` vs `valid_time`, etc.).
- `src/<step>.py` — one pipeline step each, exposing `run(cfg)`.
- `tests/` — pytest; `tests/synthetic.py` builds a small ERA5-like file for end-to-end tests.
- `tests/scientific_validation/` — independent acceptance tests written by Codex (second reviewer) for the missing-data (F1) and complete-day / UTC+8 calendar (F2) rules. Expected values are fixed or hand-calculated; do not "fix" a failure there by editing the expectation — a failure means production behaviour changed.
- `tools/` — independent, read-only ERA5 acceptance tools written by Codex: `validate_era5` (single-period files), `validate_batch` (explicit per-month binding), `validate_multiyear` (the phase 2 layout of 75 monthly + 25 Beijing-time edge files + mask; reports, re-check against a previous report). They never download, modify inputs or run the pipeline; they complement, not replace, the download's own `check_raw_file`. Docs: `tools/*_VALIDATION.md`. Tests: `tests/test_era5_validation.py`, `test_batch_validation.py`, `test_monthly_batch.py`, `test_multiyear_validation.py`, `test_phase2_independent.py` (helpers in `tests/phase2_support.py`; `tests/run_phase2_demo.py` is a demo script, not a test).

## Rules
- One code path for every period: never copy a script to change dates, region or cities — add a config file.
- No hard-coded paths, dates or regions in `src/`; they come from the config.
- Refactors must not change results. Re-run all four configs and check `git diff --stat data/processed` is empty. A change that is meant to alter results gets its own commit, with a before/after note.
- Formula changes need tests against an independent reference (NWS heat index chart, MetPy, published values), never against our own output.
- Time: ERA5 timestamps are UTC. Daily aggregation is by local calendar day via `src/time_utils.local_daily` (config `utc_offset_hours`; 8 = Beijing time, day D = 16:00 UTC on D−1 to 15:00 UTC on D). Never resample hourly data to days directly. A day is complete only with exactly the 24 local hours, each once; duplicate or non-hourly timestamps raise (never de-duplicate silently). Hourly fields keep UTC timestamps.
- Machine learning (phase 3): split train/test by year, never by random shuffle. The climatology baseline uses training years only.
- CDS downloads: one request per year and month (the CDS combines the year/month/day lists). Downloads write `<name>.part`, check it and rename; never treat a `.part` file as complete.
- Keep code simple and readable: English docstrings and comments, units in variable names or attributes.
- Python virtual environment in `.venv/`; dependencies in `requirements.txt`.
- Large NetCDF files are not tracked; CSV outputs and figures are.
- Commit after each meaningful step.

## Known issues
- Tracked CSVs (`data/processed/`) and figures are the v0.3 outputs of the four single-period configs and both multi-year configs. The folders `figures/v0.1_sample_7day` and `figures/v0.2_july_2022` keep their names but were regenerated with the v0.3 methods; the old-method outputs are in the git history (`851b496`).
- Regional-mean time series (`plot_figures`, `plot_heat_index_figures`) are plain means over all grid cells: no cos(latitude) weights, sea included.
- The original single-period Beijing-time raw files remain in the validation worktree (`projects/ehhr-1a-verify`). The complete multi-year raw inputs and land-sea mask are in this checkout. Preserve the validation worktree and CDS state database; do not run CDS downloads without the user's confirmation.
- Phase 2: download and both full production runs completed on 2026-10-29. Each run covers 2,300 summer days; the per-year climatology inputs exist locally (25 years x 92 days). Six CSVs, two run-info JSONs and fourteen figures are tracked; eight NetCDF outputs remain local. See `reports/phase2_real_outputs.md`. M1-D real-data calibration and M1-C input/output acceptance are complete (see `reports/phase2_real_acceptance.md`); CC second review is pending and phase 3 awaits advisor confirmation. The performance harness remains an external tool, not integrated.

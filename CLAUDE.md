# CLAUDE.md — East China Humid Heat Risk

## Project context
- Beginner-friendly atmospheric data science portfolio project.
- Analysing summer (JJA) humid heat risk in East China using ERA5 single-level data.
- Goal: learn the full pipeline — download, process, calculate indices, visualise.

## Rules
- Keep all code simple and readable; prefer clarity over cleverness.
- Functions should have docstrings and comments in English.
- Variable names, units, and physical quantities must be clearly labelled.
- Use xarray for NetCDF I/O; matplotlib + cartopy for maps.
- No hard-coded absolute paths — use pathlib and relative-to-project paths.
- Commit after each meaningful step.

## Environment
- Python 3.x with a virtual environment (`.venv/`).
- Install dependencies from `requirements.txt`.

## Data
- Source: ERA5 hourly data on single levels from CDS (Copernicus Climate Data Store).
- Region: East China (~110°E–125°E, ~20°N–40°N).
- Period: Summer months (June–August), multiple years TBD.

## Output
- Figures saved to `figures/`.
- Processed data saved to `data/processed/`.
- Reports in `reports/`.

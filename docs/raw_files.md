# Raw ERA5 files

All raw files live in `data/raw/` (not tracked by Git). Names are built in
`src/config.py` only. **The multi-year layout below is stable:** external
checks (e.g. an ERA5 acceptance tool) may rely on it. Changing it needs a
roadmap entry.

Common to every ERA5 request (`src/download_era5.py`):

| Key | Value |
|---|---|
| Dataset | `reanalysis-era5-single-levels` |
| `product_type` | `reanalysis` |
| `variable` | `2m_temperature`, `2m_dewpoint_temperature` (land-sea mask: `land_sea_mask`) |
| `area` | `[36, 112, 25, 123]` (N, W, S, E; 0.25° grid, 45 × 45 points) |
| `data_format` / `download_format` | `netcdf` / `unarchived` |

Timestamps are UTC. Variables arrive as `t2m` and `d2m` (Kelvin) on
`valid_time`, `latitude` (descending), `longitude`.

## Multi-year configs (`period: {years, months}`, phase 2)

One CDS request per file, always a single year and a single month. The
CDS returns every combination of the requested days and hours, so each
file holds exactly `days × hours`.

| File | Request | Time steps (UTC) | Used by |
|---|---|---|---|
| `era5_east_china_YYYY_MM.nc` | year YYYY, month MM, every day, hours 00–23 | every hour of the month (720 or 744) | every multi-year config over the region |
| `era5_east_china_YYYY_MM_DD_hHH-HH.nc` | one day, hours HH to HH | (last − first + 1) hours | configs with a UTC offset |

For `jja_2000_2024.yaml` (UTC days): 75 monthly files,
`era5_east_china_2000_06.nc` … `era5_east_china_2024_08.nc`.

For `jja_2000_2024_bjt.yaml` (Beijing days): the same 75 monthly files
plus 25 edge files `era5_east_china_YYYY_05_31_h16-23.nc` (31 May,
16:00–23:00 UTC, 8 time steps). Why only these: Beijing day D runs from
16:00 UTC on D−1 to 15:00 UTC on D. Inside a summer, 16:00–23:00 UTC of
the previous day is in the previous month's file (1 July uses 30 June
from the June file), so only 1 June needs extra hours. The last Beijing
day, 31 August, ends at 15:00 UTC on 31 August, inside the August file.
In general (`build_raw_files`): east of UTC, an edge file before each
season; west of UTC, one after it; none where that day lies in a month
of the run.

`era5_east_china_2022_07.nc` is also the raw file of `july_2022.yaml`
(same request, same content).

A file at its final name is complete: the download writes
`<name>.part`, checks it (`check_raw_file`: `t2m` and `d2m` present,
time steps exactly the expected hours, each once and in order, grid
spanning the region, no missing values, all values within 180–340 K)
and only then renames it. `.part` files are
leftovers of interrupted downloads and are replaced by the next run.

Check the state of a config's raw files without downloading:

```bash
python -m src.pipeline --config configs/jja_2000_2024_bjt.yaml --check-raw
```

## Single-period configs (`period: {start, end}`, v0.1–v0.3)

| File | Content |
|---|---|
| `era5_east_china_<file_tag>.nc` | the period's days, hours 00–23; Beijing-time configs (`*_bjt`) also 16:00–23:00 UTC of the day before, merged from two requests |

## Land-sea mask

`era5_land_sea_mask_<N>N_<W>E_<S>N_<E>E.nc`, e.g.
`era5_land_sea_mask_36N_112E_25N_123E.nc`: `land_sea_mask` at one time
step (time-invariant), shared by every config over the region.

# Phase 2 — Multi-year summers

> **Status (2026-10-16):** code and tests done on branch
> `phase-2-multiyear`. Downloads started and were paused for CDS recovery
> work. Three monthly files are now complete, including 2000-07 recovered
> from its original request. No multi-year processing results yet. The new
> one-hour smoke was rejected by CDS's temporary queue limit, so the full
> queue remains stopped; see [recovery acceptance](cds_recovery_2026-10-16.md).

Goal: June–August 2000–2024 (25 summers, 2300 days) over the same
region, for UTC days and Beijing-time days. Phase 3 needs this to split
by year.

## Design

**Config format.** `period: {years: {first, last}, months: [...]}`.
Months must be consecutive within one calendar year, so each year's
season is one block of days. Single-period configs (`start`/`end`) are
unchanged.

**Raw files.** One CDS request per year and month, since the CDS combines
the year/month/day lists. Each request goes to its own file,
`era5_east_china_YYYY_MM.nc` (layout: `docs/raw_files.md`). The monthly
files hold all 24 UTC hours, so they are the same for UTC and Beijing
time, and both configs share them.

**Edge hours for Beijing time.** Beijing day D needs 16:00–23:00 UTC of
D−1. With monthly files that run without gaps through a summer, those
hours are in the previous month's file for every day except 1 June. So
only one small edge request per summer is needed, 31 May 16:00–23:00
UTC: 25 requests, not one per month. The last day, 31 August, ends at
15:00 UTC on 31 August, inside the August file. The rule is general
(`build_raw_files`): no edge file where the edge day lies in a month of
the run, e.g. for whole years.

**Resumable download.** A complete file at its final name is skipped. A
missing one is downloaded as `<name>.part`, checked (`t2m`/`d2m`, exactly
the requested hours, each once and in order, grid spanning the region,
no NaN, all values within 180–340 K) and only then renamed. Failures are reported and the next file is
tried. Existing files are never overwritten. After 3 failed requests in
a row (in the order they finish), which points to an account, licence or
CDS problem, no new request is started and the running ones finish.
Re-running resumes. `--check-raw` reports progress.

**Parallel requests.** `--download-workers N` (1–4, default 3) keeps up to
N requests in the CDS queue at once. Each file runs in its own daemon
thread with its own CDS client, so Ctrl+C or killing the process stops at
once and leaves only `.part` files. Files are started in the config's
order; N = 1 fetches them one after the other. Log lines start with the
file name (CDS messages too; progress bars are off).

**Month-by-month processing.** `src/hourly_input.py` yields one block
per calendar month, cut to the UTC hours of the block's local days.
Beijing-time July runs from 16:00 UTC on 30 June to 15:00 UTC on
31 July. Each block goes through `local_daily` with the phase 1b rules:
a missing hour makes the cell-day Missing (−1), and a day needs exactly
24 distinct hours. The daily blocks are then concatenated and checked
against the run's local days. The days between summers are in no
block, so they are never counted as missing. At most two monthly files
are in memory. Multi-year runs save daily variables only.

**Multi-year figures.** On a calendar axis the September–May gaps would
take about 75 % of the width. The daily time series (regional means and
cities) therefore put the seasons side by side (`src/plot_utils.py`):
x is the position in the run's local days, with a 10-day gap before
each later season, a light line and the year at each season start (every
2nd year for 25 summers), and one line per season. The area-fraction
chart shows one bar per season. Single-period figures are unchanged
(byte-identical PNGs for the four existing configs).

**Climatology.** `heat_index_climatology_<tag>.nc` stores the daily
maximum heat index per (year, day of year, grid cell). Day of year uses
a 365-day calendar, so 1 June is day 152 in every year. No all-years
climatology is stored. `climatology_for_years(by_year, years,
window_days=1)` averages the given years only, with an optional running
mean cut at the season edges. Missing values stay missing. This way
the phase 3 baseline can be restricted to training years.

## Checks

- **Synthetic** (`tests/test_multi_year.py`,
  `tests/test_download_multi_year.py`, `tests/test_climatology.py`):
  - Two summers (2003, 2004 leap year) split into monthly and edge files
    exactly as downloaded. Every daily maximum heat index and Tmax equals
    a reference cut from one continuous hourly series, for UTC and
    Beijing days.
  - Boundaries, tested with temperature spikes: 31 May 16:00 UTC counts
    for Beijing 1 June, 30 June 15:00/16:00 UTC for 30 June/1 July, and
    31 August 16:00 UTC falls outside the run.
  - Errors: a missing edge file, a missing hour in an edge file (F2), a
    NaN across a month boundary (F1), and a grid mismatch between files.
  - Download: a fresh download; resuming after Ctrl+C and after a network
    error; a leftover `.part`; an incomplete existing file (reported,
    not overwritten); a downloaded file with a missing hour (not
    renamed); the stop after repeated failures.
  - Parallel download (3 workers): 3 requests really run at once (each
    waits for the other two); one failed file leaves the others intact
    and is reported; after 3 failures in a row no new file starts and the
    running request still finishes; resume redoes only `.part` files;
    Ctrl+C leaves only `.part` files; 1 worker keeps the config order.
  - Climatology: leap-year alignment; changing the test years' data
    leaves a training-year climatology unchanged.
- **Real data (July 2022)**, run in a scratch directory:
  - All four single-period configs give byte-identical CSVs and identical
    NetCDF values to main `851b496`.
  - A multi-year config cut down to July 2022 reproduces `july_2022.yaml`
    and `july_2022_bjt.yaml` byte for byte. For Beijing time, the
    30 June 16–23 UTC hours were in their own edge file.
  - The real CDS file `era5_east_china_2022_07.nc` passes the raw-file
    check.

## Download estimate (to be confirmed before running)

| | UTC config | Beijing config | Both (shared monthly files) |
|---|---|---|---|
| CDS requests | 75 monthly | 75 monthly + 25 edge | 100 (+1 land-sea mask) |
| Raw data | ~0.36 GB | ~0.37 GB | ~0.37 GB |

- **Sizes.** The July 2022 monthly file is 4.9 MB (744 h); a summer is
  about 14.6 MB, and an edge file under 0.2 MB.
- **Already downloaded.** `era5_east_china_2022_07.nc` and the land-sea
  mask exist in the validation worktree and can be copied, which saves
  one request.
- **Processed outputs.** About 120 MB of NetCDF per config (not tracked).
  The CSVs come to about 0.9 MB per config.
- **Time.** On 2026-10-15 the CDS queue took 35 s to 1 h 50 min per
  request. Sequential requests therefore take between about 1 hour and
  about a week in the worst case. The download runs unattended, can be
  stopped and resumed at any time, and `--check-raw` shows the progress.
- **CDS terms.** The CDS terms of use change on 2026-10-28. According to
  the CDS notice (shown with each request since 2026-10-06), continued use
  after that date counts as accepting the new terms; users who do not
  accept must contact ECMWF Support before 2026-10-27. If requests do
  start failing, the download stops after 3 failures in a row; re-run it
  to resume.
- **Started** 2026-10-16 14:01 (user's go-ahead), Beijing-time config
  (all 100 files; the UTC config's 75 are a subset). The 2022-07 file and
  the land-sea mask were copied from the validation worktree.
- **Observed:** the first request (2000-06) took 342 s; the second was
  still queued at CDS after more than an hour. One request at a time would
  then take days, so the download was restarted with 3 workers (user's
  go-ahead, 2026-10-16); completed files were kept.

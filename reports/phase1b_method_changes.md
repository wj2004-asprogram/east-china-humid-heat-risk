# Phase 1b — Method changes and their effect on July 2022

> **Status:** validated on the real ERA5 data for July 2022 (2026-10-15).
> The numbers below come from validation runs in a separate worktree (see
> Provenance). The tracked outputs were regenerated afterwards, in a
> separate step (branch `v0.3-outputs`): `data/processed/*_2022_07.csv`
> (UTC days) reproduce the "+ land only" stage and `*_2022_07_bjt.csv`
> (Beijing days) the "+ Beijing-time days" stage (period means, all 31
> daily Danger+ values and the city table).

Each fix is a separate commit on branch `phase-1b-methods`:

1. `2eb0f82` Heat index algorithm (NOAA/WPC selection rule).
2. `2b8358c` Land-sea mask: area fractions over land only.
3. `39a815a` Beijing-time (UTC+8) calendar days, with separate `*_bjt`
   configs.
4. Review fixes: `35de607` missing data is counted as "Missing", not
   "No Risk" (new CSV column `Valid Land Coverage`); `bf73072` a local
   day must hold exactly its 24 hourly time steps.

---

## Summary

The stages are applied cumulatively: each row adds one change to the row
above. The order matters, so the differences between rows are not a
unique causal split between the fixes.

Period-mean daily area fraction (%), July 2022:

| Stage | No Risk | Caution | Extreme Caution | Danger | Extreme Danger | Days with Danger+ > 0 |
|---|---|---|---|---|---|---|
| v0.2 (old heat index, all cells, UTC days) | 4.1 | 7.8 | 50.8 | 37.3 | 0.0 | 31/31 |
| + NOAA/WPC heat index | 2.7 | 9.2 | 49.9 | 38.2 | 0.0 | 31/31 |
| + land only | 1.7 | 5.2 | 45.2 | 48.0 | 0.0 | 31/31 |
| + Beijing-time days | 1.4 | 5.0 | 45.6 | 48.0 | 0.0 | 31/31 |

"Danger+" is Danger + Extreme Danger. Extreme Danger is 0.0 at every stage
(period mean, rounded): a single grid cell (30.75°N, 121.25°E) reaches it
on 07-13, in v0.2 as in both v0.3 runs (0.05 % of the domain area,
0.06 % of the land).

- **Heat index fix: small effect on the area fractions.** Mean Danger
  rises by 0.9 percentage points (37.3 → 38.2 %), by at most 2.5 points on
  a single day (07-24), and falls on no day. The city daily maxima do not
  change. In this humid domain the daily maximum heat index is mostly
  reached at RH ≥ 40 %, where the old and new formulas agree, so the old
  RH < 40 % switch rarely set the daily maximum. The per-point error of the
  old code was real (e.g. −2.7 °C at 40 °C and RH 30 %), it just rarely
  mattered for the daily maximum.
- **Land mask: the largest effect.** Mean Danger rises by 9.8 points
  (38.2 → 48.0 %), by 4.0 to 16.7 points per day (most on 07-09). The sea
  is 22 % of the domain (cos(latitude) weighted); sea cells almost never
  reach Danger, so counting them diluted the land signal.
- **Beijing-time days: small effect.** The period mean is unchanged
  (48.0 %); daily differences run from −1.5 points (07-22) to +2.2 points
  (07-17). City period means change by at most 0.2 °C.
- **Review fixes (missing data, complete days): no effect on this data**,
  which has no missing values or time defects. They protect later runs,
  e.g. the multi-year downloads of phase 2.
- **Hottest risk day** (used for the maps): 2022-07-09 at every stage, with
  a Danger+ share of 63.8 → 64.2 → 80.9 → 80.9 %.

---

## Provenance

- **Code:** `cc969fa` (v0.2 baseline, `main`) → `2eb0f82` (heat index) →
  `2b8358c` (land mask) → `39a815a` (Beijing time) → `bf73072` (review
  fixes). Each commit was run with the full pipeline on the same real
  files, in the worktree `D:\Claudecode_workspace\projects\ehhr-1a-verify`.
  Up to `2b8358c` the UTC config `configs/july_2022.yaml` was used; from
  `39a815a` on, the Beijing-time config `configs/july_2022_bjt.yaml`.
- **Environment:** Python 3.13.1, numpy 2.5.4, pandas 3.0.6,
  xarray 2026.9.0, matplotlib 3.11.2 (project `.venv`).
- **Inputs** (CDS, downloaded 2026-10-15):
  - `era5_east_china_2022_07.nc`: UTC days, 744 h.
  - `era5_east_china_2022_07_bjt.nc`: 752 h, 2022-06-30 16:00 to
    2022-07-31 23:00 UTC.
  - `era5_land_sea_mask_36N_112E_25N_123E.nc`.
  - The 7-day files were used for checks only.
- **Raw-data checks** (independent scripts, not pipeline code):
  - Beijing-time file: hourly, no gaps or duplicates. On the 744
    overlapping hours, t2m and d2m equal the UTC file exactly. Jump at the
    seam between the two download parts: 0.17 K. t2m 13.3–43.3 °C,
    d2m ≤ t2m, no NaN.
  - Land-sea mask: values in [0, 1], same grid as the data. Land
    (lsm ≥ 0.5) is 78.0 % of the domain, cos(latitude) weighted. Spot
    checks: Nanjing 0.94, Hefei 0.99, 28°N 123°E 0.00, 34°N 122°E 0.00.
  - Beijing-day windows: the daily maximum heat index recomputed with
    plain numpy windows (UTC D−1 16:00 to D 15:00) matches the pipeline
    with 0 error on all 31 days (and on all 7 days of the 7-day file).
- **Regression checks:**
  - `39a815a` with the UTC config reproduces `2b8358c` byte for byte
    (3 of 3 CSVs).
  - `bf73072` vs `39a815a`, both configs: identical category columns,
    `Valid Land Coverage` = 100.000000 on every day, byte-identical city
    CSVs.

---

## 1. Heat index algorithm (`2eb0f82`)

**Before (v0.2):** the Rothfusz regression was used only where T ≥ 80 °F
**and** RH ≥ 40 %. Elsewhere the simple formula was used and then averaged
with the air temperature a second time; below 70 °F the heat index was set
to the air temperature.

**After:** NOAA/WPC algorithm. The simple formula
`HI = 0.5 * (T + 61 + (T − 68) * 1.2 + RH * 0.094)` is computed first. If
the average of that value and T is ≥ 80 °F, the Rothfusz regression (with
its RH < 13 % and RH > 85 % adjustments) is used, whatever the humidity.
Otherwise the heat index is the simple formula itself. Tested against the
NWS chart (RH ≥ 40 %) and MetPy `heat_index` (all humidities). The
adjustment ranges include their bounds (80 ≤ T ≤ 112 °F and
80 ≤ T ≤ 87 °F), as in the NWS calculator (PR #2 review); this only matters
at exactly 80 °F with RH > 85 %. At 80 °F and RH < 13 % the simple formula
is used, so the dry-air adjustment never applies there (MetPy also adjusts
the simple formula in that case and gives 77.9 instead of 78.2 °F).

Formula-only effect. Heat index in °C, old (phase 1a) → new; "(=)" means
unchanged. The new values agree with MetPy 1.7.1 `heat_index` to within
1e-12 °C.

| T (°C) | RH 10 % | RH 20 % | RH 30 % | RH 39.9 % | RH 40 % | RH 60 % |
|---|---|---|---|---|---|---|
| 30 | 29.7 → 27.9 | 29.8 → 28.2 | 29.9 → 28.8 | 30.0 → 29.7 | 29.7 (=) | 32.8 (=) |
| 33 | 32.8 → 30.2 | 32.9 → 30.9 | 33.1 → 32.0 | 33.2 → 33.8 | 33.8 (=) | 39.5 (=) |
| 35 | 34.9 → 31.9 | 35.0 → 33.0 | 35.2 → 34.7 | 35.3 → 37.2 | 37.2 (=) | 45.1 (=) |
| 38 | 38.1 → 34.7 | 38.2 → 36.7 | 38.3 → 39.4 | 38.4 → 43.4 | 43.4 (=) | 55.0 (=) |
| 40 | 40.2 → 36.7 | 40.3 → 39.4 | 40.4 → 43.1 | 40.5 → 48.2 | 48.3 (=) | 62.6 (=) |

What this means:

- The old rule did **not** simply underestimate dry-air heat index. For
  very dry air (RH ≲ 25 %) it **overestimated** it (e.g. 35 °C, 10 %:
  34.9 → 31.9 °C), because NOAA's regression gives a heat index below the
  air temperature there. For RH between about 30 % and 40 % it
  **underestimated** it, by up to ~8 °C just below RH = 40 % at 40 °C.
- The old jump at RH = 40 % (1.9 °C at 35 °C, 7.7 °C at 40 °C) is gone.
- For RH ≥ 40 % and hot air nothing changes.
- Cool air changes slightly: the heat index is now the simple formula
  rather than the air temperature below 21 °C (15 °C, 60 %: 15.0 → 14.1 °C)
  or the simple formula averaged with T a second time between 21 and
  27 °C (24 °C, 50 %: 23.9 → 23.8 °C). These cells stay in "No Risk".
- Known difference from MetPy and the NWS online calculator: they switch
  to the regression when the simple formula itself is ≥ 79 °F, while the
  WPC equation page (followed here) uses the average of the simple formula
  and T ≥ 80 °F. The two rules differ only for 25.0–27.3 °C at high
  humidity, by up to 1.3 °C (26.1 °C, RH 93 %). That is right at the
  No Risk / Caution threshold (26.7 °C), so a few cool, humid days could
  be classified differently. The MetPy tests avoid this band.

**Effect on July 2022:** mean Danger 37.3 → 38.2 % (+0.9 points), No Risk
4.1 → 2.7 %, Caution 7.8 → 9.2 %. Danger+ rose on 22 of 31 days (by at
most 2.5 points, 07-24), stayed the same on 9 and fell on none (daily
table below). The city
daily maxima are unchanged (city table below). The daily maximum heat
index is mostly reached at RH ≥ 40 %, so the old RH < 40 % rule rarely
decided it, although its per-point error was real (e.g. −2.7 °C at 40 °C,
RH 30 %).

---

## 2. Land-sea mask (`2b8358c`)

**Before:** area fractions (and the choice of the "hottest risk day" for
the maps) counted every grid cell of the 112–123°E, 25–36°N box, including
the East China Sea and the Yellow Sea.

**After:** the ERA5 `land_sea_mask` is downloaded once per region
(`data/raw/era5_land_sea_mask_36N_112E_25N_123E.nc`, one time step).
A grid cell is land where the mask is ≥ `land_sea_mask_threshold`
(0.5 in both configs). Area fractions are cos(latitude)-weighted fractions
of the land area, so each day still sums to 100 %. The mask is saved as
`land_mask` in `heat_risk_categories_<tag>.nc`. The maps still show the
sea, greyed out. Regional-mean time series are unchanged (all cells).

**Effect on July 2022: the largest of the three fixes.** Land is 78.0 % of
the domain and sea 22 % (cos(latitude) weighted). Mean Danger rises from
38.2 to 48.0 % (+9.8 points); No Risk falls from 2.7 to 1.7 % and Caution
from 9.2 to 5.2 %. Per day, Danger+ rises by 4.0 to 16.7 points (most on
07-09: 64.2 → 80.9 %). Sea cells almost never reach Danger, so counting
them diluted the land signal. The city points and the hottest risk day
(07-09) are not affected.

---

## 3. Beijing-time days (UTC+8) (`39a815a`)

**Before:** daily maximum temperature, daily maximum heat index and daily
mean RH were computed over UTC days, i.e. 08:00 to 07:00 Beijing time.

**After:** daily values are computed over local calendar days set by
`utc_offset_hours`. New Beijing-time configs (`july_2022_bjt.yaml`,
`sample_2022_07_01_07_bjt.yaml`, `utc_offset_hours: 8`) use Beijing days:
day D runs from 16:00 UTC on D−1 to 15:00 UTC on D. Their download also
fetches 16:00–23:00 UTC of the day before the period (a separate CDS
request, merged into the same raw file). They have their own file tag
(`2022_07_bjt`), so raw, processed and figure files never mix with the
UTC files. The v0.1/v0.2 configs keep UTC days (`utc_offset_hours: 0`)
and the phase 1a raw files. Days without all 24 hours are dropped, so the
trailing partial day (16:00–23:00 UTC on the last day) is not used. If a
day of the period is incomplete, for example because a UTC raw file was
given to a Beijing-time config, the run stops with an error.

Checks: on synthetic data, the day boundaries (15:00 UTC belongs to
Beijing day D, 16:00 UTC to D+1), the first day of July using
16:00–23:00 UTC on 2022-06-30, a month boundary, the year boundary of the
download request, and a full run of `sample_2022_07_01_07_bjt.yaml`
checked day by day. On the real data, the 31 Beijing-day maxima were
recomputed independently with 0 error (see Provenance).

**Effect on July 2022: small.** The period-mean Danger share is unchanged
(48.0 %); No Risk 1.7 → 1.4 %, Caution 5.2 → 5.0 %, Extreme Caution
45.2 → 45.6 %. Daily Danger+ differences run from −1.5 points (07-22) to
+2.2 points (07-17); 28 of 31 days change by less than 0.5 points (19 not
at all). The
afternoon maximum (about 14:00–16:00 Beijing time) falls on the same date
under both conventions, so most daily maxima stay the same. City period
means change by at most 0.2 °C (Shanghai −0.1, Huai'an −0.2 °C); city
maxima are unchanged.

---

## 4. Review fixes: missing data and complete days (`35de607`, `bf73072`)

### Missing data

**Before:** a grid cell whose daily maximum heat index was missing (NaN)
reached no threshold and was counted as "No Risk"; a day without any data
was reported as 100 % No Risk. Daily maxima and means skipped missing
hours, i.e. they could come from a partial day.

**After:** daily aggregation uses `skipna=False`, so a cell-day with any
missing hour is NaN. NaN gets category −1 "Missing". Area fractions are
shares of the valid land area (land cells with data, cos(latitude)
weighted) and still sum to 100 %. The new CSV column `Valid Land Coverage`
is the valid share of the land area: 100 when nothing is missing, 0 (with
empty category columns) when nothing is valid. The hottest risk day for
the maps uses the same valid-area shares and skips days without data.
Maps draw missing cells in dark grey.

### Complete days

**Before:** a local day counted as complete if it had 24 time stamps, so a
day with one hour missing and another hour duplicated was accepted.

**After:** a local day is complete only if it holds exactly the 24 hourly
time steps 00:00–23:00 local time, each once. Duplicate or non-hourly time
steps anywhere in the input raise an error naming them; they are never
dropped silently. Incomplete days outside the period are still dropped,
and an incomplete day inside the period still stops the run.

**Effect on July 2022: none.** The real files have no missing values, gaps
or duplicates. `Valid Land Coverage` is 100.000000 on every day, the
category columns equal those of `39a815a`, and the city CSVs are
byte-identical. The fixes protect later runs, e.g. the multi-year
downloads of phase 2.

---

## Daily Danger + Extreme Danger share (%)

Share of the area (v0.2 and "+ heat index": all cells; from "+ land
only": land cells) in Danger or Extreme Danger, per day. Dates are UTC
days for the first three columns and Beijing days for the last one.

| Date | v0.2 | + heat index | + land only | + Beijing days |
|---|---|---|---|---|
| 2022-07-01 | 29.9 | 29.9 | 38.3 | 38.3 |
| 2022-07-02 | 28.3 | 28.3 | 36.3 | 36.2 |
| 2022-07-03 | 21.2 | 21.2 | 27.2 | 27.2 |
| 2022-07-04 | 16.9 | 16.9 | 21.6 | 21.6 |
| 2022-07-05 | 15.6 | 15.6 | 19.6 | 19.4 |
| 2022-07-06 | 21.0 | 21.0 | 26.4 | 26.4 |
| 2022-07-07 | 21.7 | 21.7 | 27.1 | 26.7 |
| 2022-07-08 | 49.3 | 49.5 | 62.0 | 61.5 |
| 2022-07-09 | 63.8 | 64.2 | 80.9 | 80.9 |
| 2022-07-10 | 59.4 | 59.8 | 74.8 | 74.8 |
| 2022-07-11 | 59.3 | 60.1 | 75.2 | 75.2 |
| 2022-07-12 | 59.8 | 60.2 | 75.3 | 75.3 |
| 2022-07-13 | 58.6 | 59.2 | 73.2 | 73.2 |
| 2022-07-14 | 62.4 | 63.3 | 78.7 | 78.9 |
| 2022-07-15 | 58.6 | 60.6 | 76.4 | 76.4 |
| 2022-07-16 | 49.7 | 51.1 | 64.7 | 64.8 |
| 2022-07-17 | 32.1 | 32.5 | 40.5 | 42.7 |
| 2022-07-18 | 17.1 | 17.1 | 21.7 | 21.7 |
| 2022-07-19 | 18.0 | 18.0 | 22.9 | 22.9 |
| 2022-07-20 | 33.4 | 33.6 | 41.4 | 41.4 |
| 2022-07-21 | 33.3 | 34.2 | 43.1 | 43.1 |
| 2022-07-22 | 52.6 | 53.8 | 68.1 | 66.6 |
| 2022-07-23 | 32.0 | 34.2 | 40.9 | 41.0 |
| 2022-07-24 | 28.5 | 31.0 | 39.1 | 39.2 |
| 2022-07-25 | 52.4 | 54.7 | 69.4 | 69.4 |
| 2022-07-26 | 37.5 | 39.3 | 49.8 | 49.8 |
| 2022-07-27 | 46.3 | 48.6 | 61.6 | 61.6 |
| 2022-07-28 | 39.4 | 41.1 | 52.1 | 52.3 |
| 2022-07-29 | 15.9 | 17.7 | 22.6 | 22.6 |
| 2022-07-30 | 23.4 | 24.4 | 31.2 | 31.2 |
| 2022-07-31 | 19.7 | 20.5 | 26.0 | 25.9 |
| **Mean** | **37.3** | **38.2** | **48.0** | **48.0** |

The review fixes (`bf73072`) give the same values as "+ Beijing days".

## City daily maximum heat index (°C), period mean / maximum

| Stage | Nanjing | Shanghai | Hangzhou | Huai'an |
|---|---|---|---|---|
| v0.2 | 42.5 / 49.9 | 43.5 / 51.8 | 45.6 / 49.0 | 39.8 / 47.7 |
| + heat index | 42.5 / 49.9 | 43.5 / 51.8 | 45.6 / 49.0 | 39.8 / 47.7 |
| + land only | same as "+ heat index" (the land mask does not affect city points) | | | |
| + Beijing days | 42.5 / 49.9 | 43.4 / 51.8 | 45.6 / 49.0 | 39.6 / 47.7 |

---

## Reproducing these numbers

Raw files go in `data/raw/` (the land-sea mask and the Beijing-time raw
file come from `--download`, which needs a CDS account). Run the July 2022
config at each commit in a separate worktree, so the tracked v0.2 CSVs in
the main checkout are not overwritten:

```bash
git worktree add ../ehhr-verify <commit>
cd ../ehhr-verify
python -m src.pipeline --config configs/july_2022.yaml       # cc969fa, 2eb0f82, 2b8358c
python -m src.pipeline --config configs/july_2022_bjt.yaml   # 39a815a, bf73072
```

Compare `data/processed/daily_heat_risk_area_fraction_*.csv` and
`city_daily_heat_index_*.csv` between consecutive commits. Older CSVs have
no `Valid Land Coverage` column, so compare only the columns both runs
share.

The v0.3 outputs are tracked: the UTC configs regenerated the v0.2 CSVs
and figures (the old-method outputs are in the git history, `851b496`),
and the Beijing-time configs added `*_bjt.csv` and `figures/v0.3_*`.

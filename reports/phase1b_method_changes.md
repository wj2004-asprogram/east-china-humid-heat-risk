# Phase 1b — Method changes and their effect on July 2022

> **TODO — the real-data comparison has not been done yet.**
> No raw ERA5 files were available (`data/raw/` was empty) when phase 1b
> was implemented, so the three fixes were implemented and tested on
> synthetic data only (`python -m pytest`). The tracked CSVs in
> `data/processed/` and the figures in `figures/` are still the **v0.2
> outputs (old methods)**. Fill in the tables below after re-running both
> configs on real ERA5 data (procedure at the end of this note).

Each fix is a separate commit on branch `phase-1b-methods`:

1. Heat index algorithm (NOAA/WPC selection rule).
2. Land-sea mask: area fractions over land only.
3. Beijing-time (UTC+8) calendar days.

---

## 1. Heat index algorithm

**Before (v0.2):** the Rothfusz regression was used only where T ≥ 80 °F
**and** RH ≥ 40 %. Elsewhere the simple formula was used and then averaged
with the air temperature a second time; below 70 °F the heat index was set
to the air temperature.

**After:** NOAA/WPC algorithm. The simple formula
`HI = 0.5 * (T + 61 + (T − 68) * 1.2 + RH * 0.094)` is computed first. If
the average of that value and T is ≥ 80 °F, the Rothfusz regression (with
its RH < 13 % and RH > 85 % adjustments) is used, whatever the humidity.
Otherwise the heat index is the simple formula itself. Tested against the
NWS chart (RH ≥ 40 %) and MetPy `heat_index` (all humidities).

Formula-only effect (no data needed). Heat index in °C, old (phase 1a) →
new; "(=)" means unchanged. The new values agree with MetPy 1.7.1
`heat_index` to within 1e-12 °C.

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
- On real data, only hours with RH < 40 % (or cool hours) can change.
  At 38–40 °C, RH of 30–40 % now gives a much higher heat index and
  RH below about 25 % a lower one, so the hottest inland afternoons are
  where the area fractions and the daily maxima may move. How often this
  happened in July 2022 is what the table below must show.

### July 2022 daily area fractions — TODO

| Date | No Risk | Caution | Extreme Caution | Danger | Extreme Danger |
|---|---|---|---|---|---|
| TODO: before → after for each day, 2022-07-01 … 2022-07-31 | | | | | |

### City daily maximum heat index — TODO

| Date | Nanjing | Shanghai | Hangzhou | Huai'an |
|---|---|---|---|---|
| TODO: before → after (°C) | | | | |

---

## 2. Land-sea mask

**Before:** area fractions (and the choice of the "hottest risk day" for
the maps) counted every grid cell of the 112–123°E, 25–36°N box, including
the East China Sea and the Yellow Sea (roughly a fifth of the domain).

**After:** the ERA5 `land_sea_mask` is downloaded once per region
(`data/raw/era5_land_sea_mask_36N_112E_25N_123E.nc`, one time step).
A grid cell is land where the mask is ≥ `land_sea_mask_threshold`
(0.5 in both configs). Area fractions are cos(latitude)-weighted fractions
of the land area, so each day still sums to 100 %. The mask is saved as
`land_mask` in `heat_risk_categories_<tag>.nc`. The maps still show the
sea, greyed out. Regional-mean time series are unchanged (all cells).

Likely effect (to be checked): sea cells have a smaller diurnal cycle, so
their daily maximum heat index is usually lower than inland, and removing
them would shift the fractions towards the higher categories. With real
data, also record the number of land cells printed by `classify_heat_risk`.

### July 2022 daily area fractions — TODO

| Date | No Risk | Caution | Extreme Caution | Danger | Extreme Danger |
|---|---|---|---|---|---|
| TODO: fix 1 → fix 2 for each day | | | | | |

City daily maximum heat index is not affected by this fix.

---

## 3. Beijing-time days (UTC+8)

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

Validated on synthetic data only: day boundaries (15:00 UTC belongs to
Beijing day D, 16:00 UTC to D+1), the first day of July using
16:00–23:00 UTC on 2022-06-30, a month boundary (Beijing 2022-08-01 =
16:00 UTC on 07-31 to 15:00 UTC on 08-01), the year boundary of the
download request (2023-01-01 needs 2022-12-31), and a full run of
`sample_2022_07_01_07_bjt.yaml` checked day by day against the 24-hour
UTC windows.

Likely effect (to be checked): the afternoon maximum (about 14:00–16:00 Beijing time,
06:00–08:00 UTC) falls in the same day under both conventions, so most
daily maxima should not change. Changes are expected where the UTC-day
maximum came from 16:00–23:00 UTC (00:00–07:00 Beijing time on the next
day), e.g. on cooler days or when warm nights follow hot days. Daily mean RH
changes more, because a different night is included.

### July 2022 daily area fractions — TODO

| Date | No Risk | Caution | Extreme Caution | Danger | Extreme Danger |
|---|---|---|---|---|---|
| TODO: fix 2 → fix 3 for each day | | | | | |

### City daily maximum heat index — TODO

| Date | Nanjing | Shanghai | Hangzhou | Huai'an |
|---|---|---|---|---|
| TODO: fix 2 → fix 3 (°C) | | | | |

---

## How to fill this report

Only after the phase 1a real-data acceptance is done, and only with the
user's confirmation for any CDS download. Running the UTC config
overwrites the tracked `*_2022_07.csv` files, so work in a separate
worktree or restore them afterwards.

1. Fixes 1 and 2 (UTC days, the phase 1a raw file
   `data/raw/era5_east_china_2022_07.nc`; fix 2 also needs the land-sea
   mask): run `configs/july_2022.yaml` at `phase-1a-refactor` ("before"),
   at the heat index commit ("fix1") and at the land-sea mask commit
   ("fix2") (`git log --oneline phase-1a-refactor..phase-1b-methods`):

   ```bash
   git checkout <commit>
   python -m src.pipeline --config configs/july_2022.yaml
   ```

   After each run, copy `data/processed/*_2022_07.csv` to a folder outside
   the repository, e.g. `../phase1b_runs/<label>/`, then restore the
   tracked files with `git checkout -- data/processed figures`.
2. Fix 3 (Beijing days): at the tip of `phase-1b-methods`, compare the UTC
   config (`july_2022.yaml`, same as "fix2") with the Beijing-time config
   (`july_2022_bjt.yaml`, label "fix3"). The latter needs its own raw file
   `data/raw/era5_east_china_2022_07_bjt.nc`, with 16:00–23:00 UTC of
   2022-06-30 (`--download`, after confirmation). Its outputs are
   `data/processed/*_2022_07_bjt.csv`.
3. Build the before → after tables, one fix at a time, for example:

   ```python
   import pandas as pd

   def compare(before_csv, after_csv):
       before = pd.read_csv(before_csv, index_col="date").round(1)
       after = pd.read_csv(after_csv, index_col="date").round(1)
       # Days present in both runs only.
       table = (before.astype(str) + " → " + after.astype(str)).dropna()
       print("| date | " + " | ".join(table.columns) + " |")
       print("|---" * (len(table.columns) + 1) + "|")
       for date, row in table.iterrows():
           print(f"| {date} | " + " | ".join(row) + " |")

   compare("../phase1b_runs/before/daily_heat_risk_area_fraction_2022_07.csv",
           "../phase1b_runs/fix1/daily_heat_risk_area_fraction_2022_07.csv")
   compare("../phase1b_runs/before/city_daily_heat_index_2022_07.csv",
           "../phase1b_runs/fix1/city_daily_heat_index_2022_07.csv")
   # Fix 3: ../phase1b_runs/fix2/*_2022_07.csv vs ../phase1b_runs/fix3/*_2022_07_bjt.csv
   ```

4. Finally, decide with the user which outputs are tracked for v0.3 (the
   new `_bjt` CSVs and figures; whether the v0.2 UTC CSVs are regenerated
   or kept as the v0.2 record), and commit them together with this report.

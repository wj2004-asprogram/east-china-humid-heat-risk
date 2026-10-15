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
- Expected effect on July 2022: humid East China afternoons rarely have
  RH < 40 % at high temperatures, so most grid cells should not change;
  look for changes on the hottest, driest inland days.

### July 2022 daily area fractions — TODO

| Date | No Risk | Caution | Extreme Caution | Danger | Extreme Danger |
|---|---|---|---|---|---|
| TODO: before → after for each day, 2022-07-01 … 2022-07-31 | | | | | |

### City daily maximum heat index — TODO

| Date | Nanjing | Shanghai | Hangzhou | Huai'an |
|---|---|---|---|---|
| TODO: before → after (°C) | | | | |

---

## How to fill this report

1. Put the raw July 2022 ERA5 file in `data/raw/` (or run the pipeline
   with `--download`).
2. Run the July 2022 config once at the parent commit (`phase-1a-refactor`,
   label "before") and once at each fix commit
   (`git log --oneline phase-1a-refactor..phase-1b-methods`):

   ```bash
   git checkout <commit>
   python -m src.pipeline --config configs/july_2022.yaml
   ```

   After each run, copy `data/processed/*_2022_07.csv` to a folder outside
   the repository, e.g. `../phase1b_runs/<label>/`, then restore the
   tracked files with `git checkout -- data/processed figures`.
3. Build the before → after tables, one fix at a time (previous commit →
   fix commit), for example:

   ```python
   import pandas as pd

   def compare(before_csv, after_csv):
       before = pd.read_csv(before_csv, index_col="date").round(1)
       after = pd.read_csv(after_csv, index_col="date").round(1)
       table = before.astype(str) + " → " + after.astype(str)
       print("| date | " + " | ".join(table.columns) + " |")
       print("|---" * (len(table.columns) + 1) + "|")
       for date, row in table.iterrows():
           print(f"| {date} | " + " | ".join(row) + " |")

   compare("../phase1b_runs/before/daily_heat_risk_area_fraction_2022_07.csv",
           "../phase1b_runs/fix1/daily_heat_risk_area_fraction_2022_07.csv")
   compare("../phase1b_runs/before/city_daily_heat_index_2022_07.csv",
           "../phase1b_runs/fix1/city_daily_heat_index_2022_07.csv")
   ```

4. Finally, at the tip of `phase-1b-methods`, re-run both configs and
   commit the regenerated CSVs and figures together with this report.

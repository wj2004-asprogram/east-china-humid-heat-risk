# Roadmap — v0.3

Time frame: about 8 weeks from mid-October 2026 (to mid-December 2026).

Working rules:
- One phase at a time, one branch and one pull request per phase.
- Pull request reviews focus on formulas and time handling.
- Every change that alters results comes with a before/after comparison.

---

## Phase 1a — Single config-driven pipeline (weeks 1–2) · branch `phase-1a-refactor`

Goal: remove the duplicated per-period scripts **without changing any result**.

- [x] One module per step in `src/`, each exposing `run(cfg)`. Period, region, cities and file names come from `configs/*.yaml`.
- [x] Single entry point: `python -m src.pipeline --config configs/july_2022.yaml`.
- [x] Shared helpers defined once: coordinate detection, unit conversion, risk categories, cos(latitude) weights.
- [x] Remove `src/calculate_indices.py`. It was a third, different heat index implementation, no workflow called it, and its input file was never produced.
- [x] pytest: conversions, relative humidity, heat index against the NWS chart (RH ≥ 40 %), risk thresholds, area fractions, configs, and an end-to-end run on synthetic data.
- [x] Update CLAUDE.md, README (how to run), requirements.
- [ ] Re-run both configs on the real ERA5 files and check the tracked CSVs are unchanged.

Acceptance:
- `python -m pytest` passes. The RH < 40 % heat index test is a strict `xfail` until phase 1b.
- Old scripts and new pipeline give byte-identical CSVs on the same input. On synthetic data this has already been checked. On real data, `git diff --stat data/processed` must be empty after re-running both configs.
- If the old scripts themselves cannot reproduce the tracked CSVs in the current environment (different library versions), compare old vs. new in the same environment and record the versions used.

## Phase 1b — Method fixes (week 2) · branch `phase-1b-methods`

Each fix is a separate commit, with a before/after table of the July 2022 daily area fractions per category.

- [ ] **Heat index algorithm.** Follow NOAA/WPC: compute the simple formula first, then apply the Rothfusz regression (with its RH < 13 % and RH > 85 % adjustments) whenever the simple estimate is ≥ 80 °F, whatever the humidity. Tests: NWS chart, MetPy `heat_index`, and continuity across RH = 40 % (the current `xfail`).
- [ ] **Land-sea mask.** Download the ERA5 `land_sea_mask` once and compute area fractions over land only (document the threshold, e.g. lsm ≥ 0.5). Maps keep the ocean, but greyed out.
- [ ] **Beijing-time days (UTC+8).** Beijing day D runs from 16:00 UTC on D−1 to 15:00 UTC on D. The download must include the last 8 hours of the previous day.
- [ ] Short note in `reports/` quantifying how each fix changed the July 2022 results.

## Phase 2 — Multi-year summers (week 3) · branch `phase-2-multiyear`

Phase 3 needs this. One summer (92 days) cannot be split by year.

- [ ] Download June–August for 2000–2024 (at least), one CDS request per month. The download must be resumable and skip existing files. Expect long CDS queues, so run it unattended. Size is modest (two variables, about 45 × 45 grid points, hourly: well under 1 GB per variable).
- [ ] Pipeline supports multi-month periods (process month by month, concatenate daily outputs).
- [ ] Daily climatology of maximum heat index per grid cell (baseline for phase 3).

## Phase 3 — Machine learning (weeks 4–6) · direction to be confirmed with the advisor

Default, **Option B: next-day heat risk prediction.**
- Target: next-day risk category (or daily max heat index) per grid cell or per city.
- Baselines: persistence (tomorrow = today) and climatology. A model only counts if it beats both.
- Models: random forest, then gradient boosting (XGBoost / LightGBM).
- Features: lagged T, Td, heat index, RH. Add 500 hPa geopotential height (Western Pacific Subtropical High index) for physical grounding. The 2022 heatwave was a subtropical-high event.
- Split by year, e.g. train 2000–2017, validate 2018–2020, test 2021–2024. Never shuffle randomly.

Bonus, **Option A: station bias correction.** Use NOAA ISD stations in East China, correct ERA5 at station points with ML, and report RMSE before and after.

## Phase 4 — Wrap-up (weeks 7–8)

- [ ] README and figures for v0.3: what changed, results, limitations.
- [ ] Update the project description on the CV ("NOAA heat index + machine learning").

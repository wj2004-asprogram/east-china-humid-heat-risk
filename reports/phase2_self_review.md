# Phase 2 supplementary self-review — 2026-10-29

Reviewed commit: `48066f4b49be46d6c4bfc9b069939b75a6c8cbb8`, against base
`38af41caaefe23d24c55f28bb021a121f9d56a5e`. The user reassigned the review to
Codex because CC was unavailable. Codex also authored these outputs; this is
an author self-review with additional checks, not independent reviewer approval.

## Conclusion

No blocking findings in the output/documentation change. No production source,
configuration, dependency, scientific test or existing single-period output
changes were found. Push and merge still require the user's authorization.

One validation-reporting gap was closed: the earlier area audit supplied
`atol=1e-9` but left NumPy's default relative tolerance enabled. The new review
checks explicitly use `rtol=0, atol=1e-9` for category fractions, coverage and
sums. All pass; the largest recomputed category-share difference is
1.4210854715202004e-14 percentage points for both day definitions. No production
result needed correction. The previous audit file remains preserved as history.

## Supplementary checks

- All 30 processing-artifact SHA-256 values match the saved manifest (including
  the eight untracked NetCDF files). The tracked working tree matched the fixed
  review commit before these review documents were added.
- Both stored land masks equal the original ERA5 mask at threshold 0.5 on the
  exact same coordinates: 1,580 land cells. Heat-index arrays in the risk and
  humid-heat products match exactly.
- Seven selected dates for each day definition were reconstructed from their
  raw 24-hour UTC windows: 2000-06-01, 2000-07-01, 2004-08-31, 2013-07-01,
  2022-07-01, 2024-08-04 and 2024-08-31. These include a May edge, month seams,
  leap-year summers, the July overlap case and the selected highest-risk day.
- For these 28,350 grid-cell days, a standalone float64 polynomial evaluation
  of the documented heat-index method, humidity and dry/wet adjustments was
  compared with the daily maximum heat-index output. It calls no production
  index or aggregation helpers. Maximum difference: 0.00020178819350746835 C;
  all passed the preselected absolute 0.002 C tolerance (`rtol=0`). This small
  difference is consistent with float32 production arithmetic. The method is
  the same documented algorithm, not an independently authored scientific
  reference; the check does not replace existing NWS/MetPy tests.
- Recomputed median processing times and maximum RSS directly from all nine
  measured trials, excluding warmups; they match the committed summary.
- Supplementary visual inspection of the UTC hottest-day heat-index map and
  mean Tmax time series found legible units, day definitions and year labels.

Machine evidence: [phase2_self_review.json](phase2_self_review.json).
The supplementary script and its stdout are retained at
`D:\AS\ehhr-codex-output\self-review-48066f4` (`review_checks.py`, `checks.log`).
No complete pipeline or benchmark rerun was needed; output files were read-only.
No full unit-suite rerun is claimed. Full 25-summer peak memory remains unmeasured;
performance estimates retain the scope limitations in the calibration report.

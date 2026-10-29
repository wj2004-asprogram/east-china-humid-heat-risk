# Phase 2 real-data outputs — 2026-10-29

Both complete nine-step workflows ran successfully on production source
`38af41caaefe23d24c55f28bb021a121f9d56a5e`, using the existing Python 3.13.1
virtual environment. No source code or configuration was changed.

## Reproduce

From the project root, with the completed raw inputs already present:

```powershell
.\.venv\Scripts\python.exe -B -m src.pipeline --config configs/jja_2000_2024_bjt.yaml
.\.venv\Scripts\python.exe -B -m src.pipeline --config configs/jja_2000_2024.yaml
```

Packages: numpy 2.5.4, pandas 3.0.6, xarray 2026.9.0, netCDF4 1.7.4,
matplotlib 3.11.2. These runs used no download flag or CDS requests.

## Evidence

- Independent input validation: BJT PASS (2,259 checks; 101 files including
  mask); UTC PASS (1,734 checks; 76 files including mask). Both cover all
  75 local months and 2,300 local days. No failed, errored or unrun checks.
- Each output has the exact JJA 2000–2024 dates, 45 x 45 grid, and finite
  values. City Tmax is a long table: 2,300 days per city, 9,200 rows total.
  The other two CSVs have 2,300 rows each.
- Risk codes were checked against thresholds 26.7, 32.2, 40.6 and 54.4 C.
  Area fractions were independently recomputed with cosine-latitude weights
  on the stored land mask (1,580 cells). Every day has 100% valid land
  coverage and category shares sum to 100% (tolerance 1e-9 percentage points).
- Each climatology input has 25 years x 92 days (365-day calendar days
  152–243), no missing values, and exactly reproduces the daily heat-index
  array after reshaping. This is per-year input, not an all-years baseline.
- July 2022 overlap: all three CSV products match their existing same-time-zone
  single-period outputs numerically, maximum absolute difference 0 for both
  BJT and UTC. Existing tracked single-period files were not modified.
- All 101 BJT input files, including the shared mask, retain the SHA-256
  recorded before processing. The CDS journal and validation worktree were
  not modified by these commands.
- Each run produced seven PNGs. Representative BJT city time series, seasonal
  area fractions and category map were visually inspected for labels/layout.

Both configurations select 2024-08-04 as the largest combined Danger + Extreme
Danger land-area day, at 95.28315285454929%. This is a descriptive result within
these summers, not a trend estimate or a statement that all BJT/UTC values agree.

Machine-readable checks: [phase2_output_checks.json](phase2_output_checks.json).
Output sizes and byte hashes: [phase2_output_manifest.json](phase2_output_manifest.json).
NetCDF byte hashes document this environment; other encoders may produce different
bytes while retaining equal scientific values. Git newline conversion can also
change CSV byte hashes across platforms.

Full input reports, pipeline logs and the one-off output audit script are retained
locally in `D:\AS\ehhr-codex-output\phase2-production-20261029`:
`input-bjt.json/.md`, `input-utc.json/.md`, `pipeline-bjt.log`,
`pipeline-utc.log`, and `check_outputs.py`.
The audit initially assumed a wide city Tmax table; that audit assumption was
corrected to validate each city in the existing long format. Production data and
code did not need a fix.

## Tracked artifacts and remaining work

Track six CSVs, two run-info JSONs and fourteen figures, following the existing
repository policy. All eight new NetCDF outputs stay local under `data/processed/`
and remain ignored, as do raw data and the CDS state database.

This change contains outputs and documentation only. The full unit suite was not
rerun: 349 passing tests are prior evidence for the unchanged source, not a new
result of this run. Real-data input and output checks above were executed anew.
M1-D real-data performance calibration and M1-C independent acceptance are still
pending. No benchmark timing claim is inferred from this workflow run.
Phase 3 still requires advisor confirmation and training-years-only climatology.
Regional mean series still include sea cells and use unweighted grid means.

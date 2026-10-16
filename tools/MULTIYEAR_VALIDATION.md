# M1-B: actual Phase 2 file layout

This optional adapter validates Phase 2 inputs without downloading, running the
production pipeline, changing scientific methods, or altering input files. It
uses CC's years/months YAML config. Single-period validation remains in
`tools.validate_era5`; M1's explicit per-month binding tool is also retained.

## Two levels of coverage

- Physical files: a complete UTC month per `era5_east_china_YYYY_MM.nc`, shared
  between UTC and Beijing-time runs. JJA 2000–2024 requires 75 such files.
- BJT season edges: 25 `era5_east_china_YYYY_05_31_h16-23.nc` files, each with
  exactly 8 hours. One shared land-sea mask is checked separately.
- Logical months: the 75 target local-calendar months / 2,300 dates. BJT July
  needs 30 June 16–23 UTC from the June file and 1–31 July through 15 UTC from
  the July file. Its physical July file still has 744 hours, not 752.
- The last 8 UTC hours of August are outside the BJT target season. Winter
  gaps are intentional. The adapter also handles other consecutive selected
  months with the same UTC/BJT rule; UTC offsets besides 0 and 8 are rejected.

The independent physical calendar/name plan is compared with the production
`cfg.raw_files`. It is not generated using production `build_raw_files` or
`RawFile.utc_times`, so an erroneous production edge request can be detected.

## Use

Run from a clone containing the fixed Phase 2 production code and prior Codex
tools, with a new output path outside the raw directory:

```powershell
.\.venv\Scripts\python.exe -B -m tools.validate_multiyear --config configs/jja_2000_2024_bjt.yaml --raw-dir D:\approved\raw --output D:\AS\ehhr-codex-output\new-phase2-report.json
```

`D:\approved\raw` is a placeholder; use the actual agreed read-only path. The
tool does not infer permission or discover another download directory. If
`--raw-dir` is omitted, it uses this clone's configured raw directory.

JSON retains physical checks, input hashes and observed UTC timestamps, plus
each logical month and its file dependencies. Markdown gives both matrices.
Final filenames are explicit: `.part` files cannot satisfy a requirement and
are never opened. Missing final files and unavailable logical comparisons are
NOT_RUN. Corrupt files are ERROR; data/time/units/grid failures are FAIL.
Summary priority is ERROR, FAIL, INCOMPLETE, PASS; counts retain every category.
Exit codes: 0/PASS, 1/FAIL, 2/ERROR-INCOMPLETE-invalid input.

The original T-C3 scanner is reused with an optional pair of explicit required
and physical UTC indexes. Its default legacy path is unchanged; the additional
`observed_utc` and `expectation_mode` fields appear only for explicit windows.
Physical inputs are labelled UTC even when their consumer uses BJT.

Quality checks include exact timestamps, uniqueness/order/full hours, Kelvin
units, finite values, per-time/grid NaN locations, engineering temperature
limits, Td−T tolerance, regular grid and mask alignment. Defaults remain the
T-C3 policy: 0.25 degrees, 180..340 K, 0.5 K Td tolerance. Raw NaN reports FAIL
for review, while the production pipeline can correctly process such data as
Missing; these are different statements. Nothing is clipped or filled.

## Resume review and dependencies

```powershell
.\.venv\Scripts\python.exe -B -m tools.validate_multiyear --config configs/jja_2000_2024_bjt.yaml --raw-dir D:\approved\raw --previous-report D:\AS\ehhr-codex-output\new-phase2-report.json --output D:\AS\ehhr-codex-output\new-phase2-recheck.json
```

Previous evidence must have the same config bytes, physical paths and options.
Newly valid inputs are NEW_VALID; unchanged accepted files UNCHANGED_VALID.
Changes yield CHANGED_INPUT/FAIL and retain the last accepted hash across later
audits or temporarily missing files. After reviewing affected outputs, a fresh
run without a previous report can explicitly establish another baseline.

A changed June file requests review for both June and July BJT blocks because
the physical file serves both. This is conservative file-level dependency
tracking, not a claim that every dependent output value changed. A missing
season-edge file blocks June only; July and August may already be ready.
`quality_ready_files` counts passing current quality checks, while
`ready_local_months` additionally requires accepted dependencies and mask.

## Reproduce independent acceptance

```powershell
.\.venv\Scripts\python.exe -B -m tests.run_phase2_demo --output-dir D:\AS\ehhr-codex-output\new-m1b-demo
```

The directory must be new. Only synthetic 3×3 files are created. Four preserved
reports demonstrate missing edge / completion / reuse / changed shared month:
INCOMPLETE, PASS, PASS, FAIL (expected review requirement).

`tests/test_multiyear_validation.py` checks the adapter and independent inventory.
`tests/test_phase2_independent.py` executes real fixed-version production modules:
month reading, temperature/humid-heat/risk/climatology processing, city date CSVs,
run_info, and a downloader with an independent fake CDS client. It verifies
literal boundary temperatures, units and NaN propagation, unchanged accepted
files after an interrupted/invalid download, no redundant second-run retrieval,
and training-year climatology isolation including an absent training date.

No real CDS client, network request or multi-year real file is used in this
acceptance. Raw bytes are hashed before/after reading; production outputs exist
only in the approved synthetic temporary directories. A passing logical input
matrix proves input readiness, not that the production workflow has been run.
Full real-data results, background download health and multi-decade performance
still require a separate bounded run on the approved actual files.

Dependencies: this adapter imports the prior T-C3 and M1 tools. Integration
requires their approved files plus the small T-C3 extension, not only this new
module. Historical patches and evidence remain available separately.

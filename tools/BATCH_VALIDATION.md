# M1: monthly inventory and resume review

This is an independent acceptance tool. CC owns Phase 2 downloads, batching,
overlap handling and production interfaces. M1 does not execute a downloader,
merge hourly files, generate production configs, resume jobs, repair inputs or
choose a scientific method. It reuses T-C3's read-only input checks.

## Explicit expectations before file discovery

From the independent clone, create a new plan outside the raw directory:

```powershell
.\.venv\Scripts\python.exe -B -m tools.validate_batch plan --template-config configs/july_2022_bjt.yaml --years 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 --months 6 7 8 --output D:\AS\ehhr-codex-output\new-jja-plan.json
```

The result contains 75 expected months / 2,300 local days, with no file
bindings. An audit at this point is INCOMPLETE, with 75 missing bindings. The
year/month scope is authoritative: an entirely absent year remains visible.
Only explicitly selected months/years are required; no winter gap is invented.
Dates use the Gregorian calendar, including February 29 when selected.

The manifest copies region, UTC offset and land threshold from the explicit
template. Grid step defaults to 0.25 degrees (`--grid-step` overrides it).
It records the template hash but does not assume future naming conventions.
Each scope is for one region/offset/grid/threshold. Keep UTC and BJT separate.
This version supports the existing single-level t2m/d2m contract, not Z500.

Bind full-month configs explicitly after CC defines their locations. Example
fragment, with paths resolved relative to the manifest, not the shell directory:

```json
{
  "period": "2022-07",
  "config": "configs/july_2022_bjt.yaml",
  "raw_dir": "inputs",
  "daily": {
    "path": "daily/july.nc",
    "time_coord": "day",
    "variables": {"tmax": "K"}
  }
}
```

Put these objects in `bindings`; do not replace `years`/`months` with only
available files. `daily` is optional. Its coordinate name, variables and units
are illustrative, not a production interface proposal. Bind the actual CC
output after it is agreed. This validator supports gridded NetCDF daily fields,
not the city's CSV format. Unknown or duplicate fields/bindings are rejected.
Missing config bindings/files are NOT_RUN; a config with the wrong month,
region, offset or threshold is FAIL and is not used to interpret raw data.

## Audit and safe output

```powershell
.\.venv\Scripts\python.exe -B -m tools.validate_batch audit --manifest D:\AS\ehhr-codex-output\new-jja-plan.json --output D:\AS\ehhr-codex-output\new-inventory.json
```

Both JSON and Markdown reports must be new files, outside all bound raw/daily
input directories. The tool never enumerates a directory to decide completeness:
only each explicit final file is checked. A neighboring `.part` cannot satisfy
the plan and is not opened. Existing evidence and production inputs are not
overwritten. Config, raw, mask, daily and manifest content hashes are recorded
or checked for stability while reading. T-C3 records raw/mask size and mtime too.

The report retains the complete month matrix and detailed T-C3 results. It
distinguishes required BJT local-day hours from the current producer's extra
8 trailing UTC hours; the current full July layout is 752 / 744 required.
An absent or malformed file cannot turn into a full pass. Status priority is
ERROR, then FAIL, then INCOMPLETE (NOT_RUN present), otherwise PASS. Counts retain
every category. CLI exit codes are 0/PASS, 1/FAIL, 2/ERROR-INCOMPLETE-invalid input.

If daily fields are requested for any month, the global daily check requires
bindings and inspected dates for all planned months. It verifies exact local
midnight dates, order, uniqueness, month coverage, spatial grid, variable names,
dimensions and units. It does not compare daily numeric values or assume NaN
is invalid: approved missing values may occur. NA propagation and partition
equivalence are tested separately against the existing daily reducer.

## Review a resumed batch using previous evidence

```powershell
.\.venv\Scripts\python.exe -B -m tools.validate_batch audit --manifest D:\AS\ehhr-codex-output\new-jja-plan.json --previous-report D:\AS\ehhr-codex-output\new-inventory.json --output D:\AS\ehhr-codex-output\new-inventory-after-download.json
```

The identical manifest is required so a revised plan cannot silently inherit an
old acceptance. Bind all expected config paths before the initial audit, even
where the corresponding final files do not yet exist. Explicit rebinding calls
for a fresh baseline report.

- NOT_READY: requested checks could not pass.
- NEW_VALID: a month now passes and was not previously accepted.
- UNCHANGED_VALID: all checks pass again and all previously accepted input hashes
  match. It is safe to consider reuse, not proof that production outputs exist.
- CHANGED_INPUT: a previously accepted input has changed; report FAIL and require
  review of dependent outputs. This includes configuration comments because the
  policy conservatively compares exact bytes. It does not regenerate anything.

The last accepted hashes are preserved separately from current hashes. Repeating
an audit, or temporarily losing a file, cannot silently accept changed data.
After reviewing changed inputs and dependent outputs, deliberately run without
`--previous-report` to establish a new baseline; the older evidence remains.

`ready_months` counts months passing current input checks; it can include a
CHANGED_INPUT month. `resume_review_months` separately counts stale-input review
requirements. The previous report is local audit evidence, not a trusted signed
artifact. This tool reruns checks instead of treating file existence as validity.

## Synthetic evidence and pending integration

Reproduce initial absence, completion, unchanged reuse and changed accepted data:

```powershell
.\.venv\Scripts\python.exe -B -m tools.demo_batch_validation --output-dir D:\AS\ehhr-codex-output\new-m1-demo
```

The directory must not exist. All generated configs, raw/daily NetCDFs and a
deliberately incomplete `.part` are synthetic. Four reports are kept: INCOMPLETE,
PASS, PASS, FAIL (expected changed-input detection).

`tests/test_batch_validation.py` covers the inventory and report behavior.
`tests/test_monthly_batch.py` compares per-month reduction
with continuous reduction and independent literal values: UTC/BJT, summer,
leap February, year boundary, boundary pulses, missing values and padded overlap.
The references are 0..23 maxima/means (23/11.5), and isolated pulses (48/24 = 2;
96/24 = 4). No new production reference method is introduced.

Naively concatenating padded BJT raw months duplicates 8 hours at each seam and
is rejected by the current reducer. Processing each month and concatenating
its disjoint target daily dates passes. CC must choose and document its actual
production strategy; these are acceptance constraints, not a deduplication fix.

No production Phase 2 interrupted download, atomic final publication, multi-year
batch, daily numeric output or v0.3 city-CSV migration has been accepted here.
Those require the actual CC implementation, agreed paths/interfaces and another
bounded integration run. No multi-decade performance claim follows from the
small synthetic fixtures.

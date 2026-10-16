# T-C3: read-only ERA5 validation

This optional tool is independent of the production pipeline. It uses the
public `src.config.load_config` naming contract and never calls CDS, repairs
files, changes a scientific calculation, or regenerates production outputs.

## Usage

From the independent clone:

```powershell
.\.venv\Scripts\python.exe -B -m tools.validate_era5 --config configs/sample_2022_07_01_07.yaml configs/sample_2022_07_01_07_bjt.yaml --raw-dir D:\Claudecode_workspace\projects\ehhr-1a-verify\data\raw --output D:\AS\ehhr-codex-output\new-sample-report.json
```

The JSON report and a Markdown companion are created with exclusive writes.
Existing reports are not overwritten, and both reports must be outside every
input raw directory. All input files are opened read-only and closed promptly.
Each input's SHA-256, size and modification time are checked before/after reading.

Exit codes: 0 = PASS; 1 = a data check FAIL; 2 = ERROR, INCOMPLETE, or invalid CLI
input. Missing files and unavailable grid comparisons are NOT_RUN, never a full
pass. A corrupt/unreadable final file is ERROR, distinct from a scientific/data
assertion failure. JSON summary counts retain all four categories.

## Checks and reporting

- UTC timestamps: decoded datetime64, no NaT, strictly ordered, unique, on hours;
  complete required local-day coverage and the current producer file layout.
  BJT seven-day data requires 168 hours for daily values; the producer retains
  another 8 hours at the end, so the current file has 176. The month analog is
  744 required / 752 produced. Permitted edge hours are not called duplicates.
- T2m/Td: explicit Kelvin units, NaN counts by time and grid point, infinite
  values, sanity limits, and dewpoint-minus-temperature checks with locations.
- Grid: configured region bounds, longitude ascending, latitude descending,
  regular spacing, and pointwise equality with the land-sea mask (atol 1e-6 deg).
- Mask: static/single time, finite [0,1] fractions and cos(latitude)-weighted
  binary land percentage at the configured threshold. No mask is silently chosen
  from a directory listing; its exact name comes from the config.
- Multi-config inventory: within each region/offset/year, configured local dates
  from the first to last supplied date must be continuous, each once. Different
  years are independent, so winter gaps between JJA seasons are not invented.
  Each monthly file still gets its own coverage check, including boundary hours.

Provide all planned monthly configs, even if a corresponding file is absent.
The tool cannot infer an entirely omitted year or request from configs it never
received. Do not put both nested 7-day and full-month configs with the same UTC
offset into one inventory: these are overlapping analyses, not a monthly series.
UTC and BJT configs are separate inventory groups. Raw padding shared between
adjacent months is not itself a duplicate target local day.

Detailed NaN/dewpoint locations remain in JSON. Timestamp error examples are
capped (default 20), with total counts and an explicit truncation flag. Hourly
values are scanned in 24-hour chunks; NaN summary arrays scale with the number
of timestamps and grid cells. No performance claim is made for multi-decade runs.

## Explicit engineering options

Defaults: grid step 0.25 deg; Td tolerance 0.5 K; temperature sanity interval
180..340 K. The interval is a broad engineering guard against gross unit errors,
not an ERA5 validity limit, a July climatology, or a replacement scientific
method. CLI flags `--grid-step`, `--dewpoint-tolerance-k`, `--temperature-min-k`,
`--temperature-max-k` can specify another documented policy. Values and tolerances
are recorded in JSON. A plausible-but-mislabeled value cannot always be detected.

NaN presence is reported as a failed data-quality check for review. It does not
assert that the approved pipeline cannot process missing data: F1 explicitly
supports Missing/coverage reporting. The validator never clips, fills, sorts,
deduplicates, shifts timestamps or resamples inputs.

Recognized aliases are `time`/`valid_time`, `lat`/`latitude`, `lon`/`longitude`,
`t2m`/`2m_temperature`, `d2m`/`2m_dewpoint_temperature`, `lsm`/`land_sea_mask`.
Only one-dimensional regular lat/lon coordinates are supported. Extra singleton
dimensions are dropped from an in-memory view; unsupported non-singleton
dimensions produce an explicit error. Numeric time axes are never guessed UTC.

## Acceptance

`tests/test_era5_validation.py` has 31 synthetic tests, including defect location,
multi-month/year inventory, aliases, chunk consistency, incomplete/error status,
strict JSON for invalid coordinates and report/input overwrite guards. Fixtures
and reports are written only under the approved temporary output directory.

The delivered real reports cover only the five user-approved final files. CC
retains ownership of the Phase 1b real-data acceptance and method comparisons.

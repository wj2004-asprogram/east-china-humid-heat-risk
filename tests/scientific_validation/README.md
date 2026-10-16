# T-C1: independent missing-data and calendar acceptance

The specification is the user-approved CC task card `T-C1`, dated 2026-10-15.
Production code is unchanged. This suite adds 41 synthetic cases and imports
only the two target modules, so the same assertions can test old and new commits.

## Reference calculations

- Classification uses literal approved values: NaN -> -1, with unchanged finite
  thresholds 26.7, 32.2, 40.6 and 54.4 degrees Celsius.
- A two-latitude grid at 0 and 60 degrees has relative cell weights 1 and 1/2.
  In the hand-calculated example, total land weight is 2.5 and valid land weight
  is 1.5: coverage is 60%, Danger is 2/3 of valid land, No Risk is 1/3.
  A valid sea cell cannot enter either land denominator.
- With zero valid land, all five category shares must be NaN and coverage zero.
- Consecutive hourly values 0..23 have maximum 23 and mean 11.5. Fixed 24-hour
  blocks provide the reference across UTC/BJT month, year and leap-day edges.
  Any NaN invalidates only its own cell-day for both max and mean.
- BJT day D is exactly UTC D-1 16:00 through D 15:00. Missing/duplicate/off-hour
  cases are deliberately constructed; no production resampling helper builds
  expected values. Spikes on both sides verify hour ownership.

This task does not add Heat Index formula tests (T-C2 was not assigned), change
scientific methods, or repeat CC's real-data method-change analysis.

## Run

From the independent clone, using its approved environment:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
.\.venv\Scripts\python.exe -B -m pytest tests/scientific_validation -q -p no:cacheprovider --basetemp D:\AS\ehhr-codex-output\new-tc1-temp

.\.venv\Scripts\python.exe -B -m tools.run_scientific_validation --output-dir D:\AS\ehhr-codex-output\new-tc1-comparison
```

Use new output paths on each run. The runner reads local Git objects, exports
the exact risk/time modules, records their SHA-256 values and runs the same suite
on legacy `39a815a`, fixed `bf73072` and current HEAD. It never checks out, fetches,
pushes or changes a source branch. Raw pytest failures on legacy remain visible
in XML/logs; the comparison labels failures of the F1/F2 tests as expected
reproductions. Fixed/current run ordinary assertions without xfail suppression.

To target an existing export manually, set `EHHR_SCIENCE_SOURCE_ROOT` to the
directory containing its `src/risk.py` and `src/time_utils.py`. Leave that variable
unset for a normal full-suite run on the clone.

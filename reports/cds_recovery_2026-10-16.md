# Phase 2 CDS recovery acceptance — 2026-10-16

Code and mock/synthetic acceptance are complete. **Fresh-submission smoke
acceptance is not complete**: CDS rejected its one-hour job due to the
temporary dataset queue limit. The formal 100-file downloader remains
stopped. Do not describe this as a successful fresh-submission smoke.

## Audit and scope

- Working branch retained: `phase-2-multiyear`; starting commit `0cf7d32`.
  Initial working tree was clean. No branch was created or switched.
- Independent `ehhr-1a-verify` worktree remains on `codex-validation-tools`,
  clean and unchanged. No changes to 1a/1b datasets or processing outputs.
- Dependencies inspected locally: cdsapi 0.7.7, ecmwf-datastores-client
  0.5.3, multiurl 0.3.9, requests 2.34.2, pytest 9.1.1.
- Original code called `client.retrieve(dataset, request, target)` and
  deleted stale `.part` files. It had no persistent CDS IDs. Its current
  backend retries HTTP methods, including POST, by default.
- PID 31376 and its Python child 11556 were one old downloader, not two
  independent runs. Stopped only that local client after copying logs.
  No CDS request has been cancelled or deleted.

## Files changed and reasons

| File | Reason |
|---|---|
| `src/cds_download.py` | SQLite journal, submission capture, reconciliation, OS locks, retry/backoff, result transfer and no-clobber publication |
| `src/download_era5.py` | Connect existing download entry points/checks to the durable lifecycle |
| `src/config.py` | Derive project-local journal location beside `raw/` |
| `requirements.txt` | Pin the audited CDS client/backend versions; declare requests |
| `.gitignore` | Keep runtime state, locks, logs and smoke artifacts local |
| `tools/cds_jobs.py` | Offline historical log import and read-only verification/adoption |
| `tools/cds_smoke.py` | One-hour smoke outside `data/raw`, restartable using its original ID |
| `tests/test_cds_download.py` | 39 lifecycle/HTTP/process/disk-failure acceptance cases |
| `tests/test_download_multi_year.py` | Adapt existing synthetic downloader tests to the asynchronous adapter |
| `tests/test_pipeline_cli.py` | Adapt existing CLI download fake to the asynchronous adapter |
| `docs/cds_recovery.md` | Recovery behavior, operational limits and commands |
| `docs/raw_files.md` | Correct stale `.part` restart description; keep raw naming stable |
| `reports/phase2_multiyear.md` | Correct obsolete “not downloaded” status |
| `reports/cds_recovery_2026-10-16.md` | This acceptance record |

No changes to ERA5 variables, request dates/region, raw filename rules,
metrics, data processing or configuration formats. No dependency installs.

## Persistence, restart and duplicate prevention

The authoritative journal is `data/cds_state/requests.sqlite3`, local to
this checkout. Task ID = SHA256(dataset + canonical full parameters).
SQLite FULL-sync transactions commit submission intent **before** POST and
the returned ID immediately at the HTTP response boundary. Rows contain
target, parameters, ID, state, UTC timestamps, attempts/retries and errors.
Historical/duplicate IDs remain in `request_history`.

POST automatic retries are disabled; cleanup is disabled. A known ID is
queried after every restart. Queue/running errors and result-transfer
errors never lead back to submission. Network retries back off exponentially
with jitter and a persisted retry deadline. Explicit failures stop with
their ID intact. A lost submission response becomes `SUBMISSION_UNKNOWN`;
reconcile the complete paginated job listing by full parameters. Missing
listings after such an ambiguous POST never authorize another POST.

Result metadata is reacquired from the same ID. A partial asset uses HTTP
Range/If-Range when supported; a server ignoring Range restarts only that
asset transfer. Existing scientific validation runs before publication.
OS locks protect each task across processes; atomic final publication
refuses to overwrite existing files.

API behavior was checked against the installed source and the
[official ECMWF client reference](https://ecmwf.github.io/ecmwf-datastores-client/_api/datastores/Client.html).

## Real request recovery

All seven IDs from both old logs were imported and server parameters were
verified. Observed statuses at verification (not a live dashboard):

| File | Original request ID | Observation/action |
|---|---|---|
| 2000-06 | `0e72fec4-adc9-43a0-aa8e-5a5345c30066` | successful; existing file preserved |
| 2000-07 | `8105045f-6560-48f0-9380-6ccf7aa7df15` | successful; adopted and result recovered |
| 2000-07 duplicate | `4e6703f3-a606-421b-9841-7ec31c3132d9` | accepted; recorded, not cancelled |
| 2000-08 | `cb8a618b-9661-4a73-b7a5-31f97c4ecd5d` | accepted despite old local network failure; adopted |
| 2001-06 | `aa304467-5096-436c-b7dd-120c699c830a` | accepted; adopted |
| 2001-07 | `6252cc76-5faf-422b-bfc0-7df151f4b3ad` | accepted despite old local network failure; adopted |
| 2001-08 | `aeed837a-a0a3-4b43-8705-800222a89f22` | accepted; adopted |

Original logs are preserved in `data/cds_state/logs/`.

The one-hour smoke ID is `34ed4255-a54b-48b2-b815-5edc3da551cb`.
It was submitted, queued and persisted; its local Python process was then
killed. A new process recovered exactly that ID and queried it without a
new POST. CDS subsequently set it to `rejected`. The official results
endpoint returned HTTP 400 with the dataset's temporary queued-request
limit explanation. This is stored as `PERMANENT_ERROR`; **no automatic
replacement smoke was submitted**.

Separately, a recovery-only client that explicitly forbids `submit()`
downloaded 2000-07 using `8105045f-...`, validated it with `check_raw_file`,
and saved it in 191 seconds. No new server job was created for this file.
This verifies real completed-request recovery and result saving, but does
not turn the rejected one-hour smoke into a successful submission test.

## Tests and data boundaries

Final command (project's existing `.venv`):

```powershell
python -m pytest -q --tb=short --basetemp D:\codex_workspace\era5-resilience\test-tmp-project-final
```

**184 passed, 0 failed, 382 warnings, 69.46 s.** Warnings concern the
installed NumPy/xarray/NetCDF stack; no scientific code was changed to
suppress them. Includes 39 new lifecycle cases, all A–H, the original
download tests and the project's full synthetic regression suite.

`data/raw` **was touched only to add the recovered 2000-07 monthly file**
(and its temporary `.part`, removed after successful validation). No
existing file was removed, overwritten or re-downloaded. SHA256 verified
unchanged for all three original files:

- `era5_east_china_2000_06.nc`
- `era5_east_china_2022_07.nc`
- `era5_land_sea_mask_36N_112E_25N_123E.nc`

No large-scale ERA5 download was started. Real operations were limited to
querying existing IDs, one one-hour submission (rejected), and retrieving
one previously completed original monthly result. No data-processing
pipeline, metric calculation or actual 1a/1b regeneration was run.
No Clash/Mihomo, TUN, fake-IP, proxy, DNS, adapter or route settings changed.

`git diff --check` passes. Runtime journal/logs/raw files are ignored by
Git. Code changes are committed on the existing branch; no push is made.

## Remaining acceptance gate

Wait for CDS queue capacity; review the rejected smoke's recorded status
before deliberately authorizing a replacement of that terminal request.
Keep its historical ID. Complete the one-hour submit/query/result-save
smoke before restarting the 100-file downloader. Do not delete the journal
or cancel the live requests to make room. The old duplicate queue entry
has been documented and left intact.

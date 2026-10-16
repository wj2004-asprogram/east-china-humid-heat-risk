# CDS download recovery

The scientific requests, raw names, config format and public `run(cfg,
workers=...)` interface are unchanged. This uses **cdsapi 0.7.7**, including
its existing **ecmwf-datastores-client 0.5.3** backend. No network, DNS,
proxy or TUN settings are changed.

## Durable ownership

`data/cds_state/requests.sqlite3` belongs to this project/worktree and is
ignored by Git. Keep it across restarts; back it up rather than deleting it.
SQLite transactions use `synchronous=FULL`. Each row contains a SHA256 task
ID (dataset plus canonical parameters), parameters, absolute target path,
CDS ID, state, UTC timestamps, submission attempts, retry count, last error,
retry deadline, and result size/identity. No credentials are stored.
`request_history` retains imported and duplicate IDs without cancelling them.
OS file locks cover the whole task, including validation and publication;
they release automatically after process death. Different tasks can run
concurrently. A target/parameter conflict fails closed.

Normal states: `PENDING -> SUBMITTING -> SUBMITTED -> QUEUED -> RUNNING ->
COMPLETED -> DOWNLOADING -> DOWNLOADED` (cached results can skip queue/run).
Errors: `RETRYABLE_ERROR`, `PERMANENT_ERROR`, `SUBMISSION_UNKNOWN`.

Before POST, `SUBMITTING` and the attempt counter are committed. Client POST
retries are disabled (`retry_max=1`, verified against installed multiurl).
The successful execution HTTP response is intercepted to persist the ID
before SDK callbacks/monitoring can fail; the returned Remote ID is also
recorded. Request cleanup is disabled. There are no DELETE/cancel calls.

If the server creates a request but its response never arrives, the client
cannot know the ID. On restart, it enumerates all job pages and compares
dataset plus full request parameters. One match is adopted; multiple
matches require review. An empty listing after an uncertain submission is
**not proof that submission failed**: retain `SUBMISSION_UNKNOWN` and stop,
even over repeated restarts. This sacrifices automatic progress in an
ambiguous case to prevent another queue entry. Do not erase the journal
or reset the attempt counter to work around it.

When an ID exists, reconnect with `get_remote(ID)`, then query the original
status/results. Even a 404 does not trigger automatic replacement. Failed,
rejected, dismissed, deleted, unknown states and account/parameter errors
stop for review; the ID remains. Repairing an account or intentionally
replacing a terminal job requires reviewing its journal row first.

## Network and result transfer

Timeout/reset/aborted connection, SSL EOF, proxy/DNS errors, HTTP 408/429/5xx
use exponential backoff (60 seconds initially, capped at 900 before up to
20% jitter), at most six retries per operation per invocation. Retry
deadlines and counts survive restarts. Queue/running checks wait about
60–72 seconds without resubmitting. 400/422 parameter errors and 401/403
authentication/licence errors do not loop. Logs include task/file, ID,
status and delays, explicitly saying an existing ID will not be resubmitted.

`Results.download()` in backend 0.5.3 deletes an existing target. Instead,
obtain result location/size through `Remote.get_results()` and stream that
asset. Each retry refreshes metadata using the **same request ID**. A
matching partial asset uses Range/If-Range; validate Content-Range before
appending. A 200 response to Range restarts that asset transfer. A changed
asset starts from byte zero. Signed URL query changes do not discard the
same asset. Verify byte size, then run the existing scientific file checks.
Final publication is atomic and refuses to overwrite an existing file.

## Migrate the requests already queued before this change

Stop the old local downloader first (not the server jobs). Preserve its
logs outside temporary storage, then run from this project with `.venv`:

```powershell
python -m tools.cds_jobs --config configs/jja_2000_2024_bjt.yaml --import-log <parallel-log> --import-log <sequential-log>
python -m tools.cds_jobs --config configs/jja_2000_2024_bjt.yaml --verify
```

Import is offline. It persists every historical ID and prevents new POSTs
for those tasks even if verification loses its connection. Verification
only reads CDS jobs, checks exact dataset/parameters, and adopts an existing
successful result preferentially, otherwise the earliest live request.
All duplicate IDs remain recorded; nothing is cancelled. An inaccessible
ID is reported as unresolved. Re-run verification later if necessary.

## Acceptance before the full queue

```powershell
python -m pytest
python -m tools.cds_smoke --config configs/jja_2000_2024_bjt.yaml
```

The smoke command submits at most **one hour** of the original variables
over the original region (first configured month/day/hour), saves under
`data/cds_state/smoke/`, and never starts the monthly queue. Its own ID and
partial download survive restart. Run it again to verify completed-file
idempotency. Do not restart the formal download until smoke and recovery
are confirmed. To download without starting data processing:

```powershell
python -c "from src.config import load_config; from src.download_era5 import run; run(load_config('configs/jja_2000_2024_bjt.yaml'), workers=3)"
```

Tests `tests/test_cds_download.py` cover A–H, real child-process exit, process
locking, HTTP Range/no-Range, malformed ranges, remote terminal failures,
5xx backoff, duplicate jobs and the installed SDK over synthetic HTTP.

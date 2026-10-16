"""Durable CDS jobs, independent of the scientific processing pipeline.

Uses the asynchronous interface already behind cdsapi 0.7.7. Never retry a
submission with an uncertain outcome; reconcile it against the account first.
"""

import hashlib
import json
import os
from pathlib import Path
import random
import re
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from urllib.parse import urlsplit

import requests


class SubmissionUncertain(RuntimeError):
    """A POST may have succeeded. No further POST is safe without evidence."""


class PermanentError(RuntimeError):
    """An account, parameter, remote-job or local-state problem needs attention."""


class ResultAccessError(RuntimeError):
    """
    The result download was refused (HTTP 401/403) after the request itself
    was found completed with the same credentials: typically an expired
    signed result URL. Retried by refreshing the result metadata of the SAME
    request ID; never a reason to submit a new request.
    """


def redacted(exc):
    """
    A copy of `exc` (same type when possible) whose message has no signed URL
    parts, without the original exception chain: final tracebacks printed by
    the CLIs must not expose result signatures (review DLR-04).
    """
    text = redact(str(exc))
    try:
        copy = type(exc)(text)
    except Exception:
        copy = RuntimeError(f"{type(exc).__name__}: {text}")
    if str(copy) != text:
        copy = RuntimeError(f"{type(exc).__name__}: {text}")
    return copy


def now():
    return datetime.now(timezone.utc).isoformat()


# Signed result URLs carry credentials in their query string (signature,
# token, expiry). Keep scheme, host and path for diagnosis; drop the rest.
_URL_QUERY = re.compile(r"(https?://[^\s?#'\"<>]+)[?#][^\s'\"<>]*")
_SECRET_PARAM = re.compile(r"(?i)\b([\w-]*(?:signature|token|credential|key)[\w-]*)=[^&\s'\"<>]+")


def redact(text):
    """Remove URL query strings and signature/token/key parameters from a message."""
    if text is None:
        return None
    text = _URL_QUERY.sub(r"\1?[redacted]", str(text))
    return _SECRET_PARAM.sub(r"\1=[redacted]", text)


def canonical_request(dataset, request):
    # CDS represents selections as either scalars or lists. Selection order is
    # irrelevant except for area (N/W/S/E) and grid (latitude/longitude spacing).
    normalized = {}
    for key, value in request.items():
        values = value if isinstance(value, (list, tuple)) else [value]
        if key in ("area", "grid"):
            normalized[key] = [float(v) for v in values]
        else:
            normalized[key] = sorted(str(v) for v in values)
    return json.dumps([dataset, normalized], sort_keys=True, separators=(",", ":"))


def task_id(dataset, request):
    return hashlib.sha256(canonical_request(dataset, request).encode()).hexdigest()


class StateStore:
    """SQLite transactions with FULL sync; no credentials or result URLs stored."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY, target TEXT UNIQUE NOT NULL,
                dataset TEXT NOT NULL, parameters TEXT NOT NULL,
                request_id TEXT, status TEXT NOT NULL, created_at TEXT NOT NULL,
                submitted_at TEXT, last_checked_at TEXT, updated_at TEXT NOT NULL,
                retry_count INTEGER NOT NULL DEFAULT 0,
                submission_attempts INTEGER NOT NULL DEFAULT 0,
                error TEXT, error_kind TEXT, next_retry_at REAL,
                asset_key TEXT, asset_size INTEGER, etag TEXT)""")
            db.execute("""CREATE TABLE IF NOT EXISTS request_history (
                task_id TEXT NOT NULL, request_id TEXT NOT NULL, source TEXT NOT NULL,
                recorded_at TEXT NOT NULL, status TEXT, checked_at TEXT,
                PRIMARY KEY (task_id, request_id))""")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA synchronous=FULL")
        try:
            with db:
                yield db
        finally:
            db.close()

    def get(self, ident):
        with self.connect() as db:
            row = db.execute("SELECT * FROM tasks WHERE task_id=?", (ident,)).fetchone()
        return dict(row) if row else None

    def ensure(self, dataset, request, target):
        ident = task_id(dataset, request)
        target = str(Path(target).resolve())
        with self.connect() as db:
            row = db.execute("SELECT * FROM tasks WHERE task_id=? OR target=?", (ident, target)).fetchone()
            if row:
                if row["task_id"] != ident or row["target"] != target:
                    raise PermanentError("Task/target conflict: refusing to reuse or overwrite another task")
            else:
                db.execute("""INSERT INTO tasks
                    (task_id,target,dataset,parameters,status,created_at,updated_at)
                    VALUES (?,?,?,?,?,?,?)""",
                           (ident, target, dataset, json.dumps(request, sort_keys=True), "PENDING", now(), now()))
        return ident

    def update(self, ident, **fields):
        if fields.get("error") is not None:
            fields["error"] = redact(fields["error"])  # never persist signed result URLs
        fields["updated_at"] = now()
        with self.connect() as db:
            db.execute(f"UPDATE tasks SET {','.join(k + '=?' for k in fields)} WHERE task_id=?",
                       (*fields.values(), ident))

    def remember(self, ident, request_id, source):
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO request_history (task_id,request_id,source,recorded_at) VALUES (?,?,?,?)",
                       (ident, request_id, source, now()))

    @contextmanager
    def lock(self, ident):
        # OS locks survive neither process exit nor reboot. The lock file itself
        # remains, so unlinking/recreating it cannot split ownership. One owner
        # covers submission, polling, transfer, validation and final publication.
        path = self.path.parent / (ident + ".lock")
        with path.open("a+b") as stream:
            stream.seek(0, 2)
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError(f"TASK {ident}: another downloader owns this task; no submission") from exc
            try:
                yield
            finally:
                stream.seek(0)
                if os.name == "nt":
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream, fcntl.LOCK_UN)


def error_kind(exc):
    if isinstance(exc, ResultAccessError):
        return "RESULT_ACCESS"
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        code = exc.response.status_code
        if code in (401, 403):
            return "AUTHENTICATION"
        if code in (408, 429) or code >= 500:
            return "SERVER"
        if code in (400, 422):
            return "PARAMETERS"
        return "HTTP_PERMANENT"
    if isinstance(exc, (requests.ConnectionError, requests.Timeout,
                        requests.exceptions.ChunkedEncodingError, ConnectionError, TimeoutError)):
        return "NETWORK"
    if isinstance(exc, PermissionError):
        return "AUTHENTICATION"
    return "LOCAL_OR_PERMANENT"


class CaptureSession(requests.Session):
    """Persist the successful execution response before library callbacks/GETs."""

    def __init__(self, capture):
        super().__init__()
        self.capture = capture

    def request(self, method, url, *args, **kwargs):
        response = super().request(method, url, *args, **kwargs)
        if method.lower() == "post" and urlsplit(url).path.endswith("/execution") and response.ok:
            payload = response.json()
            ident = payload.get("jobID")
            if not ident:
                links = [link["href"] for link in payload.get("links", []) if link.get("rel") == "monitor"]
                if len(links) == 1:
                    ident = urlsplit(links[0]).path.rstrip("/").rsplit("/", 1)[-1]
            if ident:
                self.capture(str(ident))
        return response


class CDSClient:
    """Adapter for the installed cdsapi LegacyClient; never changes networking."""

    def __init__(self, capture, options=None):
        import cdsapi
        self.session = CaptureSession(capture)
        self.legacy = cdsapi.Client(**{
            **(options or {}), "wait_until_complete": False, "delete": False,
            "retry_max": 1, "timeout": 60, "session": self.session,
        })
        if not hasattr(self.legacy, "client"):
            raise PermanentError("This downloader requires the current CDS datastores backend of cdsapi")
        self.client = self.legacy.client

    def submit(self, dataset, request):
        return self.legacy.retrieve(dataset, request)

    def remote(self, ident):
        return self.client.get_remote(ident)

    def matching_jobs(self, dataset, request):
        expected = canonical_request(dataset, request)
        matches = []
        page = self.client.get_jobs(sortby="-created")
        while page is not None:
            for ident in page.request_ids:
                # Read the full submitted parameters, not just month/year. An
                # inaccessible job makes reconciliation inconclusive: fail closed.
                data = self.remote(ident).json
                if data.get("processID") != dataset:
                    continue
                parameters = data.get("metadata", {}).get("request", {}).get("ids")
                if parameters is None:
                    raise SubmissionUncertain(f"Cannot inspect parameters of CDS request {ident}")
                if canonical_request(dataset, parameters) == expected:
                    matches.append(ident)
            page = page.next
        return matches

    def download(self, remote, partial, store, ident):
        # Results.download() in 0.5.3 unlinks the target. Obtain the official
        # result metadata, then stream its asset using Range/If-Range instead.
        results = remote.get_results()
        download_asset(results.location, results.content_length, partial, store, ident,
                       verify=self.legacy.verify, timeout=self.legacy.timeout)

    def failure_detail(self, remote):
        """Use the SDK's result error to retain the CDS rejection explanation."""
        try:
            remote.get_results()
        except Exception as exc:
            return str(exc)
        return "No additional failure detail returned by CDS"


def download_asset(url, size, partial, store, ident, *, verify=True, timeout=60, session=None):
    parts = urlsplit(url)
    key = hashlib.sha256(f"{parts.netloc}{parts.path}:{size}".encode()).hexdigest()
    state = store.get(ident)
    if state["asset_key"] != key:
        # A different asset (or none recorded yet): any bytes already in
        # `partial` belong to another asset. Discard them BEFORE recording the
        # new key, so that whatever fails afterwards (no response, a crash, a
        # restart), the journal never pairs this key with another asset's bytes.
        partial.unlink(missing_ok=True)
        store.update(ident, asset_key=key, asset_size=size, etag=None)
        state = store.get(ident)
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        partial.unlink()
        offset = 0
    if offset == size and partial.exists():
        return
    headers = {"Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
        if state["etag"]:
            headers["If-Range"] = state["etag"]
    # No API token is forwarded to the result asset host.
    owner = session is None
    session = session or requests.Session()
    try:
        with session.get(url, headers=headers, stream=True, timeout=timeout, verify=verify) as response:
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                code = exc.response.status_code if exc.response is not None else None
                if code in (401, 403):
                    # Only the result asset host lands here: the request was
                    # seen completed with valid API credentials, so this is an
                    # (expired) signed URL. Retried with fresh result metadata
                    # of the same request ID. API 401/403 stay permanent.
                    raise ResultAccessError(f"result download refused (HTTP {code}); "
                                            "refreshing the result URL of the same request") from None
                raise
            if response.status_code == 206:
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                if not match or int(match[1]) != offset or int(match[3]) != size or int(match[2]) != size - 1:
                    raise PermanentError("Invalid Content-Range; partial result left untouched")
                mode = "ab" if offset else "wb"
            elif response.status_code == 200:
                mode = "wb"  # Server ignored Range: restart this asset, same CDS job.
            else:
                raise PermanentError(f"Unexpected asset HTTP status {response.status_code}")
            store.update(ident, etag=response.headers.get("ETag"))
            with partial.open(mode) as stream:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
        if partial.stat().st_size != size:
            raise requests.exceptions.ChunkedEncodingError(f"Incomplete asset: {partial.stat().st_size}/{size} bytes")
    finally:
        if owner:
            session.close()


class Downloader:
    def __init__(self, store, *, client_factory=CDSClient, sleep=time.sleep,
                 log=print, retries=6, retry_seconds=60, poll_seconds=60):
        self.store, self.client_factory = store, client_factory
        self.sleep, self.log = sleep, log
        self.retries, self.retry_seconds, self.poll_seconds = retries, retry_seconds, poll_seconds

    def message(self, ident, text):
        state = self.store.get(ident)
        self.log(redact(f"TASK {ident[:12]} {Path(state['target']).name} request_id={state['request_id']} "
                        f"status={state['status']} {text}"))

    def retry(self, ident, action, operation):
        for attempt in range(self.retries + 1):
            state = self.store.get(ident)
            if state["next_retry_at"]:
                self.sleep(max(0, state["next_retry_at"] - time.time()))
            self.store.update(ident, last_checked_at=now())
            try:
                result = operation()
            except Exception as exc:
                kind = error_kind(exc)
                transient = kind in ("NETWORK", "SERVER", "RESULT_ACCESS")
                delay = min(900, self.retry_seconds * 2 ** attempt) * random.uniform(1, 1.2)
                self.store.update(ident, status="RETRYABLE_ERROR" if transient else "PERMANENT_ERROR",
                                  retry_count=state["retry_count"] + 1, error=str(exc), error_kind=kind,
                                  next_retry_at=time.time() + delay if transient else None)
                self.message(ident, f"{kind} error while {action}: {exc}. "
                             + ("已有 request ID，因此不会重新提交。 " if state["request_id"] else "No submission retried. ")
                             + (f"Retrying in {delay:.0f} seconds" if transient and attempt < self.retries
                                else "Stopped; state retained for recovery."))
                if not transient or attempt == self.retries:
                    raise redacted(exc) from None
            else:
                self.store.update(ident, next_retry_at=None, error=None, error_kind=None)
                return result

    def retrieve(self, dataset, request, target, options=None, validator=None):
        target = Path(target)
        ident = self.store.ensure(dataset, request, target)
        with self.store.lock(ident):
            if target.exists():
                if validator:
                    validator(target)
                self.store.update(ident, status="DOWNLOADED")
                self.message(ident, "existing file preserved; no submission")
                return target
            state = self.store.get(ident)
            if state["status"] == "PERMANENT_ERROR":
                raise PermanentError(state["error"] or "Task requires review")

            def capture(request_id):
                existing = self.store.get(ident)
                if existing["request_id"] and existing["request_id"] != request_id:
                    raise PermanentError("Conflicting request IDs; no further submission")
                self.store.update(ident, request_id=request_id, status="SUBMITTED",
                                  submitted_at=existing["submitted_at"] or now(), error=None, error_kind=None)
                self.store.remember(ident, request_id, "submission or account reconciliation")

            client = self.retry(ident, "initializing client", lambda: self.client_factory(capture, options))
            state = self.store.get(ident)
            if not state["request_id"]:
                matches = self.retry(ident, "reconciling existing CDS jobs",
                                     lambda: client.matching_jobs(dataset, request))
                if len(matches) == 1:
                    capture(matches[0])
                    self.message(ident, "recovered existing CDS request; no POST")
                elif len(matches) > 1:
                    self.store.update(ident, status="SUBMISSION_UNKNOWN", error="Multiple matching CDS requests; review required")
                    raise SubmissionUncertain("Multiple matching requests; preserved all; no POST")
                elif state["submission_attempts"] or state["status"] == "SUBMISSION_UNKNOWN":
                    self.store.update(ident, status="SUBMISSION_UNKNOWN",
                                      error="Prior submission outcome unknown; no matching request visible yet")
                    raise SubmissionUncertain("Submission outcome unknown; no second POST. Reconcile again later.")
                else:
                    # Commit intent BEFORE touching the network, including crashes
                    # between a server's creation and receipt of its response.
                    self.store.update(ident, status="SUBMITTING", submission_attempts=1, submitted_at=now())
                    try:
                        remote = client.submit(dataset, request)
                        capture(remote.request_id)
                    except BaseException as exc:
                        current = self.store.get(ident)
                        kind = error_kind(exc)
                        if not current["request_id"]:
                            permanent = kind in ("AUTHENTICATION", "PARAMETERS")
                            self.store.update(ident, status="PERMANENT_ERROR" if permanent else "SUBMISSION_UNKNOWN",
                                              error=str(exc), error_kind=kind)
                        self.message(ident, "Submission interrupted; intent/ID retained. No second POST.")
                        raise
            request_id = self.store.get(ident)["request_id"]
            self.message(ident, "resuming original request; no resubmission")
            while True:
                def check():
                    remote = client.remote(request_id)
                    return remote, remote.status
                remote, status = self.retry(ident, f"checking request {request_id}", check)
                status = status.lower()
                mapped = {"accepted": "QUEUED", "queued": "QUEUED", "running": "RUNNING",
                          "successful": "COMPLETED", "completed": "COMPLETED"}.get(status)
                if mapped is None:
                    message = f"CDS request {request_id} status={status}; no automatic replacement request"
                    if status in ("failed", "rejected") and hasattr(client, "failure_detail"):
                        message += ": " + client.failure_detail(remote)
                    self.store.update(ident, status="PERMANENT_ERROR", error=message, error_kind="CDS_REQUEST")
                    self.message(ident, message)
                    raise PermanentError(message)
                self.store.update(ident, status=mapped)
                self.message(ident, "waiting..." if mapped != "COMPLETED" else "fetching original result")
                if mapped == "COMPLETED":
                    break
                self.sleep(self.poll_seconds * random.uniform(1, 1.2))
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".part")

            def transfer():
                self.store.update(ident, status="DOWNLOADING")
                # Refresh the result URL on every attempt, always by the SAME ID.
                client.download(client.remote(request_id), partial, self.store, ident)
            self.retry(ident, f"downloading result of {request_id}", transfer)
            try:
                if validator:
                    validator(partial)
                # Atomic, no-clobber publication on Windows and Unix. A late
                # writer's completed file must never be overwritten.
                if os.name == "nt":
                    partial.rename(target)  # Windows rename fails if target exists.
                else:
                    os.link(partial, target)
                    partial.unlink()
            except Exception as exc:
                self.store.update(ident, status="RETRYABLE_ERROR", error=str(exc),
                                  error_kind="LOCAL_VALIDATION", asset_key=None)
                raise
            self.store.update(ident, status="DOWNLOADED", error=None, error_kind=None)
            self.message(ident, "saved and validated")
            return target

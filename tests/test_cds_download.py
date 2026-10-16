"""CDS lifecycle acceptance tests: synthetic jobs and the real SDK over fake HTTP."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from src.cds_download import (CDSClient, CaptureSession, Downloader, PermanentError,
                              StateStore, SubmissionUncertain, download_asset, task_id)


DATASET = "reanalysis-era5-single-levels"
REQUEST = {"year": ["2000"], "month": ["06"], "variable": ["2m_temperature"],
           "area": [36, 112, 25, 123]}


class FakeServer:
    def __init__(self):
        self.posts = 0
        self.checks = []
        self.jobs = {}
        self.statuses = ["successful"]
        self.lost_submission = False
        self.hidden = False
        self.download_errors = 0
        self.downloads = []
        self.submission_error = None

    def factory(self, capture, options=None):
        server = self

        class Remote:
            def __init__(self, ident):
                self.request_id = ident

            @property
            def status(self):
                server.checks.append(self.request_id)
                value = server.statuses.pop(0) if len(server.statuses) > 1 else server.statuses[0]
                if isinstance(value, BaseException):
                    raise value
                return value

        class Client:
            def matching_jobs(self, dataset, request):
                return [] if server.hidden else list(server.jobs)

            def submit(self, dataset, request):
                server.posts += 1
                if server.submission_error:
                    raise server.submission_error
                ident = "job-001"
                server.jobs[ident] = request
                if server.lost_submission:
                    raise requests.ConnectionError("server created job; response lost")
                capture(ident)
                return Remote(ident)

            def remote(self, ident):
                assert ident in server.jobs
                return Remote(ident)

            def download(self, remote, partial, store, ident):
                server.downloads.append(remote.request_id)
                if server.download_errors:
                    server.download_errors -= 1
                    partial.write_bytes(b"partial")
                    raise requests.exceptions.ChunkedEncodingError("transfer disconnected")
                partial.write_bytes(b"complete")

        return Client()


@pytest.fixture
def setup(tmp_path):
    store = StateStore(tmp_path / "state" / "requests.sqlite3")
    server = FakeServer()
    target = tmp_path / "raw" / "result.nc"
    sleeps, logs = [], []

    def downloader(**kwargs):
        return Downloader(store, client_factory=server.factory, sleep=sleeps.append,
                          log=logs.append, retry_seconds=0.01, poll_seconds=0.01, **kwargs)

    return SimpleNamespace(store=store, server=server, target=target, downloader=downloader,
                           sleeps=sleeps, logs=logs)


def test_a_submit_complete_download(setup):
    s = setup
    s.downloader().retrieve(DATASET, REQUEST, s.target)
    state = s.store.get(task_id(DATASET, REQUEST))
    assert s.server.posts == 1 and s.target.read_bytes() == b"complete"
    assert state["status"] == "DOWNLOADED" and state["request_id"] == "job-001"
    assert all(state[k] for k in ("created_at", "submitted_at", "last_checked_at", "parameters", "target"))


def test_b_network_during_query_never_reposts(setup):
    s = setup
    s.server.statuses = [requests.exceptions.ProxyError("proxy reset"), "accepted", "running", "successful"]
    s.downloader().retrieve(DATASET, REQUEST, s.target)
    assert s.server.posts == 1 and len(s.server.checks) == 4
    assert s.sleeps and any("已有 request ID，因此不会重新提交" in line for line in s.logs)
    assert s.store.get(task_id(DATASET, REQUEST))["retry_count"] == 1


@pytest.mark.parametrize("status", ["QUEUED", "COMPLETED", "DOWNLOADING", "SUBMITTING"])
def test_c_d_restart_uses_persisted_id(setup, status):
    s = setup
    ident = s.store.ensure(DATASET, REQUEST, s.target)
    s.store.update(ident, request_id="old-job", status=status)
    s.server.jobs["old-job"] = REQUEST
    # New store/manager objects, nothing held from the original process.
    reopened = StateStore(s.store.path)
    Downloader(reopened, client_factory=s.server.factory).retrieve(DATASET, REQUEST, s.target)
    assert s.server.posts == 0 and s.server.checks == ["old-job"]
    assert s.server.downloads == ["old-job"]


def test_c_real_process_exit_releases_lock_and_preserves_queued_id(setup):
    s = setup
    script = """
import os, sys
from pathlib import Path
from src.cds_download import StateStore
store = StateStore(sys.argv[1])
ident = store.ensure(sys.argv[2], __import__('json').loads(sys.argv[3]), Path(sys.argv[4]))
with store.lock(ident):
    store.update(ident, request_id='crashed-job', status='QUEUED')
    os._exit(19)
"""
    result = subprocess.run([sys.executable, "-c", script, str(s.store.path), DATASET,
                             json.dumps(REQUEST), str(s.target)], capture_output=True)
    assert result.returncode == 19, result.stderr.decode(errors="replace")
    s.server.jobs["crashed-job"] = REQUEST
    s.downloader().retrieve(DATASET, REQUEST, s.target)
    assert s.server.posts == 0 and s.server.downloads == ["crashed-job"]


def test_e_download_disconnect_uses_same_job(setup):
    s = setup
    s.server.download_errors = 1
    s.downloader().retrieve(DATASET, REQUEST, s.target)
    assert s.server.posts == 1 and s.server.downloads == ["job-001", "job-001"]


def test_f_lost_submission_response_reconciles_without_post(setup):
    s = setup
    s.server.lost_submission = True
    with pytest.raises(requests.ConnectionError):
        s.downloader().retrieve(DATASET, REQUEST, s.target)
    state = s.store.get(task_id(DATASET, REQUEST))
    assert state["status"] == "SUBMISSION_UNKNOWN" and state["submission_attempts"] == 1
    assert state["request_id"] is None
    s.downloader().retrieve(DATASET, REQUEST, s.target)
    assert s.server.posts == 1 and s.target.exists()


def test_f_eventually_consistent_listing_does_not_authorize_second_post(setup):
    s = setup
    s.server.lost_submission = s.server.hidden = True
    with pytest.raises(requests.ConnectionError):
        s.downloader().retrieve(DATASET, REQUEST, s.target)
    for _ in range(3):
        with pytest.raises(SubmissionUncertain, match="no second POST"):
            s.downloader().retrieve(DATASET, REQUEST, s.target)
    assert s.server.posts == 1


@pytest.mark.parametrize("code", [400, 401, 403, 422])
def test_g_parameters_and_auth_fail_once(setup, code):
    s = setup
    response = requests.Response()
    response.status_code = code
    s.server.submission_error = requests.HTTPError(f"HTTP {code}", response=response)
    with pytest.raises(requests.HTTPError):
        s.downloader().retrieve(DATASET, REQUEST, s.target)
    with pytest.raises(PermanentError):
        s.downloader().retrieve(DATASET, REQUEST, s.target)
    assert s.server.posts == 1 and s.sleeps == []


def test_h_completed_file_is_never_overwritten(setup):
    s = setup
    s.target.parent.mkdir()
    s.target.write_bytes(b"precious existing file")
    s.downloader().retrieve(DATASET, REQUEST, s.target)
    assert s.target.read_bytes() == b"precious existing file"
    assert s.server.posts == 0 and s.server.downloads == []


def test_task_hash_matches_cds_scalar_and_list_encodings():
    assert task_id(DATASET, REQUEST) == task_id(DATASET, {**REQUEST, "year": "2000"})
    assert task_id(DATASET, REQUEST) != task_id(DATASET, {**REQUEST, "area": [25, 112, 36, 123]})


def test_same_target_different_request_is_blocked(setup):
    s = setup
    s.downloader().retrieve(DATASET, REQUEST, s.target)
    with pytest.raises(PermanentError, match="conflict"):
        s.downloader().retrieve(DATASET, {**REQUEST, "year": "2001"}, s.target)
    assert s.server.posts == 1


def test_cross_process_lock_prevents_second_submission(setup):
    s = setup
    ident = s.store.ensure(DATASET, REQUEST, s.target)
    script = """
import sys
from src.cds_download import StateStore
with StateStore(sys.argv[1]).lock(sys.argv[2]):
    print('locked', flush=True)
    sys.stdin.readline()
"""
    process = subprocess.Popen([sys.executable, "-c", script, str(s.store.path), ident],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == "locked"
        with pytest.raises(RuntimeError, match="another downloader"):
            s.downloader().retrieve(DATASET, REQUEST, s.target)
        assert s.server.posts == 0
    finally:
        process.communicate("exit\n", timeout=10)


@pytest.mark.parametrize("status", ["failed", "rejected", "dismissed", "deleted", "unexpected"])
def test_explicit_remote_failure_never_reposts(setup, status):
    s = setup
    s.server.statuses = [status]
    with pytest.raises(PermanentError, match=f"status={status}"):
        s.downloader().retrieve(DATASET, REQUEST, s.target)
    assert s.server.posts == 1


@pytest.mark.parametrize("code", [404, 500, 503])
def test_failed_get_never_reposts(setup, code):
    s = setup
    response = requests.Response()
    response.status_code = code
    s.server.statuses = [requests.HTTPError(str(code), response=response)]
    with pytest.raises(requests.HTTPError):
        s.downloader(retries=2).retrieve(DATASET, REQUEST, s.target)
    assert s.server.posts == 1 and s.store.get(task_id(DATASET, REQUEST))["request_id"] == "job-001"
    assert len(s.server.checks) == (3 if code >= 500 else 1)


def test_duplicates_are_preserved_and_block_new_submission(setup):
    s = setup
    s.server.jobs = {"original": REQUEST, "old-duplicate": REQUEST}
    with pytest.raises(SubmissionUncertain, match="Multiple"):
        s.downloader().retrieve(DATASET, REQUEST, s.target)
    assert s.server.posts == 0 and len(s.server.jobs) == 2


class AssetResponse:
    def __init__(self, code, headers, chunks):
        self.status_code, self.headers, self.chunks = code, headers, chunks

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, size):
        for item in self.chunks:
            if isinstance(item, Exception):
                raise item
            yield item


@pytest.mark.parametrize("ranges", [True, False])
def test_http_asset_resume_or_restart_without_new_cds_request(setup, ranges):
    s = setup
    ident = s.store.ensure(DATASET, REQUEST, s.target)
    partial = s.target.with_suffix(".nc.part")
    partial.parent.mkdir()
    responses = [AssetResponse(200, {"ETag": '"asset-1"'},
                               [b"abc", requests.exceptions.ChunkedEncodingError("reset")]),
                 AssetResponse(206 if ranges else 200,
                               {"Content-Range": "bytes 3-5/6"} if ranges else {},
                               [b"def" if ranges else b"abcdef"])]
    headers = []

    def get(url, **kwargs):
        headers.append(kwargs["headers"])
        return responses.pop(0)
    session = SimpleNamespace(get=get)
    with pytest.raises(requests.exceptions.ChunkedEncodingError):
        download_asset("https://asset.invalid/result?token=old", 6, partial, s.store, ident, session=session)
    assert partial.read_bytes() == b"abc"
    # Signed URL may change while the immutable asset remains the same.
    download_asset("https://asset.invalid/result?token=new", 6, partial,
                   StateStore(s.store.path), ident, session=session)
    assert headers[1]["Range"] == "bytes=3-" and headers[1]["If-Range"] == '"asset-1"'
    assert partial.read_bytes() == b"abcdef" and s.server.posts == 0


def test_invalid_content_range_preserves_partial(setup):
    s = setup
    ident = s.store.ensure(DATASET, REQUEST, s.target)
    partial = s.target.with_suffix(".part")
    partial.parent.mkdir()
    response = AssetResponse(206, {"Content-Range": "bytes 9-11/12"}, [b"bad"])
    with pytest.raises(PermanentError, match="Content-Range"):
        download_asset("https://asset.invalid/result", 12, partial, s.store, ident,
                       session=SimpleNamespace(get=lambda *a, **k: response))
    assert not partial.exists()


def test_sdk_post_response_is_durable_before_remote_construction(setup, monkeypatch):
    """Run installed cdsapi/datastores code over HTTP fakes, including lost response."""
    s = setup
    ident = s.store.ensure(DATASET, REQUEST, s.target)
    posts, jobs = [], {}
    base = "https://cds.invalid/api/retrieve/v1"
    lost = [False]

    def http(session, method, url, **kwargs):
        payload = {}
        if method.lower() == "post":
            assert url.endswith("/execution")
            posts.append(url)
            jobs["sdk-job"] = REQUEST
            if lost[0]:
                raise requests.ConnectionError("response lost")
            payload = {"jobID": "sdk-job", "links": [{"rel": "monitor", "href": base + "/jobs/sdk-job"}]}
        elif url.endswith("/jobs"):
            payload = {"jobs": [{"jobID": job} for job in jobs]}
        elif "/jobs/" in url:
            payload = {"processID": DATASET, "metadata": {"request": {"ids": REQUEST}},
                       "status": "successful", "links": [{"rel": "self", "href": url}]}
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(payload).encode()
        response.request = requests.Request(method, url).prepare()
        response.url = url
        return response

    monkeypatch.setattr(requests.Session, "request", http)
    def capture(value):
        s.store.update(ident, request_id=value, status="SUBMITTED")
    client = CDSClient(capture, {"url": "https://cds.invalid/api", "key": "fake-key", "quiet": True})
    remote = client.submit(DATASET, REQUEST)
    assert s.store.get(ident)["request_id"] == remote.request_id == "sdk-job"
    assert client.client.maximum_tries == 1 and client.client.cleanup is False
    assert client.matching_jobs(DATASET, REQUEST) == ["sdk-job"]
    assert client.remote("sdk-job").status == "successful"
    lost[0] = True
    with pytest.raises(requests.ConnectionError):
        client.submit(DATASET, REQUEST)
    assert len(posts) == 2  # Exactly ONE attempt per explicit submit invocation.


def test_capture_occurs_before_sdk_logging_can_fail(setup, monkeypatch):
    s = setup
    ident = s.store.ensure(DATASET, REQUEST, s.target)
    def response(*args, **kwargs):
        value = requests.Response()
        value.status_code = 201
        value._content = b'{"jobID":"persist-first"}'
        return value
    monkeypatch.setattr(requests.Session, "request", response)
    session = CaptureSession(lambda rid: s.store.update(ident, request_id=rid, status="SUBMITTED"))
    session.request("post", "https://cds.invalid/api/retrieve/v1/processes/era5/execution")
    assert StateStore(s.store.path).get(ident)["request_id"] == "persist-first"


def test_historical_sequential_and_parallel_log_import():
    from tools.cds_jobs import log_ids
    log = """Downloading [1/100] era5_east_china_2000_06.nc ...
2026-10-16 14:01:33,543 INFO Request ID is first-id
Downloading [2/100] era5_east_china_2000_07.nc ...
2026-10-16 14:07:13,756 INFO Request ID is old-id
[2/100] era5_east_china_2000_07.nc: CDS: Request ID is duplicate-id
[3/100] era5_east_china_2000_08.nc: CDS: Request ID is third-id
"""
    assert list(log_ids(log)) == [("era5_east_china_2000_06.nc", "first-id"),
                                  ("era5_east_china_2000_07.nc", "old-id"),
                                  ("era5_east_china_2000_07.nc", "duplicate-id"),
                                  ("era5_east_china_2000_08.nc", "third-id")]


def test_failed_id_write_cannot_enable_another_post(setup, monkeypatch):
    s = setup
    update = s.store.update
    failed = []
    def disk_error(ident, **fields):
        if fields.get("request_id") and not failed:
            failed.append(True)
            raise OSError("disk temporarily unavailable after server creation")
        return update(ident, **fields)
    monkeypatch.setattr(s.store, "update", disk_error)
    with pytest.raises(OSError, match="disk"):
        s.downloader().retrieve(DATASET, REQUEST, s.target)
    assert s.store.get(task_id(DATASET, REQUEST))["submission_attempts"] == 1
    s.downloader().retrieve(DATASET, REQUEST, s.target)
    assert s.server.posts == 1 and s.target.exists()


def test_persisted_deadline_and_exponential_backoff(setup):
    s = setup
    s.server.statuses = [requests.Timeout("timeout")] * 3 + ["successful"]
    s.downloader().retrieve(DATASET, REQUEST, s.target)
    assert len(s.sleeps) == 3 and 0 < s.sleeps[0] < s.sleeps[1] < s.sleeps[2]
    assert s.store.get(task_id(DATASET, REQUEST))["retry_count"] == 3


def test_h_late_existing_file_is_never_overwritten(setup):
    s = setup
    factory = s.server.factory
    def late_writer(capture, options):
        client = factory(capture, options)
        download = client.download
        def transfer(*args):
            download(*args)
            s.target.write_bytes(b"other writer completed")
        client.download = transfer
        return client
    with pytest.raises(FileExistsError):
        Downloader(s.store, client_factory=late_writer).retrieve(DATASET, REQUEST, s.target)
    assert s.target.read_bytes() == b"other writer completed" and s.server.posts == 1


def test_offline_import_blocks_submission_until_reconciled(setup, tmp_path):
    from tools.cds_jobs import import_log
    from src.config import PROJECT_ROOT, load_config
    from src.download_era5 import build_raw_file_request
    s = setup
    cfg = load_config(PROJECT_ROOT / "configs/jja_2000_2024_bjt.yaml", project_root=tmp_path)
    raw = cfg.raw_files[1]
    log = tmp_path / "old.log"
    log.write_text(f"[2/100] {raw.path.name}: CDS: Request ID is preserved-id\n", encoding="utf-8")
    assert import_log(cfg, s.store, log) == 1
    request = build_raw_file_request(cfg, raw)
    state = StateStore(s.store.path).get(task_id(DATASET, request))
    assert state["status"] == "SUBMISSION_UNKNOWN" and state["submission_attempts"] == 1
    with pytest.raises(SubmissionUncertain):
        s.downloader().retrieve(DATASET, request, raw.path)
    assert s.server.posts == 0


def test_reconciliation_checks_later_job_pages():
    adapter = CDSClient.__new__(CDSClient)
    last = SimpleNamespace(request_ids=["older-matching-job"], next=None)
    first = SimpleNamespace(request_ids=[], next=last)
    adapter.client = SimpleNamespace(get_jobs=lambda **kwargs: first)
    adapter.remote = lambda ident: SimpleNamespace(json={
        "processID": DATASET, "metadata": {"request": {"ids": REQUEST}}})
    assert adapter.matching_jobs(DATASET, REQUEST) == ["older-matching-job"]


def test_rejection_detail_is_persisted_without_reposting(setup):
    s = setup
    s.server.statuses = ["rejected"]
    factory = s.server.factory
    def with_detail(capture, options):
        client = factory(capture, options)
        client.failure_detail = lambda remote: "Number queued requests temporarily limited"
        return client
    with pytest.raises(PermanentError, match="queued requests temporarily limited"):
        Downloader(s.store, client_factory=with_detail).retrieve(DATASET, REQUEST, s.target)
    assert "queued requests temporarily limited" in s.store.get(task_id(DATASET, REQUEST))["error"]
    assert s.server.posts == 1


# --- Review fixes (Codex review of ca92faa: DLR-01, DLR-03, DLR-04) ---------

def _asset_key(url, size):
    from urllib.parse import urlsplit
    parts = urlsplit(url)
    import hashlib
    return hashlib.sha256(f"{parts.netloc}{parts.path}:{size}".encode()).hexdigest()


def test_dlr01_changed_asset_after_failed_get_never_reuses_old_bytes(setup):
    """A new asset must not inherit another asset's bytes, whatever fails in between."""
    s = setup
    ident = s.store.ensure(DATASET, REQUEST, s.target)
    partial = s.target.with_suffix(".nc.part")
    partial.parent.mkdir()
    partial.write_bytes(b"OLD")  # prefix of asset A
    s.store.update(ident, asset_key=_asset_key("https://asset.invalid/old-result", 6),
                   asset_size=6, etag='"asset-A"')

    def disconnected(url, **kwargs):
        raise requests.ConnectionError("no response")
    with pytest.raises(requests.ConnectionError):
        download_asset("https://asset.invalid/new-result", 6, partial, s.store, ident,
                       session=SimpleNamespace(get=disconnected))
    # The journal now names asset B, and no byte of asset A is left to resume from.
    assert not partial.exists()
    assert s.store.get(ident)["etag"] is None

    headers = []

    def new_asset(url, **kwargs):
        headers.append(kwargs["headers"])
        if "Range" in kwargs["headers"]:
            return AssetResponse(206, {"Content-Range": "bytes 3-5/6"}, [b"def"])
        return AssetResponse(200, {"ETag": '"asset-B"'}, [b"abcdef"])
    download_asset("https://asset.invalid/new-result", 6, partial, StateStore(s.store.path), ident,
                   session=SimpleNamespace(get=new_asset))
    assert partial.read_bytes() == b"abcdef"
    assert "Range" not in headers[0]


def test_dlr01_same_asset_still_resumes(setup):
    """The fix must not lose a genuine resume of the same asset."""
    s = setup
    ident = s.store.ensure(DATASET, REQUEST, s.target)
    partial = s.target.with_suffix(".nc.part")
    partial.parent.mkdir()
    responses = [AssetResponse(200, {"ETag": '"same"'}, [b"abc", requests.exceptions.ChunkedEncodingError("x")]),
                 AssetResponse(206, {"Content-Range": "bytes 3-5/6"}, [b"def"])]
    session = SimpleNamespace(get=lambda url, **kwargs: responses.pop(0))
    with pytest.raises(requests.exceptions.ChunkedEncodingError):
        download_asset("https://asset.invalid/r?sig=1", 6, partial, s.store, ident, session=session)
    download_asset("https://asset.invalid/r?sig=2", 6, partial, s.store, ident, session=session)
    assert partial.read_bytes() == b"abcdef"


def _http_error(code, url, body=b""):
    response = requests.Response()
    response.status_code = code
    response.url = url
    response._content = body
    return requests.HTTPError(f"{code} Client Error for url: {url}", response=response)


def test_dlr03_expired_result_url_refreshes_same_request(setup):
    s = setup
    ident = s.store.ensure(DATASET, REQUEST, s.target)
    s.store.update(ident, request_id="done-job", status="COMPLETED")
    s.server.jobs["done-job"] = REQUEST
    factory = s.server.factory
    attempts = []

    def expiring(capture, options):
        client = factory(capture, options)
        original = client.download

        def download(remote, partial, store, ident_):
            attempts.append(remote.request_id)
            if len(attempts) == 1:
                raise _http_error(403, "https://asset.invalid/result?X-Amz-Signature=old",
                                  b"<Code>ExpiredToken</Code>")
            return original(remote, partial, store, ident_)
        client.download = download
        return client
    Downloader(s.store, client_factory=expiring, sleep=s.sleeps.append, log=s.logs.append,
               retry_seconds=0.01, poll_seconds=0.01).retrieve(DATASET, REQUEST, s.target)
    assert attempts == ["done-job", "done-job"] and s.server.posts == 0
    assert s.target.read_bytes() == b"complete"
    assert s.store.get(ident)["status"] == "DOWNLOADED"


def test_dlr03_persistent_refusal_stops_after_bounded_retries_without_post(setup):
    s = setup
    ident = s.store.ensure(DATASET, REQUEST, s.target)
    s.store.update(ident, request_id="done-job", status="COMPLETED")
    s.server.jobs["done-job"] = REQUEST
    factory = s.server.factory

    def refusing(capture, options):
        client = factory(capture, options)

        def download(remote, partial, store, ident_):
            raise _http_error(403, "https://asset.invalid/result?token=t")
        client.download = download
        return client
    with pytest.raises(Exception, match="refreshing the result URL"):
        Downloader(s.store, client_factory=refusing, sleep=s.sleeps.append, log=s.logs.append,
                   retries=2, retry_seconds=0.01, poll_seconds=0.01).retrieve(DATASET, REQUEST, s.target)
    state = s.store.get(ident)
    # Bounded, retryable on the next run with the same ID; never a new request.
    assert state["status"] == "RETRYABLE_ERROR" and state["error_kind"] == "RESULT_ACCESS"
    assert state["request_id"] == "done-job" and s.server.posts == 0
    assert not s.target.exists()


def test_dlr03_api_auth_errors_are_still_permanent(setup):
    """Only result downloads are refreshed; a 403 on the request itself stays permanent."""
    s = setup
    s.server.statuses = [_http_error(403, "https://cds.invalid/api/retrieve/v1/jobs/job-001")]
    with pytest.raises(requests.HTTPError):
        s.downloader().retrieve(DATASET, REQUEST, s.target)
    assert s.store.get(task_id(DATASET, REQUEST))["status"] == "PERMANENT_ERROR"


def test_dlr04_signed_urls_are_never_persisted_or_logged(setup):
    s = setup
    ident = s.store.ensure(DATASET, REQUEST, s.target)
    signature = "SYNTHETIC_ONLY_SIGNATURE"
    error = _http_error(503, f"https://asset.invalid/result?X-Amz-Signature={signature}&X-Amz-Credential=abc")

    def operation():
        raise error
    with pytest.raises(requests.HTTPError):
        s.downloader(retries=0).retry(ident, "downloading result", operation)
    stored = s.store.get(ident)["error"]
    assert signature not in stored and "X-Amz-Credential=abc" not in stored
    assert "https://asset.invalid/result" in stored  # host and path kept for diagnosis
    assert s.logs and all(signature not in line for line in s.logs)


def test_dlr04_redact_keeps_diagnosis_and_drops_secrets():
    from src.cds_download import redact
    text = ("403 for url: https://object-store.invalid/cache/abc.nc?X-Amz-Signature=s3cr3t&X-Amz-Expires=600 "
            "and token=t0k3n, api key=k3y")
    out = redact(text)
    for secret in ("s3cr3t", "t0k3n", "k3y"):
        assert secret not in out
    assert "https://object-store.invalid/cache/abc.nc" in out and out.startswith("403 for url")
    assert redact(None) is None

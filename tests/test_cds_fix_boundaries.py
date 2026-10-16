"""Fault boundaries independent of the production fix's own tests (Codex review of 85f2603)."""
import hashlib
import traceback
from types import SimpleNamespace
import uuid

import pytest
import requests

from src.cds_download import Downloader, StateStore, download_asset

DATASET = 'synthetic-era5'
REQUEST = {'year': ['2003'], 'month': ['06']}


def state(tmp_path):
    store = StateStore(tmp_path / 'requests.sqlite3')
    target = tmp_path / 'result.nc'
    ident = store.ensure(DATASET, REQUEST, target)
    return store, target, ident


class Crash(BaseException):
    pass


@pytest.mark.parametrize('fault', ['unlink', 'journal-before-commit', 'first-get'])
def test_asset_change_crash_windows_never_associate_new_key_with_old_bytes(tmp_path, monkeypatch, fault):
    store, target, ident = state(tmp_path)
    partial = target.with_suffix('.part')
    partial.write_bytes(b'OLD')
    old_key = hashlib.sha256(b'asset.invalid/old:6').hexdigest()
    new_key = hashlib.sha256(b'asset.invalid/new:6').hexdigest()
    store.update(ident, asset_key=old_key, asset_size=6, etag='old-etag')
    unlink, update = type(partial).unlink, store.update
    with monkeypatch.context() as context:
        if fault == 'unlink':
            def fail_unlink(path, *args, **kwargs):
                if path == partial: raise Crash('at old partial removal')
                return unlink(path, *args, **kwargs)
            context.setattr(type(partial), 'unlink', fail_unlink)
        if fault == 'journal-before-commit':
            def fail_update(task, **fields):
                if fields.get('asset_key') == new_key: raise Crash('before new key commit')
                return update(task, **fields)
            context.setattr(store, 'update', fail_update)
        def fail_get(*args, **kwargs): raise Crash('before first response')
        with pytest.raises(Crash):
            download_asset('https://asset.invalid/new', 6, partial, store, ident,
                           session=SimpleNamespace(get=fail_get))
    reopened = StateStore(store.path)
    assert not (reopened.get(ident)['asset_key'] == new_key and partial.exists() and partial.read_bytes() == b'OLD')
    headers = []
    class Response:
        status_code, headers = 200, {}
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def raise_for_status(self): pass
        def iter_content(self, size): yield b'abcdef'
    def get(*args, **kwargs):
        headers.append(kwargs['headers'])
        return Response()
    download_asset('https://asset.invalid/new', 6, partial, reopened, ident,
                   session=SimpleNamespace(get=get))
    assert partial.read_bytes() == b'abcdef'
    assert 'Range' not in headers[0]


@pytest.mark.parametrize('phase', ['remote-handle', 'result-metadata'])
def test_api_403_after_completed_status_is_still_permanent(tmp_path, phase):
    """A GET to the CDS API is not a refusal from the result asset host."""
    store, target, ident = state(tmp_path)
    store.update(ident, request_id='existing-job', status='COMPLETED')
    calls = {'remote': 0, 'metadata': 0, 'post': 0}
    def api_forbidden(suffix):
        response = requests.Response()
        response.status_code = 403
        response.url = 'https://cds.invalid/api/retrieve/v1/jobs/existing-job' + suffix
        response.raise_for_status()
    class Client:
        def remote(self, request_id):
            calls['remote'] += 1
            if phase == 'remote-handle' and calls['remote'] > 1: api_forbidden('')
            return SimpleNamespace(request_id=request_id, status='successful')
        def download(self, remote, *args):
            calls['metadata'] += 1
            api_forbidden('/results')
        def submit(self, *args):
            calls['post'] += 1
            raise AssertionError('Must not POST again')
    with pytest.raises(Exception):
        Downloader(store, client_factory=lambda *args: Client(), retries=1, retry_seconds=0,
                   sleep=lambda _: None, log=lambda _: None).retrieve(DATASET, REQUEST, target)
    current = store.get(ident)
    assert current['status'] == 'PERMANENT_ERROR', (current['status'], current['error_kind'], calls)
    assert current['error_kind'] == 'AUTHENTICATION'
    assert calls['remote'] == 2 and calls['post'] == 0


def test_final_exception_traceback_contains_no_signed_url(tmp_path):
    """tools.cds_smoke leaves final exceptions uncaught, so stderr must be safe."""
    store, target, ident = state(tmp_path)
    store.update(ident, request_id='existing-job', status='COMPLETED')
    signature = 'FAKE_' + uuid.uuid4().hex
    response = requests.Response()
    response.status_code = 503
    response.url = f'https://asset.invalid/result?X-Amz-Signature={signature}'
    logs = []
    try:
        Downloader(store, retries=0, log=logs.append).retry(ident, 'downloading result', response.raise_for_status)
    except Exception as error:
        terminal = ''.join(traceback.format_exception(type(error), error, error.__traceback__))
    else:
        raise AssertionError('Synthetic failed asset must raise an error')
    assert signature not in store.get(ident)['error']
    assert all(signature not in message for message in logs)
    assert signature not in terminal, 'Final CLI traceback still exposes the synthetic result signature'

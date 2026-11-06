"""Cache explicitly selected public evidence; no discovery, retries or crawling."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


def fetch(source_id, url, directory, license_note):
    if not source_id.replace('_', '').isalnum():
        raise ValueError('Evidence IDs must be alphanumeric with underscores')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    raw = directory / f'{source_id}.raw'
    meta_path = directory / f'{source_id}.meta.json'
    if raw.exists() or meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if meta['requested_url'] != url or meta['sha256'] != hashlib.sha256(raw.read_bytes()).hexdigest():
            raise ValueError('Existing evidence differs; use a new evidence ID')
        return meta
    request = Request(url, headers={'User-Agent': 'EHHR-Public-Factory-Pilot/0.1 (bounded public research)'})
    with urlopen(request, timeout=25) as response:
        body = response.read(30 * 1024 * 1024 + 1)
        if len(body) > 30 * 1024 * 1024:
            raise ValueError('Evidence exceeds the 30 MiB retrieval limit')
        meta = dict(id=source_id, requested_url=url, final_url=response.url,
                    retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
                    http_status=response.status, content_type=response.headers.get('Content-Type'),
                    sha256=hashlib.sha256(body).hexdigest(), bytes=len(body),
                    raw_file=raw.name, license_note=license_note)
    if meta['http_status'] != 200 or not body:
        raise ValueError('No usable response')
    raw.write_bytes(body)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--id', required=True)
    parser.add_argument('--url', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--license-note', default='Publicly accessible; no open reuse license identified. Local audit copy only.')
    args = parser.parse_args()
    print(json.dumps(fetch(args.id, args.url, args.output, args.license_note), ensure_ascii=False))

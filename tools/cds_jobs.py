"""Import historical CDS log IDs offline and verify/adopt them without POST/DELETE.

python -m tools.cds_jobs --config configs/jja_2000_2024_bjt.yaml --import-log old.log
python -m tools.cds_jobs --config configs/jja_2000_2024_bjt.yaml --verify
"""

import argparse
from pathlib import Path
import re

from src.cds_download import CDSClient, StateStore, canonical_request, now
from src.config import cds_state_path, load_config
from src.download_era5 import DATASET, build_raw_file_request, check_raw_file


LOG_ID = re.compile(r"(era5_[\w.-]+\.nc): CDS: Request ID is ([\w-]+)")


def log_ids(text):
    """Read both the old sequential log and per-file parallel log prefixes."""
    current = None
    for line in text.splitlines():
        start = re.search(r"^Downloading .*?(era5_[\w.-]+\.nc) \.\.\.", line)
        if start:
            current = start[1]
        parallel = LOG_ID.search(line)
        if parallel:
            yield parallel[1], parallel[2]
        elif current:
            sequential = re.search(r"\bINFO Request ID is ([\w-]+)", line)
            if sequential:
                yield current, sequential[1]


def import_log(cfg, store, path):
    files = {raw.path.name: raw for raw in cfg.raw_files}
    count = 0
    for name, request_id in log_ids(Path(path).read_text(encoding="utf-8", errors="replace")):
        if name not in files:
            raise ValueError(f"Log task {name} is outside the selected config")
        raw = files[name]
        ident = store.ensure(DATASET, build_raw_file_request(cfg, raw), raw.path)
        with store.lock(ident):
            state = store.get(ident)
            if not state["request_id"]:
                # Even when the network is down the old server-side work blocks
                # all fresh submissions for this logical task.
                store.update(ident, status="SUBMISSION_UNKNOWN", submission_attempts=1,
                             error="Historical CDS IDs imported; verify and adopt before downloading")
            store.remember(ident, request_id, str(Path(path).resolve()))
        count += 1
        print(f"IMPORTED {name} request_id={request_id}; no network request", flush=True)
    return count


def verify(cfg, store, client):
    failures = []
    for raw in cfg.raw_files:
        request = build_raw_file_request(cfg, raw)
        ident = store.ensure(DATASET, request, raw.path)
        with store.lock(ident):
            with store.connect() as db:
                rows = db.execute("SELECT * FROM request_history WHERE task_id=?", (ident,)).fetchall()
            if not rows:
                continue
            verified = []
            for row in rows:
                request_id = row["request_id"]
                try:
                    data = client.remote(request_id).json
                    params = data.get("metadata", {}).get("request", {}).get("ids")
                    if data.get("processID") != DATASET or params is None or canonical_request(DATASET, params) != canonical_request(DATASET, request):
                        raise ValueError("Server dataset/parameters differ from this local task")
                    status = data["status"]
                    with store.connect() as db:
                        db.execute("UPDATE request_history SET status=?,checked_at=? WHERE task_id=? AND request_id=?",
                                   (status, now(), ident, request_id))
                    print(f"VERIFIED {raw.path.name} request_id={request_id} status={status}", flush=True)
                    if status in ("accepted", "running", "successful"):
                        verified.append((status != "successful", data.get("created", ""), request_id, status))
                except Exception as exc:
                    failures.append((raw.path.name, request_id, str(exc)))
                    print(f"UNRESOLVED {raw.path.name} request_id={request_id}: {exc}; no POST/DELETE", flush=True)
            if verified:
                _, created, request_id, status = min(verified)
                store.update(ident, request_id=request_id,
                             status={"accepted": "QUEUED", "running": "RUNNING", "successful": "COMPLETED"}[status],
                             submitted_at=created or store.get(ident)["created_at"], last_checked_at=now(),
                             error=None, error_kind=None)
                print(f"ADOPTED {raw.path.name} request_id={request_id}; all other IDs retained, none cancelled", flush=True)
            if raw.path.exists():
                check_raw_file(raw.path, raw, cfg)
                store.update(ident, status="DOWNLOADED")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--import-log", action="append", default=[])
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, **({"project_root": args.project_root} if args.project_root else {}))
    if not cfg.is_multi_year:
        parser.error("Select the phase-2 multi-year config")
    store = StateStore(cds_state_path(cfg.raw_files[0].path))
    for path in args.import_log:
        import_log(cfg, store, path)
    if args.verify:
        client = CDSClient(lambda _: (_ for _ in ()).throw(RuntimeError("This tool never submits")), {"quiet": True, "progress": False})
        if verify(cfg, store, client):
            raise SystemExit(1)


if __name__ == "__main__":
    main()

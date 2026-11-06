"""Read-only inventory and bounded train-period verification for the probability study.

Reuses accepted raw-file hashes instead of rerunning the full production pipeline.
Final-test meteorological values are not decoded for statistics or model selection.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.config import load_config
from src.indices import heat_index_rothfusz, relative_humidity


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def audit(config: Path, source_root: Path, previous_raw: Path, output: Path) -> dict:
    source_root, output = source_root.resolve(), output.resolve()
    if output.is_relative_to(source_root):
        raise ValueError("Audit output must be outside the read-only source project")
    output.mkdir(parents=True, exist_ok=False)
    cfg = load_config(config, project_root=source_root)
    prior = json.loads(previous_raw.read_text(encoding="utf-8"))
    prior_files = {Path(f["path"]).name: f for f in prior["files"]}
    old_outputs_path = Path("reports/phase2_output_manifest.json")
    old_outputs = json.loads(old_outputs_path.read_text(encoding="utf-8"))
    old_output_hashes = {Path(f["path"]).name: f["sha256"] for f in old_outputs}
    checks = []

    def check(name, passed, detail=None):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    check("previous_raw_acceptance", prior["summary"]["status"] == "PASS")
    accepted_config_path = Path(prior["config_path"])
    accepted_config_hash = sha256(accepted_config_path)
    check("original_config_same_accepted_bytes", prior["config_sha256"] == accepted_config_hash)
    # Git's Windows checkout may convert LF to CRLF. Preserve both hashes and
    # require full text equality after Python's universal newline decoding.
    check("clone_config_equal_except_line_endings", config.read_text(encoding="utf-8") == accepted_config_path.read_text(encoding="utf-8"))
    raw_paths = [f.path for f in cfg.raw_files] + [cfg.paths.raw_land_sea_mask]
    inventory = []
    for path in raw_paths:
        old = prior_files.get(path.name, {})
        before = path.stat()
        digest = sha256(path)
        record = {"path": str(path), "bytes": before.st_size, "sha256": digest,
                  "prior_sha256": old.get("accepted_sha256", old.get("sha256_after")),
                  "prior_quality_status": old.get("quality_status")}
        with xr.open_dataset(path) as ds:
            record["sizes"] = dict(ds.sizes)
            record["variables"] = {v: {"units": ds[v].attrs.get("units"), "dtype": str(ds[v].dtype)} for v in ds.data_vars}
            lat, lon = ds.latitude.values, ds.longitude.values
            check(path.name + ":grid", np.array_equal(lat, np.arange(36, 24.99, -.25)) and np.array_equal(lon, np.arange(112, 123.01, .25)))
            times = pd.DatetimeIndex(ds.valid_time.values)
            record["time_start_utc"], record["time_end_utc"] = str(times.min()), str(times.max())
            if "t2m" in ds:
                raw_file = next(f for f in cfg.raw_files if f.path == path)
                expected = pd.DatetimeIndex([pd.Timestamp(raw_file.year, raw_file.month, day, hour) for day in raw_file.days for hour in raw_file.hours])
                check(path.name + ":timestamps", times.equals(expected))
                check(path.name + ":units", all(str(ds[v].attrs.get("units", "")).lower() in ("k", "kelvin") for v in ("t2m", "d2m")))
        after = path.stat()
        check(path.name + ":same_accepted_bytes", digest == record["prior_sha256"])
        check(path.name + ":unchanged_while_reading", (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns))
        inventory.append(record)
    processed = []
    for stem in ("daily_tmax", "humid_heat", "heat_risk_categories", "heat_index_climatology"):
        path = source_root / "data" / "processed" / f"{stem}_{cfg.file_tag}.nc"
        digest = sha256(path)
        with xr.open_dataset(path) as ds:
            rec = {"path": str(path), "bytes": path.stat().st_size, "sha256": digest,
                   "sizes": dict(ds.sizes), "attrs": ds.attrs,
                   "variables": {v: {"dims": list(ds[v].dims), "dtype": str(ds[v].dtype), "attrs": ds[v].attrs} for v in ds.data_vars}}
            date_dim = "date" if "date" in ds.coords else "valid_time" if "valid_time" in ds.coords else None
            if date_dim:
                dates = pd.DatetimeIndex(ds[date_dim].values)
                check(path.name + ":dates", dates.equals(cfg.local_days))
                rec["days"] = len(dates)
            check(path.name + ":grid", np.array_equal(ds.latitude.values, lat) and np.array_equal(ds.longitude.values, lon))
            if "land_mask" in ds:
                land = ds.land_mask.values == 1
                rec["land_cells"] = int(land.sum())
        check(path.name + ":same_accepted_bytes", digest == old_output_hashes.get(path.name))
        processed.append(rec)
    with xr.open_dataset(cfg.paths.raw_land_sea_mask) as ds:
        fraction = ds.lsm.isel(valid_time=0).values
        check("stored_mask_equals_raw_threshold", np.array_equal(land, fraction >= cfg.land_sea_mask_threshold))
    mesh_lon, mesh_lat = np.meshgrid(lon, lat)
    held = (mesh_lon >= 118) & (mesh_lon <= 121) & (mesh_lat >= 29) & (mesh_lat <= 32)
    expanded = (mesh_lon >= 117.5) & (mesh_lon <= 121.5) & (mesh_lat >= 28.5) & (mesh_lat <= 32.5)
    geometry = {"holdout": int((land & held).sum()), "buffer": int((land & expanded & ~held).sum()), "train": int((land & ~expanded).sum())}

    # One pre-specified training month: direct UTC slicing and reshape, no daily helper.
    # HI uses the existing tested formula; this is consistency, not independent formula validation.
    month = cfg.years[0]
    if month > 2014:
        raise ValueError("Bounded numerical check must stay inside training years")
    parts = []
    for path in (source_root / "data/raw" / f"era5_east_china_{month}_05_31_h16-23.nc", source_root / "data/raw" / f"era5_east_china_{month}_06.nc"):
        with xr.open_dataset(path) as ds:
            parts.append(ds[["t2m", "d2m"]].load())
    hourly = xr.concat(parts, dim="valid_time").sel(valid_time=slice(f"{month}-05-31T16", f"{month}-06-30T15"))
    check("bounded_month_hours", pd.DatetimeIndex(hourly.valid_time.values).equals(pd.date_range(f"{month}-05-31T16", f"{month}-06-30T15", freq="h")))
    shape = (30, 24, len(lat), len(lon))
    t_c = hourly.t2m.values - 273.15
    td_k = hourly.d2m.values
    td_c = td_k - 273.15
    rh = relative_humidity(t_c, td_c)
    hi = heat_index_rothfusz(t_c, rh)
    bounded = {"period_bjt": f"{month}-06-01/{month}-06-30", "grids": len(lat)*len(lon), "method": "direct_hourly_reshape; existing_tested_HI_formula"}
    with xr.open_dataset(cfg.paths.daily_tmax_nc) as daily, xr.open_dataset(cfg.paths.humid_heat_nc) as humid:
        ds = daily.sel(valid_time=slice(f"{month}-06-01", f"{month}-06-30"))
        hh = humid.sel(date=slice(f"{month}-06-01", f"{month}-06-30"))
        pairs = {"tmax_c": (t_c.reshape(shape).max(axis=1), ds.t2m.values),
                 "stored_d2m_is_daily_max_kelvin": (td_k.reshape(shape).max(axis=1), ds.d2m.values),
                 "rh_mean_percent": (rh.reshape(shape).mean(axis=1), hh.daily_mean_rh.values),
                 "hi_max_c": (hi.reshape(shape).max(axis=1), hh.daily_max_heat_index_c.values)}
        for name, (actual, stored) in pairs.items():
            delta = float(np.max(np.abs(actual-stored)))
            bounded[name + "_max_absolute_difference"] = delta
            check("bounded_month:" + name, np.allclose(actual, stored, rtol=0, atol=2e-5, equal_nan=False), delta)
        bounded["daily_dewpoint_max_minus_mean_c_max"] = float((td_c.reshape(shape).max(axis=1)-td_c.reshape(shape).mean(axis=1)).max())
    report = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": "PASS" if all(c["passed"] for c in checks) else "FAIL",
              "scope": "hash and structure audit all years; numerical check June 2000 only; no final-test statistics or model evaluation",
              "source_root": str(source_root), "config_sha256": sha256(config),
              "accepted_config_sha256": accepted_config_hash,
              "config_byte_difference": "Git checkout line endings only; normalized full text checked",
              "previous_input_report": {"path": str(previous_raw), "sha256": sha256(previous_raw), "summary": prior["summary"]},
              "previous_output_manifest_sha256": sha256(old_outputs_path),
              "software": {"python": platform.python_version(), **{p: version(p) for p in ("numpy", "pandas", "xarray", "netCDF4", "h5netcdf", "scipy", "pytest", "scikit-learn", "lightgbm")}},
              "raw_file_count": len(inventory), "raw_bytes": sum(f["bytes"] for f in inventory),
              "land_cells": int(land.sum()), "candidate_spatial_roles_land_cells": geometry,
              "raw_files": inventory, "processed_files": processed, "bounded_numerical_check": bounded,
              "checks": checks,
              "limitations": ["All-year finite-value acceptance is inherited only where exact accepted SHA-256 matches, not recomputed.", "Mean dewpoint, Tmin, complete-night and regional historical features have not yet been extracted.", "ERA5 vintages/as-of availability cannot be reconstructed from current final reanalysis files.", "No training, calibration, model selection or final-test evaluation performed."]}
    report["audit_script_sha256"] = sha256(Path(__file__))
    (output / "data_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str)+"\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": len(checks), "failed": [c for c in checks if not c["passed"]], "raw_files": len(inventory), "land_cells": int(land.sum()), "spatial_roles": geometry, "bounded_check": bounded}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--previous-raw-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.config, args.source_root, args.previous_raw_report, args.output_dir)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

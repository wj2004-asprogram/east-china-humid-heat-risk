"""Extract annual frozen features for 2000-2019 only, with source hashes."""

import argparse
from contextlib import ExitStack
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import pandas as pd
import xarray as xr

from src.config import load_config
from src.era5_utils import detect_time_coord, detect_t2m_var, detect_d2m_var
from src.probability_complete_features import complete_daily_night, assemble_common_year


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path, value):
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def peak_memory_bytes():
    if sys.platform != "win32":
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
             "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    result = Counters()
    result.cb = ctypes.sizeof(result)
    ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(result), result.cb):
        raise ctypes.WinError()
    return int(result.PeakWorkingSetSize)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-project-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("configs/probability_protocol_v1.json"))
    parser.add_argument("--phase1-dir", type=Path, default=Path("data/probability_phase1"))
    parser.add_argument("--year", type=int, choices=range(2000, 2020), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    if args.output_dir.exists():
        parser.error("output-dir must not already exist")
    cfg = load_config(args.config, project_root=args.source_project_root)
    if cfg.utc_offset_hours != 8 or cfg.months != (6, 7, 8) or args.year not in cfg.years:
        parser.error("Requires configured BJT JJA training/calibration year")
    registry_path = args.phase1_dir/"samples/grid_registry.csv"
    sources_path = args.phase1_dir/"geometry/neighbor_sources.json"
    registry = pd.read_csv(registry_path)
    source_rows = json.loads(sources_path.read_text())
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    for path, expected in ((registry_path, protocol["neighborhood"]["grid_registry_sha256"]),
                           (sources_path, protocol["neighborhood"]["source_list_sha256"])):
        if sha256(path) != expected:
            raise ValueError(f"Frozen geometry hash mismatch: {path}")
    paths = [raw.path for raw in cfg.raw_files if raw.year == args.year]
    input_hashes = {str(path.resolve()): sha256(path) for path in paths}
    with ExitStack() as stack:
        parts = []
        for path in paths:
            ds = stack.enter_context(xr.open_dataset(path))
            parts.append(ds[[detect_t2m_var(ds), detect_d2m_var(ds)]])
        time_name = detect_time_coord(parts[0])
        hourly = xr.concat(parts, dim=time_name, data_vars="all", coords="minimal", compat="equals", join="exact").sortby(time_name).load()
    daily = complete_daily_night(hourly, args.year)
    del hourly
    args.output_dir.mkdir(parents=True, exist_ok=False)
    # Preserve computed dtypes; do not introduce a storage cast before labels.
    daily.to_netcdf(args.output_dir/"daily_night.nc", encoding={name: {"zlib": True, "complevel": 2} for name in daily})
    import numpy as np
    counts = {}
    for scope, source_key in (("temporal", "ordinary_source_row_indices"), ("strict", "strict_source_row_indices")):
        sources = [row[source_key] for row in source_rows]
        payload, report, feature_registry = assemble_common_year(daily, registry, sources, strict=scope == "strict")
        np.savez_compressed(args.output_dir/f"{scope}.npz", **payload)
        identity = hashlib.sha256(payload["day_index"].tobytes()+payload["grid_index"].tobytes()).hexdigest()
        counts[scope] = dict(report, sample_index_sha256=identity)
        del payload
    dump(args.output_dir/"feature_registry.json", feature_registry)
    outputs = {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
               for path in args.output_dir.iterdir() if path.is_file()}
    report = {"year": args.year, "purpose": "features_only_no_fitting_or_evaluation",
              "command": sys.argv, "python": platform.python_version(),
              "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "xarray", "netCDF4")},
              "protocol_sha256": sha256(args.protocol), "raw_inputs_sha256": input_hashes,
              "geometry_sha256": {"registry": sha256(registry_path), "sources": sha256(sources_path)},
              "counts": counts, "runtime_seconds": time.perf_counter()-started,
              "peak_working_set_bytes": peak_memory_bytes(), "outputs": outputs,
              "format": "NPZ row key=(year,day_index,grid_index); day_index=0 means June9; grid_index refers frozen registry; X1/X2 identical rows",
              "calibration_year_policy": "2018-2019 features prepared only; no fitting, selection, calibration or evaluation",
              "final_years": "2020-2024 never opened"}
    dump(args.output_dir/"run.json", report)
    print(json.dumps({"year": args.year, "counts": counts, "runtime_seconds": report["runtime_seconds"],
                      "peak_working_set_bytes": report["peak_working_set_bytes"]}), flush=True)


if __name__ == "__main__":
    main()

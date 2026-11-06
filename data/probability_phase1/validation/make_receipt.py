"""Assemble phase-1 evidence; run from this isolated repository root."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/probability_phase1"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


audit = json.loads((DATA / "audit-r2/data_audit.json").read_text(encoding="utf-8"))
assert audit["status"] == "PASS"
registry = pd.read_csv(DATA / "samples/grid_registry.csv")
dates = pd.read_csv(DATA / "samples/date_manifest.csv")
assert len(registry) == 1580 and registry.grid_id.nunique() == 1580
assert dates.common_eligible.sum() == 2100
pilot = pd.read_csv(DATA / "feature-pilot-2000/lagged_feature_pilot.csv")
assert len(pilot) == 42 and pilot.finite_feature_inputs.all()
assert not pilot.duplicated(["sample_id", "cutoff_lag_days"]).any()
assert pilot.sample_id.nunique() == 21
assert pilot.grid_id.isin(registry[registry.spatial_role == "train"].grid_id).all()
for row in pilot.itertuples():
    assert pd.Timestamp(row.source_end_utc_exclusive) <= pd.Timestamp(row.issue_utc)
    assert pd.Timestamp(row.night_end_utc_exclusive) <= pd.Timestamp(row.source_end_utc_exclusive)
    assert (pd.Timestamp(row.target_date) - pd.Timestamp(row.feature_end_date)).days == row.cutoff_lag_days
with xr.open_dataset(DATA / "feature-pilot-2000/daily_features_2000_bjt.nc") as ds:
    assert ds.sizes == {"date": 92, "latitude": 45, "longitude": 45}
    assert len(ds.data_vars) == 7
    assert all(np.isfinite(ds[name].values).all() for name in ds.data_vars)
schema = json.loads((DATA / "feature-pilot-2000/schema.json").read_text())
original_hashes = {record["path"]: record["sha256"] for record in audit["raw_files"]}
used_raw = [{"path": str(Path(path)), "sha256": sha(Path(path))} for path in schema["raw_sources_read_only"]]
assert all(record["sha256"] == original_hashes[record["path"]] for record in used_raw)
tests = ET.parse(DATA / "validation/targeted-tests.xml").getroot().find("testsuite")
assert int(tests.attrib["failures"]) == int(tests.attrib["errors"]) == 0
geometry = json.loads((DATA / "geometry/geometry_audit.json").read_text())
assert geometry["geometry_gate_passed"]
assert sha(DATA / "geometry/neighbor_sources.json") == geometry["neighbor_sources_sha256"]

protocol = json.loads((ROOT / "configs/probability_protocol_v1.json").read_text(encoding="utf-8"))
summary = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "status": "PASS_WITH_DOCUMENTED_LIMITATIONS",
    "base_commit": "3aa2b0872a5d9a11909776229054e0f944e21d5f",
    "branch": "research/probability-audit-20261105",
    "protocol_status": protocol["status"],
    "scientific_scope": "ERA5 reanalysis-conditioned historical hindcast preparation; no trained meteorological model",
    "software": audit["software"],
    "test_counts": {"passed": int(tests.attrib["tests"]), "failures": 0, "errors": 0, "warnings": 6},
    "test_command": "python -B -m pytest tests/test_probability_samples.py tests/test_probability_features.py tests/test_probability_baselines.py tests/test_indices.py tests/test_time_utils.py tests/test_land_sea_mask.py -q -p no:cacheprovider --basetemp <workspace>/pytest-probability-final --junitxml data/probability_phase1/validation/targeted-tests.xml",
    "audit_checks_passed": len(audit["checks"]),
    "data": {"raw_files": 101, "raw_bytes": audit["raw_bytes"], "land_cells": 1580,
             "common_dates": 2100, "structural_cell_days": 3318000,
             "sample_counts_by_year_split": pd.read_csv(DATA / "samples/yearly_counts.csv").groupby("split").common_structural_cell_days.sum().astype(int).to_dict()},
    "pilot": {"year": 2000, "daily_grid_shape": [92, 45, 45], "daily_variables": 7,
              "rows": 42, "distinct_grid_dates": 21, "finite_rows": 42, "all_seven_daily_arrays_finite": True,
              "raw_inputs_after_pilot": used_raw, "target_labels_in_sample_csv": False,
              "full_protocol_feature_union_complete": False},
    "failures_and_resolutions": [
        {"stage": "initial sandbox shell checks", "issue": "Git outside workspace and CIM blocked by sandbox; rg unavailable", "resolution": "read-only approved escalated Git/CIM succeeded; used native PowerShell file discovery"},
        {"stage": "audit initial", "issue": "AttributeError: Paths.land_sea_mask", "resolution": "Use existing cfg.paths.raw_land_sea_mask; no source data edited"},
        {"stage": "audit-r1", "issue": "Only config byte-hash check failed due LF/CRLF checkout", "resolution": "Verify original accepted hash AND entire normalized text; preserve both hashes; audit-r2 PASS"},
        {"stage": "agent synthetic test setup", "issue": "Default pytest temp path inaccessible", "resolution": "Explicit isolated writable --basetemp; combined 56-test run passed"}],
    "artifact_fix": {"issue": "Geometry source-list hash was calculated for LF bytes, while Windows text output added CRLF", "resolution": "Preserved initial output; made geometry writer explicit LF and normalized generated list only; final receipt verified exact file SHA-256"},
    "remaining_work": ["Complete all protocol feature groups and full feature-union sample masks", "Isolate and pin scikit-learn/LightGBM dependencies before their stages", "Fit and validate real baselines only after sample/feature freeze", "Keep final-test evaluation closed until model, calibration and thresholds freeze"],
    "actions_not_taken": ["real model fit", "hyperparameter search", "calibration", "final-test performance calculation", "download", "source data changes", "CDS state changes", "push", "merge", "deployment", "external messaging"],
    "model_assignment": {"scientific_design": "gpt-6-astra xhigh", "engineering": "gpt-6-astra medium", "speed_tier": "not configurable in available subagent interface"},
}
(DATA / "validation/summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
files = sorted(p for p in DATA.rglob("*") if p.is_file() and p.name != "output_manifest.json")
files += sorted(ROOT.glob("src/probability_*.py")) + sorted(ROOT.glob("tools/probability_*.py"))
files += sorted(ROOT.glob("docs/PROBABILITY_*.md")) + sorted(ROOT.glob("tests/test_probability_*.py"))
files += [ROOT / "configs/probability_protocol_v1.json"]
manifest = {"generated_at_utc": summary["created_at_utc"],
            "files": [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p),
                       "local_only": p.suffix == ".nc"} for p in files]}
(DATA / "output_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
print(json.dumps({"status": summary["status"], "tests_passed": int(tests.attrib["tests"]), "files_manifested": len(files), "raw_used_unchanged": len(used_raw), "pilot_rows": len(pilot)}, indent=2))

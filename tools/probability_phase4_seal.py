"""Create immutable phase-four seals, in the authorized scientific order.

This command never fits models or summarizes reserved labels. The scientific
seal uses only pre-existing train/validation metadata. The calibration seal
binds completed mapping/prediction artifacts. The final seal reads only index
arrays and binds feature bytes after calibration parameters have been sealed.
"""
import argparse
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from src.probability_phase4_data import check_gate, sha256


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def file_hashes(paths):
    return {str(p).replace("\\", "/"): sha256(p) for p in sorted(set(map(Path, paths)))}


def assert_expected(expected):
    for filename, digest in expected.items():
        if sha256(filename) != digest:
            raise ValueError(f"Frozen input changed: {filename}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["scientific", "calibration", "test_open"], required=True)
    p.add_argument("--root", type=Path, default=Path("data/probability_phase4"))
    a = p.parse_args(argv)
    config_path = Path("configs/probability_phase4_freeze.json")
    cfg = read_json(config_path)
    models = cfg["models"]
    identities = {(m["scope"], m["lag"], m["model_id"]) for m in models}
    if len(models) != 104 or len(identities) != 104:
        raise ValueError("Expected all 104 unique frozen model configurations")
    target = a.root / (a.stage + "_lock.json")
    if target.exists():
        raise FileExistsError(f"Immutable seal already exists: {target}")
    record = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": a.stage + "_frozen", "protocol_id": cfg["protocol_id"]}
    if a.stage == "scientific":
        expected = dict(cfg["immutable_artifacts_sha256"])
        for m in models:
            expected[m["metadata_path"]] = m["metadata_sha256"]
            expected[m["model_path"]] = m["model_sha256"]
        expected.update(file_hashes([
            "docs/PROBABILITY_MODEL_CARD.md", "docs/PROBABILITY_PHASE4_REPRODUCE.md",
            "data/probability_phase4/engineering_validation/restoration_2000.json",
            "data/probability_phase4/engineering_validation/pre_freeze_verification.json"]))
        assert_expected(expected)
        # Freeze actual inference/evaluation implementation and tests, including
        # reusable older scientific primitives, but not unrelated app files.
        code = [*Path("src").glob("probability_*.py"),
                *Path("tools").glob("probability_phase4_*.py"),
                *Path("tests").glob("test_probability_phase4_*.py"),
                Path("src/config.py"), Path("src/era5_utils.py"),
                Path("src/indices.py"), Path("src/time_utils.py"),
                Path("tools/probability_phase2_features.py")]
        record.update(freeze_config_sha256=sha256(config_path),
                      protocol_document_sha256=sha256(cfg["protocol_document"]),
                      source_code_sha256=file_hashes(code), artifacts_sha256=expected,
                      git_parent_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                      python=platform.python_version(),
                      dependencies={name: importlib.metadata.version(name) for name in
                                    ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "xarray", "netCDF4", "matplotlib", "PyYAML"]},
                      prior_history="2017 and 2022 case context previously seen; not a new blind preregistered experiment")
    elif a.stage == "calibration":
        check_gate(a.root, "calibration")
        records_path = a.root / cfg["output_contract"]["calibration_model_records"]
        if not records_path.is_file():
            raise FileNotFoundError(records_path)
        calibration = read_json(records_path)
        if read_json(a.root / "calibration/execution_state.json")["status"] != "completed":
            raise ValueError("Calibration execution has not completed")
        mappings = calibration["models"]
        if calibration["count"] != 104 or len(mappings) != 104 or {
            (m["scope"], m["lag"], m["model_id"]) for m in mappings
        } != identities:
            raise ValueError("Calibration must cover all frozen models")
        if calibration["freeze_config_sha256"] != sha256(config_path):
            raise ValueError("Calibration configuration mismatch")
        for m in mappings:
            if m["fit_years"] != [2018, 2019]:
                raise ValueError("Calibration years mismatch")
            if m["mapping"]["status"] not in ("fitted", "identity", "persistence_raw"):
                raise ValueError("Unknown mapping status")
            if not 0 <= m["budget"]["threshold"] <= 1.000001:
                raise ValueError("Invalid calibrated alert threshold")
        # Completeness of raw calibration outputs is independent of calibration
        # success; identity fallback is a recorded mapping, never an omitted row.
        required = [records_path]
        for m in models:
            base = a.root / "predictions/calibration" / m["scope"] / f"lag{m['lag']}" / m["model_id"]
            for year in cfg["splits"]["calibration"]:
                raw, receipt = base / f"{year}.npz", base / f"{year}.json"
                r = read_json(receipt)
                if r["output_sha256"] != sha256(raw):
                    raise ValueError(f"Incomplete calibration prediction: {raw}")
                required += [raw, receipt]
        required += [q for folder in ["calibration", "enriched/calibration", "dictionaries"]
                     for q in (a.root / folder).rglob("*") if q.is_file()]
        record.update(scientific_lock_sha256=sha256(a.root / "scientific_lock.json"),
                      calibration_artifacts_sha256=file_hashes(required))
    else:
        check_gate(a.root, "extract_final")
        indices = {}
        required = []
        for year in cfg["splits"]["final_test"]:
            folder = a.root / "features" / str(year)
            receipt = read_json(folder / "run.json")
            for scope in ("temporal", "strict"):
                path = folder / f"{scope}.npz"
                if receipt["outputs"][path.name]["sha256"] != sha256(path):
                    raise ValueError(f"Final feature receipt mismatch: {path}")
                indices[f"{scope}/{year}"] = receipt["counts"][scope]["sample_index_sha256"]
            required += [q for q in folder.iterdir() if q.is_file()]
        record.update(scientific_lock_sha256=sha256(a.root / "scientific_lock.json"),
                      calibration_lock_sha256=sha256(a.root / "calibration_lock.json"),
                      final_input_artifacts_sha256=file_hashes(required),
                      final_common_index_sha256=indices,
                      permitted_operation="One frozen formal evaluation, with recorded technical recovery only")
    a.root.mkdir(parents=True, exist_ok=True)
    # Exclusive-create means an earlier seal cannot be silently changed.
    with target.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    print(json.dumps({"seal": str(target), "sha256": sha256(target), "status": record["status"]}))


if __name__ == "__main__":
    main()

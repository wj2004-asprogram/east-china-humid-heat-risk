"""Execute the frozen calibration stage after the external scientific seal."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.probability_phase4_calibration import fit_platt, budget_threshold, calibration_diagnostic
from src.probability_phase4_data import check_gate, read_logged, sha256
from src.probability_phase4_evaluation import score_probability


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Refusing to emit an unexplained empty report: " + str(path))
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path, ledger, purpose):
    with read_logged(path, ledger, purpose) as logged:
        return json.loads(logged.read_text(encoding="utf-8-sig"))


def load_run(root, config_path, stage):
    gates = check_gate(root, stage)
    ledger = root / "access_ledger.jsonl"
    lock = read_json(root / "scientific_lock.json", ledger, "verify_scientific_seal")
    if sha256(config_path) != lock["freeze_config_sha256"]:
        raise ValueError("Requested config does not match scientific seal")
    config = read_json(config_path, ledger, "frozen_scientific_configuration")
    keys = [(m["scope"], m["lag"], m["model_id"]) for m in config["models"]]
    if len(keys) != 104 or len(set(keys)) != 104:
        raise ValueError("Frozen manifest must contain exactly 104 unique model instances")
    return config, gates, ledger


def start_execution(directory, gates, config_path):
    directory.mkdir(parents=True, exist_ok=True)
    state = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
             "freeze_config_sha256": sha256(config_path), "gates": gates}
    target = directory / "execution_state.json"
    if target.exists():
        previous = json.loads(target.read_text(encoding="utf-8-sig"))
        if previous["status"] != "running" or previous["gates"] != gates or previous["freeze_config_sha256"] != state["freeze_config_sha256"]:
            raise ValueError("Existing run is complete/failed or has different seals; requires documented bug reopening")
        previous["resume_count"] = previous.get("resume_count", 0) + 1
        write_json(target, previous)
        return previous
    with target.open("x", encoding="utf-8") as stream:
        json.dump(state, stream, indent=2)
    return state


def read_prediction(root, stage, model, year, gates, ledger, index_registry):
    path = root / "predictions" / stage / model["scope"] / f"lag{model['lag']}" / model["model_id"] / f"{year}.npz"
    receipt = read_json(path.with_suffix(".json"), ledger, "prediction_integrity_receipt")
    if receipt["gates"] != gates or sha256(path) != receipt["output_sha256"]:
        raise ValueError("Prediction provenance differs from sealed inputs: " + str(path))
    with read_logged(path, ledger, "calibration_prediction" if stage == "calibration" else "one_formal_final_evaluation") as logged:
        with np.load(logged, allow_pickle=False) as data:
            arrays = {k: data[k] for k in ("p_raw", "y", "evaluation_previous_y", "day_index", "grid_index")}
    n = len(arrays["p_raw"])
    if not n or any(v.shape != (n,) for v in arrays.values()):
        raise ValueError("Prediction arrays must align and be nonempty")
    day, grid = arrays["day_index"], arrays["grid_index"]
    if not np.issubdtype(day.dtype, np.integer) or not np.issubdtype(grid.dtype, np.integer):
        raise ValueError("Prediction indices must be integers")
    if ((day < 0) | (day >= 84)).any() or (grid < 0).any():
        raise ValueError("Prediction indices outside frozen season/grid")
    combined = day.astype(np.int64) * 1000000 + grid
    if (np.diff(combined) <= 0).any():
        raise ValueError("Prediction keys must be unique and canonically sorted")
    canonical = hashlib.sha256(day.tobytes() + grid.tobytes()).hexdigest()
    if canonical != receipt["sample_index_sha256"]:
        raise ValueError("Prediction index receipt mismatch")
    if stage == "final_test":
        seal = json.loads((root / "test_open_lock.json").read_text(encoding="utf-8-sig"))
        if canonical != seal["final_common_index_sha256"][f"{model['scope']}/{year}"]:
            raise ValueError("Prediction indices differ from final common-index seal")
    joined = hashlib.sha256(b"".join(arrays[k].astype("<i8").tobytes() for k in
                                    ("day_index", "grid_index", "y", "evaluation_previous_y"))).hexdigest()
    key = (model["scope"], year)
    if index_registry.setdefault(key, joined) != joined:
        raise ValueError("Models/lags do not share identical indices and evaluation labels")
    return arrays, {str(path): receipt["output_sha256"], str(path.with_suffix(".json")): sha256(path.with_suffix(".json"))}


def read_geometry(root, year, ledger):
    path = root / "dictionaries" / str(year) / "grid_dictionary.csv"
    with read_logged(path, ledger, "geometry_dictionary") as logged:
        geometry = pd.read_csv(logged).set_index("grid_index", drop=False)
    return geometry


def check_spatial(arrays, geometry, scope, stage):
    selected = geometry.loc[arrays["grid_index"]]
    if scope == "strict":
        role = "train" if stage == "calibration" else "holdout"
        if not selected.spatial_role.eq(role).all():
            raise ValueError("Forbidden strict spatial role in prediction bundle")
    return selected


def enrich(root, stage, model, year, arrays, calibrated, threshold, protocol_id):
    path = root / "enriched" / stage / model["scope"] / f"lag{model['lag']}" / model["model_id"] / f"{year}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(arrays, p_calibrated=calibrated, alert_primary=(calibrated >= .5).astype(np.int8),
                   alert_budget=(calibrated >= threshold).astype(np.int8), primary_threshold=np.array(.5),
                   budget_threshold=np.array(threshold), lag_days=np.array(model["lag"]), model_id=np.array(model["model_id"]),
                   protocol_version=np.array(protocol_id), sample_eligibility=np.ones(len(calibrated), dtype=np.int8))
    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            if set(existing.files) != set(payload) or any(not np.array_equal(existing[k], v) for k, v in payload.items()):
                raise ValueError("Interrupted enriched output differs from deterministic recomputation: " + str(path))
    else:
        np.savez_compressed(path, **payload)
    return path


def checkpoint_path(out, model):
    return out / "checkpoints" / f"{model['scope']}_lag{model['lag']}_{model['model_id']}.json"


def load_checkpoint(path, state, ledger):
    if not path.exists():
        return None
    checkpoint = read_json(path, ledger, "resume_completed_model_checkpoint")
    if checkpoint["gates"] != state["gates"] or checkpoint["freeze_config_sha256"] != state["freeze_config_sha256"]:
        raise ValueError("Checkpoint differs from sealed run")
    for artifact, expected in checkpoint["artifacts_sha256"].items():
        if sha256(artifact) != expected:
            raise ValueError("Checkpoint artifact changed: " + artifact)
    return checkpoint


def save_checkpoint(path, state, artifacts, **payload):
    checkpoint = {"gates": state["gates"], "freeze_config_sha256": state["freeze_config_sha256"],
                  "artifacts_sha256": {str(p): sha256(p) for p in artifacts}, **payload}
    temporary = path.with_suffix(".json.part")
    write_json(temporary, checkpoint)
    temporary.replace(path)


def run(root, config_path):
    root, config_path = Path(root), Path(config_path)
    config, gates, ledger = load_run(root, config_path, "calibration")
    out = root / "calibration"
    state = start_execution(out, gates, config_path)
    records, diagnostics, index_registry = [], [], {}
    years = config["splits"]["calibration"]
    geometry = {year: read_geometry(root, year, ledger) for year in years}
    try:
        for model in config["models"]:
            checkpoint_file = checkpoint_path(out, model)
            checkpoint = load_checkpoint(checkpoint_file, state, ledger)
            if checkpoint is not None:
                records.append(checkpoint["record"])
                diagnostics.extend(checkpoint["diagnostics"])
                for scope, year, digest in checkpoint["index_registry"]:
                    if index_registry.setdefault((scope, year), digest) != digest:
                        raise ValueError("Checkpoint common index mismatch")
                print(json.dumps({"resumed_calibrated_instances": len(records)}), flush=True)
                continue
            first_diagnostic = len(diagnostics)
            bundles, inputs = [], {}
            for year in years:
                arrays, hashes = read_prediction(root, "calibration", model, year, gates, ledger, index_registry)
                check_spatial(arrays, geometry[year], model["scope"], "calibration")
                bundles.append(arrays)
                inputs.update(hashes)
            mapping = fit_platt(np.concatenate([a["y"] for a in bundles]),
                                np.concatenate([a["p_raw"] for a in bundles]),
                                np.concatenate([np.repeat(year, len(a["y"])) for year, a in zip(years, bundles)]),
                                persistence=model["model_id"] == "persistence")
            calibrated = [mapping.apply(a["p_raw"]) for a in bundles]
            budget = budget_threshold(calibrated)
            record = {k: model[k] for k in ("scope", "lag", "model_id")}
            record.update(mapping=mapping.to_dict(), budget=budget, fit_years=years, inputs_sha256=inputs)
            records.append(record)
            artifacts = list(inputs)
            for year, arrays, p_cal in zip(years, bundles, calibrated):
                selected = geometry[year].loc[arrays["grid_index"]]
                for kind, p in [("raw", arrays["p_raw"]), ("calibrated", p_cal)]:
                    diagnostics.append({"scope": model["scope"], "lag": model["lag"], "model_id": model["model_id"],
                        "year": year, "probability": kind, "interpretation": "calibration_in_sample_not_generalization",
                        **score_probability(arrays["y"], p, selected.latitude.to_numpy()),
                        **calibration_diagnostic(arrays["y"], p, np.repeat(year, len(p)))})
                artifacts.append(enrich(root, "calibration", model, year, arrays, p_cal, budget["threshold"], config["protocol_id"]))
            save_checkpoint(checkpoint_file, state, artifacts, record=record, diagnostics=diagnostics[first_diagnostic:],
                            index_registry=[[scope, year, digest] for (scope, year), digest in index_registry.items() if scope == model["scope"]])
            print(json.dumps({"calibrated_instances": len(records), "total": len(config["models"]),
                              "scope": model["scope"], "lag": model["lag"], "model_id": model["model_id"],
                              "mapping_status": mapping.status}), flush=True)
        write_json(out / "calibrators.json", {"schema_version": 1, "protocol_id": config["protocol_id"],
                   "freeze_config_sha256": sha256(config_path), "gates": gates, "models": records, "count": len(records)})
        write_csv(out / "diagnostics.csv", diagnostics)
        state.update(status="completed", count=len(records), completed_utc=datetime.now(timezone.utc).isoformat(),
                     fallback_count=sum(r["mapping"]["status"] == "identity" for r in records))
    except Exception as exc:
        state.update(status="failed_requires_documented_bug_reopen", error=repr(exc), completed_instances=len(records))
        raise
    finally:
        write_json(out / "execution_state.json", state)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/probability_phase4"))
    parser.add_argument("--config", type=Path, default=Path("configs/probability_phase4_freeze.json"))
    args = parser.parse_args(argv)
    print(json.dumps(run(args.root, args.config)), flush=True)


if __name__ == "__main__":
    main()

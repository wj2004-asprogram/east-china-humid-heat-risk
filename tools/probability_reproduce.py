"""Replay frozen probability models in a new directory, without training or calibration.

The publication manifest is the recovery contract. Historical seals remain evidence
of the original experiment; they are neither modified nor used as recovery locks.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import platform
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import xarray as xr

from src.era5_utils import detect_time_coord, detect_t2m_var, detect_d2m_var
from src.probability_complete_features import complete_daily_night, assemble_common_year
from src.probability_phase4_calibration import PlattMapping
from src.probability_phase4_data import complete_final_daily_night, assemble_phase4, predict_frozen
from src.probability_phase4_evaluation import risk_masks, score_probability


PATHS = {
    "config": "configs/probability_phase4_freeze.json",
    "calibrators": "data/probability_phase4/calibration/calibrators.json",
    "grid_registry": "data/probability_phase1/samples/grid_registry.csv",
    "neighbor_sources": "data/probability_phase1/geometry/neighbor_sources.json",
    "feature_registry": "data/probability_phase2/features/2000/feature_registry.json",
    "annual_scores": "data/probability_phase4/evaluation/annual_scores.csv",
}
PROBABILITY_ARRAYS = {"p_raw", "p_calibrated"}
RISK_SETS = ("all", "onset", "persistence_or_clearance")
YEARS = tuple(range(2018, 2025))
REQUIRED_SCIENTIFIC_FILES = tuple(PATHS.values()) + (
    "tools/probability_reproduce.py", "tools/__init__.py", "src/__init__.py",
    "src/era5_utils.py", "src/indices.py", "src/time_utils.py",
    "src/probability_samples.py", "src/probability_features.py",
    "src/probability_complete_features.py", "src/probability_baselines.py",
    "src/probability_metrics.py", "src/probability_phase4_data.py",
    "src/probability_phase4_calibration.py", "src/probability_phase4_evaluation.py",
    "configs/probability_protocol_v1.json",
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_array_digest(values, decimals=None):
    """Hash dtype, shape and little-endian C-order values; independent of NPZ packaging."""
    values = np.asarray(values)
    if values.dtype.hasobject:
        raise ValueError("Object arrays are not an auditable numeric representation")
    if decimals is not None:
        if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
            raise ValueError("Rounded fingerprints require finite floating-point arrays")
        values = np.round(values.astype(np.float64), decimals=decimals)
        # Treat signed zero identically after rounding.
        values = np.where(values == 0, 0., values)
    dtype = values.dtype.newbyteorder("<")
    normalized = np.ascontiguousarray(values.astype(dtype, copy=False))
    header = json.dumps({"dtype": dtype.str, "shape": list(values.shape)},
                        sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(header + b"\n" + normalized.tobytes(order="C")).hexdigest()


def era5_content_digest(dataset):
    """Fingerprint scientific input values, excluding NetCDF packaging and history.

    Coordinate order and variable names are normalized. Temperatures and coordinates
    are represented as float64, time as datetime64[ns]. No rounding is performed.
    Different scientific values are never accepted merely because metadata matches.
    """
    time_name = detect_time_coord(dataset)
    t_name, td_name = detect_t2m_var(dataset), detect_d2m_var(dataset)
    for name in (t_name, td_name):
        if str(dataset[name].attrs.get("units", "")).lower() not in ("k", "kelvin"):
            raise ValueError("ERA5 temperature and dewpoint must declare Kelvin units")
    for name in (time_name, "latitude", "longitude"):
        values = np.asarray(dataset[name].values)
        if values.ndim != 1 or len(np.unique(values)) != len(values):
            raise ValueError("ERA5 coordinates must be one-dimensional and unique")
        if name == time_name:
            dates = pd.DatetimeIndex(values)
            if dates.hasnans or dates.tz is not None or not dates.equals(dates.floor("h")):
                raise ValueError("ERA5 timestamps must be finite whole UTC hours")
        elif not np.isfinite(values).all():
            raise ValueError("ERA5 coordinates must be finite")
    ordered = dataset.sortby([time_name, "latitude", "longitude"])
    arrays = {
        "time_utc": ordered[time_name].values.astype("datetime64[ns]"),
        "latitude": ordered.latitude.values.astype(np.float64),
        "longitude": ordered.longitude.values.astype(np.float64),
        "t2m_kelvin": ordered[t_name].transpose(time_name, "latitude", "longitude").values.astype(np.float64),
        "d2m_kelvin": ordered[td_name].transpose(time_name, "latitude", "longitude").values.astype(np.float64),
    }
    fingerprints = {name: canonical_array_digest(value) for name, value in arrays.items()}
    return hashlib.sha256(json.dumps(fingerprints, sort_keys=True,
                                    separators=(",", ":")).encode("ascii")).hexdigest()


def prediction_reference(arrays):
    """Build a portable, non-sensitive reference entry from an original NPZ payload."""
    fingerprints = {}
    for name, value in arrays.items():
        record = {"sha256": canonical_array_digest(value)}
        if name in PROBABILITY_ARRAYS:
            record["rounded_12_sha256"] = canonical_array_digest(value, decimals=12)
        fingerprints[name] = record
    return {"n": len(arrays["y"]), "arrays": fingerprints}


def verify_prediction(arrays, reference):
    expected = reference["arrays"]
    if set(arrays) != set(expected) or len(arrays["y"]) != reference["n"]:
        raise ValueError("Recovered prediction fields or row count differ from reference")
    rounded = []
    for name, value in arrays.items():
        record = expected[name]
        if canonical_array_digest(value) == record["sha256"]:
            continue
        if name in PROBABILITY_ARRAYS and record.get("rounded_12_sha256") == canonical_array_digest(value, decimals=12):
            rounded.append(name)
            continue
        raise ValueError("Recovered prediction differs from reference array: " + name)
    return {"status": "exact" if not rounded else "probabilities_match_12_decimal_fingerprint",
            "rounded_probability_arrays": rounded, "n": len(arrays["y"])}


def repository_path(root, relative):
    relative = str(relative).replace("\\", "/")
    part = PurePosixPath(relative)
    if part.is_absolute() or ":" in relative or ".." in part.parts or not part.parts:
        raise ValueError("Manifest paths must be repository-relative: " + relative)
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("Manifest path escapes the repository")
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def verify_manifest(repo_root, manifest_path):
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported publication recovery manifest")
    required = dict(PATHS)
    required.update(manifest.get("paths", {}))
    hashes = manifest["scientific_files_sha256"]
    for path in set(required.values()) | set(REQUIRED_SCIENTIFIC_FILES):
        if path not in hashes:
            raise ValueError("Manifest omits required scientific file: " + path)
    for relative, expected in hashes.items():
        if file_sha256(repository_path(repo_root, relative)) != expected:
            raise ValueError("Scientific file hash mismatch: " + relative)
    config = read_json(repository_path(repo_root, required["config"]))
    models = config["models"]
    identities = {(m["scope"], m["lag"], m["model_id"]) for m in models}
    if len(models) != 104 or len(identities) != 104:
        raise ValueError("Recovery requires all 104 frozen model registrations")
    for model in models:
        for path_key, hash_key in (("metadata_path", "metadata_sha256"), ("model_path", "model_sha256")):
            path = model[path_key]
            if hashes.get(path) != model[hash_key]:
                raise ValueError("Model registration differs from publication manifest: " + path)
    calibrators = read_json(repository_path(repo_root, required["calibrators"]))["models"]
    mappings = {(m["scope"], m["lag"], m["model_id"]): m for m in calibrators}
    if len(calibrators) != 104 or set(mappings) != identities:
        raise ValueError("Calibrators must cover the same 104 frozen model instances")
    if any(m["fit_years"] != [2018, 2019] for m in calibrators):
        raise ValueError("Unexpected calibration years")
    raw = manifest["raw_files"]
    raw_names = [r["name"] for r in raw]
    if len(raw_names) != len(set(raw_names)) or any(Path(n).name != n or "/" in n or "\\" in n for n in raw_names):
        raise ValueError("Raw input names must be unique basenames")
    expected_raw = {f"era5_east_china_{year}_{month:02d}.nc" for year in YEARS for month in (6, 7, 8)}
    expected_raw |= {f"era5_east_china_{year}_05_31_h16-23.nc" for year in YEARS}
    if set(raw_names) != expected_raw:
        raise ValueError("Recovery manifest must contain exactly the 28 required hourly inputs")
    for record in raw:
        if record["year"] not in YEARS or not record["name"].startswith(f"era5_east_china_{record['year']}_"):
            raise ValueError("Raw input year does not match its filename")
    references = manifest["reference_predictions"]
    expected_keys = {prediction_key(year, m) for year in YEARS for m in models}
    if set(references) != expected_keys:
        raise ValueError("Recovery manifest must bind all 728 frozen prediction bundles")
    return manifest, required, config, mappings


def prediction_key(year, model):
    stage = "calibration" if year < 2020 else "final_test"
    return f"{stage}/{model['scope']}/lag{model['lag']}/{model['model_id']}/{year}"


def validate_raw(path, record):
    digest = file_sha256(path)
    if digest == record["file_sha256"]:
        return {"name": record["name"], "file_sha256": digest, "status": "byte_match"}
    with xr.open_dataset(path) as dataset:
        content = era5_content_digest(dataset)
    if content != record["content_sha256"]:
        raise ValueError("Raw scientific content differs from the frozen reference: " + record["name"])
    return {"name": record["name"], "file_sha256": digest,
            "content_sha256": content, "status": "content_match"}


def validate_output_location(repo_root, raw_dir, output):
    output = output.resolve()
    for protected in (repo_root / "data", repo_root / "src", repo_root / "tools", repo_root / "configs", repo_root / "publication", repo_root / ".git", raw_dir):
        protected = protected.resolve()
        if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("Output must be separate from scientific source and raw data directories")
    if output.exists():
        raise FileExistsError("Output directory already exists; choose a new recovery directory")
    return output


def enrich_prediction(payload, probability, model, mapping_record, protocol_id):
    mapping = PlattMapping(**mapping_record["mapping"])
    calibrated = mapping.apply(probability)
    threshold = mapping_record["budget"]["threshold"]
    return {"p_raw": probability, "y": payload["y"], "evaluation_previous_y": payload["previous_y"],
            "day_index": payload["day_index"], "grid_index": payload["grid_index"],
            "p_calibrated": calibrated, "alert_primary": (calibrated >= .5).astype(np.int8),
            "alert_budget": (calibrated >= threshold).astype(np.int8), "primary_threshold": np.array(.5),
            "budget_threshold": np.array(threshold), "lag_days": np.array(model["lag"]),
            "model_id": np.array(model["model_id"]), "protocol_version": np.array(protocol_id),
            "sample_eligibility": np.ones(len(probability), dtype=np.int8)}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def verify_annual_score(score, reference):
    if score["n"] != reference.n or score["events"] != reference.events:
        raise ValueError("Recovered annual sample or event count differs from reference")
    if score["brier"] is None:
        if not pd.isna(reference.brier):
            raise ValueError("Recovered undefined Brier score differs from reference")
    elif not np.isfinite(reference.brier) or abs(score["brier"] - reference.brier) > 1e-12:
        raise ValueError("Recovered annual Brier score differs from reference")


def run(args):
    repo_root = args.repo_root.resolve()
    manifest_path = repository_path(repo_root, args.manifest)
    manifest, paths, config, mappings = verify_manifest(repo_root, manifest_path)
    years = sorted(set(args.years))
    if not years or not set(years) <= set(YEARS):
        raise ValueError("Recovery years must be a nonempty subset of 2018-2024")
    requested = set(args.models) if args.models else None
    available = {m["model_id"] for m in config["models"]}
    if requested is not None and not requested <= available:
        raise ValueError("Unknown requested models: " + ", ".join(sorted(requested - available)))
    models = [m for m in config["models"] if requested is None or m["model_id"] in requested]
    summary = {"schema_version": 1, "source_commit": manifest.get("source_commit"),
               "manifest_sha256": file_sha256(manifest_path), "years": years,
               "model_ids": sorted({m["model_id"] for m in models}), "model_instances": len(models),
               "expected_prediction_bundles": len(years) * len(models),
               "scope": "complete_recovery" if years == list(YEARS) and len(models) == 104 else "partial_recovery",
               "interpretation": "replay_of_previously_evaluated_frozen_hindcasts_not_a_new_blind_test",
               "training_or_calibration_performed": False}
    if args.verify_only:
        return dict(summary, status="manifest_verified", raw_inputs_checked=False)
    if args.raw_dir is None or args.output_dir is None:
        raise ValueError("--raw-dir and --output-dir are required unless --verify-only is used")
    raw_dir = args.raw_dir.resolve()
    output = validate_output_location(repo_root, raw_dir, args.output_dir)
    raw_records = [r for r in manifest["raw_files"] if r["year"] in years]
    raw_checks = [validate_raw(raw_dir / r["name"], r) for r in raw_records]
    registry = pd.read_csv(repository_path(repo_root, paths["grid_registry"]))
    sources = read_json(repository_path(repo_root, paths["neighbor_sources"]))
    feature_registry = read_json(repository_path(repo_root, paths["feature_registry"]))
    references = pd.read_csv(repository_path(repo_root, paths["annual_scores"]), float_precision="round_trip")
    score_index = {(r.scope, r.lag, r.model_id, r.year, r.probability, r.risk_set): r for r in references.itertuples()}
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(summary, status="running", started_utc=datetime.now(timezone.utc).isoformat(),
                   raw_validation=raw_checks, prediction_checks=[], python=platform.python_version(),
                   dependencies={n: importlib.metadata.version(n) for n in ("numpy", "pandas", "xarray", "netCDF4", "scipy", "lightgbm")})
    write_json(output / "reproduction_receipt.json", receipt)
    scores = []
    try:
        for year in years:
            with ExitStack() as stack:
                parts = [stack.enter_context(xr.open_dataset(raw_dir / r["name"])) for r in raw_records if r["year"] == year]
                parts = [d[[detect_t2m_var(d), detect_d2m_var(d)]] for d in parts]
                time_name = detect_time_coord(parts[0])
                hourly = xr.concat(parts, dim=time_name, data_vars="all", coords="minimal", compat="equals", join="exact").sortby(time_name).load()
            daily = complete_daily_night(hourly, year) if year < 2020 else complete_final_daily_night(hourly, year)
            del hourly
            for scope, source_key in (("temporal", "ordinary_source_row_indices"), ("strict", "strict_source_row_indices")):
                source_lists = [r[source_key] for r in sources]
                if year < 2020:
                    payload, _, schema = assemble_common_year(daily, registry, source_lists, strict=scope == "strict")
                else:
                    payload, _, schema = assemble_phase4(daily, registry, source_lists, scope=scope, stage="final_test")
                if schema != feature_registry:
                    raise ValueError("Reconstructed feature schema differs from the frozen registry")
                if scope == "strict":
                    expected_role = "train" if year < 2020 else "holdout"
                    if not registry.iloc[payload["grid_index"]].spatial_role.eq(expected_role).all():
                        raise ValueError("Recovered strict samples use a forbidden spatial role")
                if args.save_features:
                    folder = output / "features" / str(year)
                    folder.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(folder / f"{scope}.npz", **payload)
                grid = registry.iloc[payload["grid_index"]]
                for model in (m for m in models if m["scope"] == scope):
                    state = read_json(repository_path(repo_root, model["metadata_path"]))
                    booster = None
                    if model["model_id"].startswith("lightgbm_"):
                        import lightgbm as lgb
                        booster = lgb.Booster(model_file=str(repository_path(repo_root, model["model_path"])))
                    p = predict_frozen(state, payload, schema["names"], registry, year, model["lag"], scope, model["model_id"], booster)
                    identity = (scope, model["lag"], model["model_id"])
                    arrays = enrich_prediction(payload, p, model, mappings[identity], config["protocol_id"])
                    key = prediction_key(year, model)
                    check = verify_prediction(arrays, manifest["reference_predictions"][key])
                    for risk, mask in risk_masks(arrays["y"], arrays["evaluation_previous_y"]).items():
                        if risk not in RISK_SETS:
                            continue
                        for kind, p_key in (("raw", "p_raw"), ("calibrated", "p_calibrated")):
                            score = score_probability(arrays["y"][mask], arrays[p_key][mask], grid.latitude.to_numpy()[mask])
                            if year >= 2020:
                                ref = score_index[(scope, model["lag"], model["model_id"], year, kind, risk)]
                                verify_annual_score(score, ref)
                            scores.append(dict(scope=scope, lag=model["lag"], model_id=model["model_id"], year=year,
                                               probability=kind, risk_set=risk, **score))
                    target = output / "predictions" / (key + ".npz")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(target, **arrays)
                    receipt["prediction_checks"].append(dict(key=key, output_sha256=file_sha256(target), **check))
                print(json.dumps({"recovered_year": year, "scope": scope, "completed_prediction_bundles": len(receipt["prediction_checks"])}), flush=True)
                write_json(output / "reproduction_receipt.json", receipt)
                del payload
            del daily
        for relative, expected in manifest["scientific_files_sha256"].items():
            if file_sha256(repository_path(repo_root, relative)) != expected:
                raise ValueError("Scientific file changed during recovery: " + relative)
        for record in raw_checks:
            if file_sha256(raw_dir / record["name"]) != record["file_sha256"]:
                raise ValueError("Raw input changed during recovery: " + record["name"])
        if file_sha256(manifest_path) != receipt["manifest_sha256"]:
            raise ValueError("Recovery manifest changed during execution")
        pd.DataFrame(scores).to_csv(output / "annual_scores.csv", index=False, float_format="%.17g")
        receipt.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat(),
                       annual_scores_sha256=file_sha256(output / "annual_scores.csv"),
                       inputs_unchanged_after_recovery=True,
                       final_test_brier_absolute_tolerance=1e-12)
    except BaseException as exc:
        # Do not copy potentially private absolute paths from exception messages.
        receipt.update(status="failed", error_type=type(exc).__name__)
        write_json(output / "reproduction_receipt.json", receipt)
        raise
    write_json(output / "reproduction_receipt.json", receipt)
    return {k: receipt[k] for k in ("status", "scope", "years", "model_instances", "expected_prediction_bundles")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", default="publication/recovery_manifest.json")
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--years", type=int, choices=YEARS, nargs="+", default=list(YEARS))
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--save-features", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    main()

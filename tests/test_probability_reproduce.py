"""Recovery safeguards: scientific inputs, frozen labels and isolated outputs."""
from argparse import Namespace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tools import probability_reproduce as recovery


def weather():
    values = np.arange(12, dtype=np.float32).reshape(3, 2, 2) / 100 + 300
    return xr.Dataset({
        "t2m": (("valid_time", "latitude", "longitude"), values, {"units": "K"}),
        "d2m": (("valid_time", "latitude", "longitude"), values - 5, {"units": "K"}),
    }, coords={"valid_time": pd.date_range("2020-06-01", periods=3, freq="h"),
               "latitude": [31., 30.], "longitude": [120., 121.]},
        attrs={"history": "original packaging"})


def test_array_fingerprint_preserves_shape_and_normalizes_endianness():
    a = np.array([1., 2.], dtype="<f8")
    assert recovery.canonical_array_digest(a) == recovery.canonical_array_digest(a.astype(">f8"))
    assert recovery.canonical_array_digest(a) != recovery.canonical_array_digest(a.reshape(1, 2))
    with pytest.raises(ValueError, match="Object"):
        recovery.canonical_array_digest(np.array([object()]))


def test_raw_metadata_and_order_can_change_but_values_and_units_cannot(tmp_path):
    original = weather()
    path = tmp_path / "input.nc"
    original.to_netcdf(path)
    record = {"name": path.name, "file_sha256": recovery.file_sha256(path),
              "content_sha256": recovery.era5_content_digest(original)}
    assert recovery.validate_raw(path, record)["status"] == "byte_match"
    changed = original.sortby(["latitude", "longitude", "valid_time"], ascending=False)
    changed.attrs["history"] = "new container; scientifically identical"
    changed.to_netcdf(path)
    assert recovery.validate_raw(path, record)["status"] == "content_match"
    changed = changed.copy(deep=True)
    changed["t2m"].values[0, 0, 0] += .01
    changed.to_netcdf(path)
    with pytest.raises(ValueError, match="scientific content"):
        recovery.validate_raw(path, record)
    changed["t2m"].attrs["units"] = "degC"
    with pytest.raises(ValueError, match="Kelvin"):
        recovery.era5_content_digest(changed)


def test_raw_digest_rejects_duplicate_and_non_hourly_time():
    ds = weather()
    with pytest.raises(ValueError, match="unique"):
        recovery.era5_content_digest(ds.assign_coords(valid_time=[ds.valid_time.values[0]] * 3))
    with pytest.raises(ValueError, match="whole UTC hours"):
        recovery.era5_content_digest(ds.assign_coords(valid_time=ds.valid_time + np.timedelta64(1, "m")))


def test_prediction_reference_never_tolerates_label_or_alert_changes():
    arrays = {"y": np.array([0, 1], dtype=np.int8),
              "p_raw": np.array([.1234567890121, .9]),
              "p_calibrated": np.array([.2, .95]),
              "alert_primary": np.array([0, 1], dtype=np.int8)}
    reference = recovery.prediction_reference(arrays)
    assert recovery.verify_prediction(arrays, reference)["status"] == "exact"
    close = dict(arrays, p_raw=arrays["p_raw"] + 1e-14)
    assert recovery.verify_prediction(close, reference)["rounded_probability_arrays"] == ["p_raw"]
    for name in ("y", "alert_primary"):
        wrong = dict(arrays, **{name: 1 - arrays[name]})
        with pytest.raises(ValueError, match=name):
            recovery.verify_prediction(wrong, reference)
    with pytest.raises(ValueError, match="p_raw"):
        recovery.verify_prediction(dict(arrays, p_raw=arrays["p_raw"] + 1e-5), reference)


def test_paths_and_existing_outputs_are_protected(tmp_path):
    repo, raw = tmp_path / "repo", tmp_path / "raw"
    repo.mkdir(); raw.mkdir()
    for relative in ("../elsewhere", "C:/private/file", "/absolute", "data/../../escape"):
        with pytest.raises(ValueError):
            recovery.repository_path(repo, relative)
    for target in (repo, repo / "data" / "new", repo / ".git" / "new", raw / "new"):
        with pytest.raises(ValueError):
            recovery.validate_output_location(repo, raw, target)
    existing = repo / ".reproduce" / "existing"
    existing.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        recovery.validate_output_location(repo, raw, existing)
    assert recovery.validate_output_location(repo, raw, repo / ".reproduce" / "new") == repo / ".reproduce" / "new"


def test_manifest_missing_scientific_binding_fails_without_output(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "scientific_files_sha256": {}}))
    with pytest.raises(ValueError, match="omits required scientific file"):
        recovery.verify_manifest(tmp_path, path)


def test_annual_score_keeps_undefined_empty_risk_sets():
    score = {"n": 0, "events": 0, "brier": None}
    recovery.verify_annual_score(score, SimpleNamespace(n=0, events=0, brier=np.nan))
    with pytest.raises(ValueError, match="undefined"):
        recovery.verify_annual_score(score, SimpleNamespace(n=0, events=0, brier=0.))
    with pytest.raises(ValueError, match="count"):
        recovery.verify_annual_score(score, SimpleNamespace(n=1, events=0, brier=np.nan))


def test_small_replay_uses_frozen_state_and_writes_only_new_output(tmp_path, monkeypatch):
    repo, raw = tmp_path / "repo", tmp_path / "raw"
    repo.mkdir(); raw.mkdir()
    manifest_path = repo / "manifest.json"
    manifest_path.write_text("{}")
    source_path = raw / "input.nc"
    weather().to_netcdf(source_path)
    registry = pd.DataFrame({"latitude": [30., 31.], "longitude": [120., 121.],
                             "spatial_role": ["train", "holdout"], "grid_id": ["a", "b"]})
    registry.to_csv(repo / "grid.csv", index=False)
    (repo / "neighbors.json").write_text(json.dumps([
        {"ordinary_source_row_indices": [], "strict_source_row_indices": []},
        {"ordinary_source_row_indices": [], "strict_source_row_indices": []}]))
    schema = {"names": ["x"], "groups": ["C"]}
    (repo / "features.json").write_text(json.dumps(schema))
    (repo / "model.json").write_text("{}")
    models = [dict(scope=scope, lag=lag, model_id="persistence", metadata_path="model.json")
              for scope in ("temporal", "strict") for lag in (1, 2)]
    config = {"models": models, "protocol_id": "synthetic_recovery"}
    mapping = {"mapping": {"status": "persistence_raw", "reason": "fixed"},
               "budget": {"threshold": 1.000001}}
    mappings = {(m["scope"], m["lag"], m["model_id"]): mapping for m in models}

    def payload(scope):
        grids = np.array([0, 1] if scope == "temporal" else [1], dtype=np.int16)
        n = len(grids)
        return {"X1": np.zeros((n, 1)), "X2": np.ones((n, 1)),
                "day_index": np.zeros(n, dtype=np.int16), "grid_index": grids,
                "y": (grids % 2).astype(np.int8), "previous_y": (1 - grids % 2).astype(np.int8),
                "state1": (1 - grids % 2).astype(np.int8), "state2": (grids % 2).astype(np.int8)}

    reference_predictions, reference_scores = {}, []
    for m in models:
        data = payload(m["scope"])
        p = data[f"state{m['lag']}"].astype(float)
        arrays = recovery.enrich_prediction(data, p, m, mapping, config["protocol_id"])
        reference_predictions[recovery.prediction_key(2020, m)] = recovery.prediction_reference(arrays)
        for risk, mask in recovery.risk_masks(arrays["y"], arrays["evaluation_previous_y"]).items():
            if risk in recovery.RISK_SETS:
                for kind in ("raw", "calibrated"):
                    score = recovery.score_probability(arrays["y"][mask], p[mask], np.ones(mask.sum()))
                    reference_scores.append(dict(scope=m["scope"], lag=m["lag"], model_id="persistence", year=2020,
                                                 probability=kind, risk_set=risk, **score))
    pd.DataFrame(reference_scores).to_csv(repo / "scores.csv", index=False)
    manifest = {"scientific_files_sha256": {}, "raw_files": [{"name": "input.nc", "year": 2020,
                                "file_sha256": recovery.file_sha256(source_path),
                                "content_sha256": recovery.era5_content_digest(weather())}],
                "reference_predictions": reference_predictions}
    paths = {"grid_registry": "grid.csv", "neighbor_sources": "neighbors.json",
             "feature_registry": "features.json", "annual_scores": "scores.csv"}
    monkeypatch.setattr(recovery, "verify_manifest", lambda *args: (manifest, paths, config, mappings))
    monkeypatch.setattr(recovery, "complete_final_daily_night", lambda *args: None)
    monkeypatch.setattr(recovery, "assemble_phase4", lambda d, r, s, scope, stage: (payload(scope), {}, schema))
    monkeypatch.setattr(recovery.importlib.metadata, "version", lambda name: "test")
    original_files = {p: p.read_bytes() for p in repo.iterdir()}
    raw_before = source_path.read_bytes()
    args = Namespace(repo_root=repo, manifest="manifest.json", raw_dir=raw, output_dir=repo / ".reproduce" / "new",
                     years=[2020], models=None, save_features=False, verify_only=False)
    result = recovery.run(args)
    assert result["status"] == "completed" and result["scope"] == "partial_recovery"
    receipt = recovery.read_json(args.output_dir / "reproduction_receipt.json")
    assert len(receipt["prediction_checks"]) == 4
    assert str(tmp_path) not in (args.output_dir / "reproduction_receipt.json").read_text()
    for path, before in original_files.items():
        assert path.read_bytes() == before
    assert source_path.read_bytes() == raw_before
    with np.load(args.output_dir / "predictions/final_test/temporal/lag2/persistence/2020.npz") as z:
        np.testing.assert_array_equal(z["p_raw"], [0., 1.])
    with pytest.raises(FileExistsError):
        recovery.run(args)

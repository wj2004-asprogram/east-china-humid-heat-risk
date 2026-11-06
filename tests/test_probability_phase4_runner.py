"""Entire sealed pipeline exercised using generated arrays only, never ERA5."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.probability_phase4_data import check_gate, sha256
from tools import probability_phase4_calibrate as calibrate
from tools import probability_phase4_evaluate as evaluate


def synthetic_config(tmp_path, monkeypatch):
    config = json.loads(Path("configs/probability_phase4_freeze.json").read_text(encoding="utf-8-sig"))
    config["uncertainty"]["replicates"] = 20
    config["uncertainty"]["ci_minimum_valid_replicates"] = 20
    monkeypatch.chdir(tmp_path)
    path = Path("configs/probability_phase4_freeze.json")
    calibrate.write_json(path, config)
    protocol = Path("docs/PROBABILITY_PHASE4_PROTOCOL.md")
    protocol.parent.mkdir()
    protocol.write_text("Synthetic test protocol; no reserved data.")
    root = Path("synthetic")
    calibrate.write_json(root / "scientific_lock.json", {"status": "scientific_frozen",
        "freeze_config_sha256": sha256(path), "protocol_document_sha256": sha256(protocol),
        "source_code_sha256": {}, "artifacts_sha256": {}})
    return root, path, config


def prepare_predictions(root, config, stage):
    gates = check_gate(root, stage)
    index_hashes = {}
    for year in config["splits"][stage]:
        dictionary = root / "dictionaries" / str(year) / "grid_dictionary.csv"
        dictionary.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"grid_index": [0,1,2,3], "latitude": [29.,31.,29.,31.],
                      "longitude": [116.,119.,116.,119.], "spatial_role": ["train","train","holdout","holdout"]}).to_csv(dictionary,index=False)
        for model in config["models"]:
            grids = [0,1,2,3] if model["scope"] == "temporal" else [0,1] if stage == "calibration" else [2,3]
            day = np.repeat(np.arange(84,dtype=np.int16),len(grids))
            grid = np.tile(np.array(grids,dtype=np.int16),84)
            y = ((day + grid + year) % 3 == 0).astype(np.int8)
            previous = (grid % 2).astype(np.int8)
            p = (.1 + ((day + 2*grid) % 5)*.17).astype(float)
            if model["model_id"] == "persistence":
                p = previous.astype(float)
            path = root / "predictions" / stage / model["scope"] / f"lag{model['lag']}" / model["model_id"] / f"{year}.npz"
            path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,p_raw=p,y=y,evaluation_previous_y=previous,day_index=day,grid_index=grid)
            index = hashlib.sha256(day.tobytes()+grid.tobytes()).hexdigest()
            index_hashes[f"{model['scope']}/{year}"] = index
            calibrate.write_json(path.with_suffix(".json"),{"gates":gates,"output_sha256":sha256(path),"sample_index_sha256":index})
    return index_hashes


def test_fails_closed_without_scientific_and_second_seals(tmp_path):
    with pytest.raises(FileNotFoundError):
        calibrate.run(tmp_path, tmp_path / "absent.json")
    with pytest.raises(FileNotFoundError):
        evaluate.run(tmp_path, tmp_path / "absent.json")
    assert not (tmp_path / "evaluation").exists()


def test_full_synthetic_matrix_and_interruption_resume(tmp_path, monkeypatch):
    root, config_path, config = synthetic_config(tmp_path, monkeypatch)
    prepare_predictions(root, config, "calibration")
    original = calibrate.save_checkpoint
    counter = 0
    def interrupt_after_checkpoint(*args, **kwargs):
        nonlocal counter
        original(*args, **kwargs)
        counter += 1
        if counter == 2:
            raise KeyboardInterrupt("synthetic process interruption")
    monkeypatch.setattr(calibrate,"save_checkpoint",interrupt_after_checkpoint)
    with pytest.raises(KeyboardInterrupt):
        calibrate.run(root,config_path)
    checkpoint = calibrate.checkpoint_path(root / "calibration",config["models"][0])
    before = checkpoint.read_bytes()
    monkeypatch.setattr(calibrate,"save_checkpoint",original)
    result = calibrate.run(root,config_path)
    assert result["status"] == "completed" and result["count"] == 104
    assert checkpoint.read_bytes() == before
    records_path = root / "calibration" / "calibrators.json"
    records = json.loads(records_path.read_text())["models"]
    assert len(records) == 104
    assert sum(r["mapping"]["status"] == "persistence_raw" for r in records) == 4
    with pytest.raises(ValueError, match="Existing run"):
        calibrate.run(root,config_path)
    with pytest.raises(FileNotFoundError):
        evaluate.run(root,config_path)
    calibrate.write_json(root / "calibration_lock.json",{"status":"calibration_frozen",
        "scientific_lock_sha256":sha256(root / "scientific_lock.json"),
        "calibration_artifacts_sha256":{str(records_path):sha256(records_path)}})
    # Construct canonical final index seals from synthetic indices before writing predictions.
    indices = {}
    for scope in config["scopes"]:
        grids = [0,1,2,3] if scope == "temporal" else [2,3]
        day = np.repeat(np.arange(84,dtype=np.int16),len(grids))
        grid = np.tile(np.array(grids,dtype=np.int16),84)
        digest = hashlib.sha256(day.tobytes()+grid.tobytes()).hexdigest()
        for year in config["splits"]["final_test"]:
            indices[f"{scope}/{year}"] = digest
    calibrate.write_json(root / "test_open_lock.json",{"status":"test_open_frozen",
        "scientific_lock_sha256":sha256(root / "scientific_lock.json"),
        "calibration_lock_sha256":sha256(root / "calibration_lock.json"),
        "final_input_artifacts_sha256":{},"final_common_index_sha256":indices})
    assert prepare_predictions(root,config,"final_test") == indices
    counter = 0
    monkeypatch.setattr(evaluate,"save_checkpoint",interrupt_after_checkpoint)
    with pytest.raises(KeyboardInterrupt):
        evaluate.run(root,config_path)
    first = calibrate.checkpoint_path(root / "evaluation",config["models"][0])
    before = first.read_bytes()
    monkeypatch.setattr(evaluate,"save_checkpoint",original)
    result = evaluate.run(root,config_path)
    assert result["status"] == "completed" and result["model_year_bundles"] == 520
    assert first.read_bytes() == before
    scores = pd.read_csv(root / "evaluation" / "annual_scores.csv")
    assert len(scores) == 104*5*2*11
    assert set(scores.probability) == {"raw","calibrated"}
    intervals = pd.read_csv(root / "evaluation" / "bootstrap_intervals.csv")
    assert set(intervals.block_days) == {7,14,21}
    assert intervals.groupby("block_days").draw_sha256.nunique().eq(1).all()
    assert (intervals.roles == "calibration_change").any()
    assert set(intervals.weighting) == {"equal_grid","coslat"}
    assert len(json.loads((root / "evaluation" / "application_gate.json").read_text())) == 2
    with pytest.raises(ValueError, match="Existing run"):
        evaluate.run(root,config_path)
    ledger = (root / "access_ledger.jsonl").read_text()
    assert "one_formal_final_evaluation" in ledger
    assert "resume_completed_model_checkpoint" in ledger
    records_path.write_text("{}")
    with pytest.raises(ValueError, match="Gate hash mismatch"):
        evaluate.run(root,config_path)

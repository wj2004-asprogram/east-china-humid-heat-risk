"""Verify the completed model matrix and closed-year metadata without fitting."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path("data/probability_phase3/lightgbm/run1")
    config_path = Path("configs/probability_phase3_run.json")
    config = json.loads(config_path.read_text())
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    state = json.loads((root/"execution_registry.json").read_text())
    assert state["completed_or_fitted"] == 58
    assert state["exported_models"] == 44
    assert state["failed"] == state["interrupted"] == state["pending_total"] == 0
    expected = {(scope, lag, arm) for scope in config["input"]["scopes"]
                for lag in config["input"]["lags"] for arm in config["features"]["arms"]}
    found = set()
    selections = {scope: json.loads((root/f"selection_{scope}.json").read_text())["winner"]
                  for scope in config["input"]["scopes"]}
    for path in sorted((root/"models").rglob("*.json")):
        model = json.loads(path.read_text())
        key = (model["scope"], model["lag"], model["arm"])
        assert key not in found, key
        found.add(key)
        winner = selections[model["scope"]]
        assert model["run_config_sha256"] == config_hash, path
        assert model["prediction_years"] == list(range(2000, 2018)), path
        assert model["calibration_years_opened"] == model["final_years_opened"] == [], path
        assert model["prediction_num_iteration"] == winner["best_iteration"], path
        assert model["params"]["num_leaves"] == winner["num_leaves"], path
        assert model["params"]["min_data_in_leaf"] == winner["min_child_samples"], path
        assert model["params"]["lambda_l2"] == winner["reg_lambda"], path
        assert hashlib.sha256(path.with_suffix(".txt").read_bytes()).hexdigest() == model["exported_model_sha256"], path
    assert found == expected, (found-expected, expected-found)
    result = {"status": "passed", "models_checked": len(found), "formal_fits_completed": 58,
              "run_config_sha256": config_hash,
              "checks": ["complete scope/lag/arm matrix", "unique model metadata",
                         "per-scope winner parameters and iteration reuse", "exported model byte hashes",
                         "only 2000-2017 prediction years; no calibration/final years opened"],
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "limits": "Metadata and file-integrity verification; earlier numerical replay and scientific review are reused."}
    target = Path("data/probability_phase3/completion_verification.json")
    target.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

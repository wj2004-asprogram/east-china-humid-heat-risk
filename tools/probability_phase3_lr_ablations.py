"""Complete the eight prespecified LR ablations, without parameter search.

Only 2000-2017 cached arrays are opened. The full F4 training-only scaler and
selected C are restored from phase 2; existing F0/F1/F4 results are referenced.
"""

import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import warnings

import numpy as np
import pandas as pd

from tools.probability_phase2_baselines import year_data, train_matrix, write_prediction, save_tables
from tools.probability_phase2_features import dump, sha256, peak_memory_bytes

ARMS = ("F2", "F3", "F0P", "F1P", "F2_minus_NT", "F2_minus_NM", "F3_minus_RT", "F3_minus_RM")


def arm_columns(names, groups, arms):
    """Keep registry order; P contains exactly y_anchor, not the H group."""
    if len(names) != len(groups) or len(set(names)) != len(names):
        raise ValueError("Invalid feature registry")
    if names.count("y_anchor") != 1 or groups[names.index("y_anchor")] != "H":
        raise ValueError("Expected one y_anchor in H")
    result = {}
    for arm, wanted in arms.items():
        if set(wanted) - (set(groups) | {"P"}):
            raise ValueError("Unknown feature group")
        result[arm] = np.array([i for i, (name, group) in enumerate(zip(names, groups))
                                if group in wanted or ("P" in wanted and name == "y_anchor")], dtype=int)
    return result


def restore_scaler(state, names, scope, lag, training_rows):
    from sklearn.preprocessing import StandardScaler
    if (state["features"] != names or state["scope"] != scope or state["lag"] != lag
            or state["train_years"] != list(range(2000, 2015))
            or state["training_rows"] != training_rows or state["C"] != .1):
        raise ValueError("Phase 2 scaler provenance mismatch")
    scaler = StandardScaler(copy=False)
    scaler.mean_ = np.asarray(state["scaler_mean"], dtype=float)
    scaler.scale_ = np.asarray(state["scaler_scale"], dtype=float)
    scaler.n_features_in_ = len(names)
    if (scaler.mean_.shape != (len(names),) or scaler.scale_.shape != (len(names),)
            or not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all()
            or np.any(scaler.scale_ <= 0)):
        raise ValueError("Invalid restored scaler")
    return scaler


def fit_candidate(x, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits
    started = time.perf_counter()
    model = LogisticRegression(C=.1, solver="lbfgs", l1_ratio=0., max_iter=2000,
                               tol=1e-6, fit_intercept=True, class_weight=None, random_state=20261105)
    with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=2):
        warnings.simplefilter("always")
        model.fit(x, y)
    failed = any(issubclass(w.category, ConvergenceWarning) for w in captured)
    record = {"C": .1, "converged": not failed, "iterations": int(model.n_iter_[0]),
              "runtime_seconds": time.perf_counter()-started,
              "warnings": [str(w.message) for w in captured],
              "peak_working_set_bytes": peak_memory_bytes()}
    return (None if failed else model), record


def main(argv=None):
    from threadpoolctl import threadpool_limits, threadpool_info
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=Path("data/probability_phase2/features"))
    parser.add_argument("--phase2-dir", type=Path, default=Path("data/probability_phase2/experiments/run1"))
    parser.add_argument("--registry", type=Path, default=Path("data/probability_phase1/samples/grid_registry.csv"))
    parser.add_argument("--protocol", type=Path, default=Path("configs/probability_protocol_v1.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("output-dir must not exist")
    started = time.perf_counter()
    registry = pd.read_csv(args.registry)
    feature_registry = json.loads((args.feature_dir/"2000/feature_registry.json").read_text())
    names, groups = feature_registry["names"], feature_registry["groups"]
    protocol = json.loads(args.protocol.read_text())
    columns = arm_columns(names, groups, protocol["features"]["arms"])
    if len(names) != 117:
        raise ValueError("Expected frozen 117 columns")
    inputs = {str(args.registry): sha256(args.registry), str(args.protocol): sha256(args.protocol)}
    for year in range(2000, 2018):
        annual = args.feature_dir/str(year)
        run = json.loads((annual/"run.json").read_text())
        if json.loads((annual/"feature_registry.json").read_text()) != feature_registry:
            raise ValueError("Annual registry mismatch")
        for scope in ("temporal", "strict"):
            path = annual/f"{scope}.npz"
            inputs[str(path)] = sha256(path)
            if inputs[str(path)] != run["outputs"][path.name]["sha256"]:
                raise ValueError("Feature hash mismatch")
    args.output_dir.mkdir(parents=True)
    dump(args.output_dir/"design_lock.json", {
        "created_before_fitting": True, "arms": list(ARMS), "selected_C": .1,
        "solver": "lbfgs", "penalty": "l2", "max_iter": 2000, "tol": 1e-6, "threads": 2,
        "train_years": list(range(2000, 2015)), "validation_years": [2015, 2016, 2017],
        "columns": {a: columns[a].tolist() for a in columns},
        "features": {a: [names[i] for i in columns[a]] for a in columns},
        "P": "only y_anchor; no other H columns", "search": None,
        "source_sha256": {str(p): sha256(p) for p in (Path(__file__), Path("tools/probability_phase2_baselines.py"), Path("src/probability_complete_features.py"), Path("src/probability_metrics.py"))}})
    schema_path = args.phase2_dir/"prediction_schema.json"
    inputs[str(schema_path)] = sha256(schema_path)
    dump(args.output_dir/"prediction_schema.json", json.loads(schema_path.read_text()))
    dump(args.output_dir/"threadpools.json", threadpool_info())
    records, reliability, experiments, reused = [], [], [], []
    for scope in ("temporal", "strict"):
        selection_path = args.phase2_dir/f"lr_selection_{scope}.json"
        inputs[str(selection_path)] = sha256(selection_path)
        if json.loads(selection_path.read_text())["selected_C"] != .1:
            raise ValueError("Unexpected previously selected C")
        yearly = {year: year_data(args.feature_dir, scope, year, registry) for year in range(2000, 2018)}
        expected = 132720 if scope == "temporal" else 109116
        for data in yearly.values():
            if len(data["y"]) != expected or (scope == "strict" and not data["frame"].spatial_role.eq("train").all()):
                raise ValueError("Common sample or strict training-space mismatch")
        y_train = np.concatenate([yearly[y]["y"] for y in range(2000, 2015)])
        for lag in (1, 2):
            scaler_path = args.phase2_dir/"models"/scope/f"lag{lag}_logistic_F4.json"
            inputs[str(scaler_path)] = sha256(scaler_path)
            scaler = restore_scaler(json.loads(scaler_path.read_text()), names, scope, lag, len(y_train))
            matrix = scaler.transform(train_matrix(args.feature_dir, scope, lag, yearly))
            if not np.isfinite(matrix).all():
                raise ValueError("Nonfinite standardized matrix")
            models = {}
            for arm in ARMS:
                print(json.dumps({"stage": "fit_start", "scope": scope, "lag": lag, "arm": arm}), flush=True)
                cols = columns[arm]
                subset = np.ascontiguousarray(matrix[:, cols])
                model, fit = fit_candidate(subset, y_train)
                del subset
                state = {"kind": "standardized_l2_logistic", "scope": scope, "lag": lag, "arm": arm,
                         "features": [names[i] for i in cols], "columns": cols.tolist(),
                         "training_rows": len(y_train), "validation_rows": 3*expected,
                         "train_years": list(range(2000, 2015)), "C": .1, "solver": "lbfgs",
                         "tol": 1e-6, "max_iter": 2000, "threads": 2, "class_weight": None,
                         "intercept_penalized": False, "scaler_source": str(scaler_path),
                         "scaler_source_sha256": inputs[str(scaler_path)],
                         "scaler_mean": scaler.mean_[cols].tolist(), "scaler_scale": scaler.scale_[cols].tolist(),
                         "fit": fit, "status": "complete" if model is not None else "convergence_failure"}
                if model is not None:
                    state.update(coefficient=model.coef_[0].tolist(), intercept=float(model.intercept_[0]))
                    models[arm] = model
                path = args.output_dir/"models"/scope/f"lag{lag}_logistic_{arm}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                dump(path, state)
                experiments.append({"scope": scope, "lag": lag, "model": f"logistic_{arm}",
                                    "state": str(path.relative_to(args.output_dir)), "status": state["status"],
                                    "training_rows": len(y_train), "validation_rows": 3*expected, **fit})
                dump(args.output_dir/"experiments.json", experiments)
                print(json.dumps({"stage": "fit_end", "scope": scope, "lag": lag, "arm": arm, **fit}), flush=True)
            del matrix
            gc.collect()
            for year, data in yearly.items():
                with np.load(args.feature_dir/str(year)/f"{scope}.npz", allow_pickle=False) as source:
                    x = scaler.transform(source[f"X{lag}"].astype(np.float64))
                with threadpool_limits(limits=2):
                    for arm, model in models.items():
                        probability = model.predict_proba(x[:, columns[arm]])[:, 1]
                        write_prediction(args.output_dir, scope, lag, f"logistic_{arm}", year, data, probability, records, reliability)
            save_tables(args.output_dir, records, reliability)
            for arm in ("F0", "F1", "F4"):
                path = args.phase2_dir/"models"/scope/f"lag{lag}_logistic_{arm}.json"
                reused.append({"scope": scope, "lag": lag, "arm": arm, "state": str(path), "sha256": sha256(path),
                               "predictions": str(args.phase2_dir/"predictions"/scope/f"lag{lag}"/f"logistic_{arm}")})
            del x, models, scaler
            gc.collect()
    dump(args.output_dir/"reused_phase2_models.json", reused)
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl", "joblib"]
    report = {"command": sys.argv, "python": platform.python_version(), "executable": sys.executable,
              "packages": {p: importlib.metadata.version(p) for p in packages},
              "input_sha256": inputs, "runtime_seconds": time.perf_counter()-started,
              "peak_working_set_bytes": peak_memory_bytes(), "fits": len(experiments),
              "successful_fits": sum(e["status"] == "complete" for e in experiments),
              "failed_fits": sum(e["status"] != "complete" for e in experiments),
              "calibration_years_opened": [], "final_years_opened": [],
              "strict_validation_space": "train only; no spatial-transfer inference",
              "training_metrics": "resubstitution", "probabilities": "raw; fixed research threshold 0.5"}
    dump(args.output_dir/"run.json", report)
    dump(args.output_dir/"output_manifest.json", {str(p.relative_to(args.output_dir)): {"sha256": sha256(p), "bytes": p.stat().st_size}
         for p in args.output_dir.rglob("*") if p.is_file()})
    print(json.dumps({"stage": "complete", **report}), flush=True)


if __name__ == "__main__":
    main()

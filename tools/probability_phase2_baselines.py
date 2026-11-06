"""Train frozen raw baselines on 2000-2014; report 2000-2017 only.

The finite LR search is F4/lag1 only, independently in temporal and strict scope.
Chosen C is reused for F0, F1 and lag2. No calibration or final-year file is read.
"""

import argparse
import copy
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

from src.probability_baselines import ClimatologyBaseline, TransitionBaseline, persistence_probabilities
from src.probability_metrics import evaluate_year, reliability_counts
from tools.probability_phase2_features import dump, sha256, peak_memory_bytes


def year_data(root, scope, year, registry):
    if year not in range(2000, 2018):
        raise ValueError("Model runner may read only training/validation feature files")
    with np.load(root/str(year)/f"{scope}.npz", allow_pickle=False) as source:
        data = {key: source[key] for key in ("day_index", "grid_index", "y", "previous_y", "state1", "state2")}
    grid = registry.iloc[data["grid_index"]].reset_index(drop=True)
    dates = pd.Timestamp(year, 6, 9)+pd.to_timedelta(data["day_index"], unit="D")
    frame = pd.DataFrame({"date": dates, "grid_id": grid.grid_id, "spatial_role": grid.spatial_role,
                          "label": data["y"], "state_lag1": data["state1"], "state_lag2": data["state2"],
                          "state_date_lag1": dates-pd.Timedelta(days=1),
                          "state_date_lag2": dates-pd.Timedelta(days=2)})
    data["frame"], data["latitude"] = frame, grid.latitude.to_numpy()
    return data


def write_prediction(output, scope, lag, model_id, year, data, probability, records, reliability):
    probability = np.asarray(probability, dtype=np.float64)
    path = output/"predictions"/scope/f"lag{lag}"/model_id
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path/f"{year}.npz", p_raw=probability,
                        day_index=data["day_index"], grid_index=data["grid_index"],
                        y=data["y"], evaluation_previous_y=data["previous_y"])
    base = {"scope": scope, "lag": lag, "model": model_id, "year": year,
            "split": "train_resubstitution" if year <= 2014 else "validation"}
    for risk_set, metrics in evaluate_year(data["y"], data["previous_y"], probability, data["latitude"]).items():
        records.append(dict(base, risk_set=risk_set, **metrics))
    for row in reliability_counts(data["y"], probability):
        reliability.append(dict(base, **row))


def save_tables(output, records, reliability):
    frame = pd.DataFrame(records)
    frame.to_csv(output/"metrics_yearly.csv", index=False)
    pd.DataFrame(reliability).to_csv(output/"reliability_yearly.csv", index=False)
    summary = frame.groupby(["scope", "lag", "model", "split", "risk_set"], sort=True).agg(
        years=("year", "count"), n=("n", "sum"), events=("events", "sum"),
        mean_annual_brier=("brier", "mean"), mean_annual_ap=("ap", "mean"),
        mean_annual_prevalence=("prevalence", "mean"), mean_annual_brier_coslat=("brier_coslat", "mean")).reset_index()
    summary.to_csv(output/"metrics_summary.csv", index=False)


def fit_simple(args, scope, registry, records, reliability, experiments):
    yearly = {year: year_data(args.feature_dir, scope, year, registry) for year in range(2000, 2018)}
    train = pd.concat([yearly[year]["frame"] for year in range(2000, 2015)], ignore_index=True)
    for lag in (1, 2):
        started = time.perf_counter()
        seasonal = ClimatologyBaseline(cutoff_lag_days=lag, strict_spatial=scope == "strict").fit(train)
        pooled = copy.deepcopy(seasonal)
        pooled.state["grid_counts"] = []
        pooled.state["prediction_mode"] = "pooled_training_region"
        transition = TransitionBaseline(cutoff_lag_days=lag, strict_spatial=scope == "strict").fit(train)
        models = {"persistence": None, "seasonal": seasonal, "seasonal_pooled": pooled, "transition": transition}
        for name, model in models.items():
            model_path = args.output_dir/"models"/scope/f"lag{lag}_{name}.json"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            state = model.to_state() if model else {"kind": "persistence", "cutoff_lag_days": lag,
                                                    "fit": False, "probability": f"y(D-{lag})"}
            dump(model_path, state)
            for year, data in yearly.items():
                probability = (persistence_probabilities(data["frame"], cutoff_lag_days=lag) if model is None else
                               model.predict(data["frame"], cutoff_lag_days=lag))
                write_prediction(args.output_dir, scope, lag, name, year, data, probability, records, reliability)
            experiments.append({"scope": scope, "lag": lag, "model": name, "training_rows": len(train),
                                "state": str(model_path.relative_to(args.output_dir)), "status": "complete"})
        save_tables(args.output_dir, records, reliability)
        print(json.dumps({"stage": "simple_baselines", "scope": scope, "lag": lag,
                          "runtime_seconds": time.perf_counter()-started}), flush=True)
    del train, yearly
    gc.collect()


def train_matrix(root, scope, lag, yearly):
    count = sum(len(yearly[y]["y"]) for y in range(2000, 2015))
    result = np.empty((count, 117), dtype=np.float64)
    offset = 0
    for year in range(2000, 2015):
        with np.load(root/str(year)/f"{scope}.npz", allow_pickle=False) as source:
            values = source[f"X{lag}"]
            result[offset:offset+len(values)] = values
            offset += len(values)
    return result


def fit_lr(args, scope, registry, feature_registry, records, reliability, experiments):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits, threadpool_info

    yearly = {year: year_data(args.feature_dir, scope, year, registry) for year in range(2000, 2018)}
    y_train = np.concatenate([yearly[year]["y"] for year in range(2000, 2015)])
    groups = np.array(feature_registry["groups"])
    columns = {"F0": np.flatnonzero(np.isin(groups, ["C", "T"])),
               "F1": np.flatnonzero(np.isin(groups, ["C", "T", "M"])), "F4": np.arange(117)}
    selection, chosen_c = [], None
    dump(args.output_dir/f"threadpools_{scope}.json", threadpool_info())

    def fit_candidate(x, c):
        started = time.perf_counter()
        model = LogisticRegression(C=c, solver="lbfgs", l1_ratio=0., max_iter=2000,
                                   tol=1e-6, fit_intercept=True, class_weight=None, random_state=20261105)
        with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=2):
            warnings.simplefilter("always")
            model.fit(x, y_train)
        failed = any(issubclass(w.category, ConvergenceWarning) for w in captured)
        return model, {"C": c, "converged": not failed, "iterations": int(model.n_iter_[0]),
                       "runtime_seconds": time.perf_counter()-started,
                       "warnings": [str(w.message) for w in captured]}

    for lag in (1, 2):
        print(json.dumps({"stage": "loading_lr", "scope": scope, "lag": lag}), flush=True)
        matrix = train_matrix(args.feature_dir, scope, lag, yearly)
        scaler = StandardScaler(copy=False)
        matrix = scaler.fit_transform(matrix)
        if not np.isfinite(matrix).all():
            raise ValueError("Nonfinite standardized training matrix")
        validation = {}
        for year in range(2015, 2018):
            with np.load(args.feature_dir/str(year)/f"{scope}.npz", allow_pickle=False) as source:
                validation[year] = scaler.transform(source[f"X{lag}"].astype(np.float64))
        models, fit_records = {}, {}
        if lag == 1:
            best_score = float("inf")
            for c in (.01, .1, 1., 10.):
                model, fit_record = fit_candidate(matrix, c)
                if fit_record["converged"]:
                    with threadpool_limits(limits=2):
                        scores = {year: float(np.mean((model.predict_proba(x)[:, 1]-yearly[year]["y"])**2))
                                  for year, x in validation.items()}
                    score = float(np.mean(list(scores.values())))
                    fit_record.update(annual_validation_brier=scores, mean_annual_validation_brier=score)
                    if score < best_score:
                        best_score, chosen_c = score, c
                        models["F4"], fit_records["F4"] = model, fit_record
                selection.append(fit_record)
                dump(args.output_dir/f"lr_selection_{scope}.json", {"scope": scope, "search_arm": "F4", "search_lag": 1,
                     "solver": "lbfgs", "max_iter": 2000, "tol": 1e-6, "threads": 2,
                     "selection": "mean annual raw Brier over 2015,2016,2017; smaller C wins exact ties",
                     "candidates": selection, "selected_C": chosen_c,
                     "reuse": "F0,F1,F4 and both lags within this scope"})
                print(json.dumps({"stage": "lr_candidate", "scope": scope, **fit_record}), flush=True)
            if chosen_c is None:
                raise RuntimeError("All frozen LR candidates failed convergence")
        for arm in ("F0", "F1", "F4"):
            if arm in models:
                continue
            subset = matrix if arm == "F4" else np.ascontiguousarray(matrix[:, columns[arm]])
            model, fit_record = fit_candidate(subset, chosen_c)
            if subset is not matrix:
                del subset
            if not fit_record["converged"]:
                experiments.append({"scope": scope, "lag": lag, "model": f"logistic_{arm}",
                                    "status": "convergence_failure", **fit_record})
            else:
                models[arm], fit_records[arm] = model, fit_record
            print(json.dumps({"stage": "lr_arm", "scope": scope, "lag": lag, "arm": arm, **fit_record}), flush=True)
        del matrix
        gc.collect()
        for arm, model in models.items():
            path = args.output_dir/"models"/scope/f"lag{lag}_logistic_{arm}.json"
            cols = columns[arm]
            state = {"kind": "standardized_l2_logistic", "scope": scope, "lag": lag, "arm": arm,
                     "features": [feature_registry["names"][i] for i in cols],
                     "train_years": list(range(2000, 2015)), "training_rows": len(y_train),
                     "C": chosen_c, "solver": "lbfgs", "tol": 1e-6, "max_iter": 2000,
                     "class_weight": None, "intercept_penalized": False,
                     "scaler_mean": scaler.mean_[cols].tolist(), "scaler_scale": scaler.scale_[cols].tolist(),
                     "coefficient": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]),
                     "fit": fit_records[arm]}
            dump(path, state)
            experiments.append({"scope": scope, "lag": lag, "model": f"logistic_{arm}", "status": "complete",
                                "state": str(path.relative_to(args.output_dir)), **fit_records[arm]})
        for year, data in yearly.items():
            if year in validation:
                x = validation[year]
            else:
                with np.load(args.feature_dir/str(year)/f"{scope}.npz", allow_pickle=False) as source:
                    x = scaler.transform(source[f"X{lag}"].astype(np.float64))
            with threadpool_limits(limits=2):
                for arm, model in models.items():
                    probability = model.predict_proba(x[:, columns[arm]])[:, 1]
                    write_prediction(args.output_dir, scope, lag, f"logistic_{arm}", year, data, probability, records, reliability)
        save_tables(args.output_dir, records, reliability)
        dump(args.output_dir/"experiments.json", experiments)
        print(json.dumps({"stage": "lr_lag_complete", "scope": scope, "lag": lag,
                          "peak_working_set_bytes": peak_memory_bytes()}), flush=True)
        del validation, x, models, scaler
        gc.collect()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=Path("data/probability_phase2/features"))
    parser.add_argument("--registry", type=Path, default=Path("data/probability_phase1/samples/grid_registry.csv"))
    parser.add_argument("--protocol", type=Path, default=Path("configs/probability_protocol_v1.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scopes", nargs="+", choices=["temporal", "strict"], default=["temporal", "strict"])
    parser.add_argument("--skip-lr", action="store_true")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("output-dir must not exist")
    registry = pd.read_csv(args.registry)
    feature_registry = json.loads((args.feature_dir/"2000/feature_registry.json").read_text())
    if len(feature_registry["names"]) != 117:
        raise ValueError("Frozen feature registry mismatch")
    inputs = {}
    for year in range(2000, 2018):
        path = args.feature_dir/str(year)/"run.json"
        run = json.loads(path.read_text())
        for scope in args.scopes:
            array = args.feature_dir/str(year)/f"{scope}.npz"
            if sha256(array) != run["outputs"][array.name]["sha256"]:
                raise ValueError("Annual feature hash mismatch")
        inputs[str(path)] = sha256(path)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    records, reliability, experiments = [], [], []
    dump(args.output_dir/"design_lock.json", {
        "created_before_fitting": True, "solver": "lbfgs", "penalty": "l2",
        "intercept_penalized": False, "C_candidates": [.01, .1, 1., 10.],
        "max_iter": 2000, "tol": 1e-6, "threads": 2, "random_seed": 20261105,
        "search": "F4 lag1 independently for temporal and strict; reuse chosen C for both lags and all arms",
        "train_years": list(range(2000, 2015)), "validation_years": [2015, 2016, 2017],
        "source_sha256": {str(path): sha256(path) for path in (
            Path("src/probability_complete_features.py"), Path("src/probability_baselines.py"),
            Path("src/probability_metrics.py"), Path(__file__))}})
    dump(args.output_dir/"prediction_schema.json", {
        "representation": "Factorized annual NPZ predictions and immutable grid registry",
        "grid_registry": str(args.registry), "grid_registry_sha256": sha256(args.registry),
        "row_key": "grid_id[grid_index] + '|' + (year-June09 + day_index)",
        "date_rule": "day_index 0 = June9, 83 = August31",
        "issue_target_start": "target date 00 BJT = preceding date 16 UTC",
        "target_end_exclusive": "next date 00 BJT = target date 16 UTC",
        "max_input_valid_time": "D-lag 23 BJT = D-lag 15 UTC",
        "spatial_role_latitude_longitude": "join exact registry row grid_index",
        "model_id_lag_scope_year": "directory and filename",
        "p_calibrated": None, "raw_research_alert_threshold": .5,
        "sample_eligibility": "same finite 117-feature union for both lags within each scope",
        "previous_y": "evaluation only, never a D2 model input",
        "protocol_sha256": sha256(args.protocol)})
    for scope in args.scopes:
        fit_simple(args, scope, registry, records, reliability, experiments)
        dump(args.output_dir/"experiments.json", experiments)
        if not args.skip_lr:
            fit_lr(args, scope, registry, feature_registry, records, reliability, experiments)
    save_tables(args.output_dir, records, reliability)
    dump(args.output_dir/"experiments.json", experiments)
    packages = ["numpy", "pandas", "scipy"] + ([] if args.skip_lr else ["scikit-learn", "joblib", "threadpoolctl", "cloudpickle", "narwhals"])
    report = {"command": sys.argv, "python": platform.python_version(),
              "packages": {name: importlib.metadata.version(name) for name in packages},
              "protocol_sha256": sha256(args.protocol), "annual_input_run_sha256": inputs,
              "runtime_seconds": time.perf_counter()-started, "peak_working_set_bytes": peak_memory_bytes(),
              "train_years": list(range(2000, 2015)), "validation_years": [2015, 2016, 2017],
              "calibration_years_opened": [], "final_years_opened": [],
              "training_metrics": "resubstitution, not out-of-sample skill",
              "strict_validation_space": "training-space eligible queries only, not evidence of spatial transfer",
              "uncertainty": "no final-test bootstrap or calibrated decision claims in this stage"}
    dump(args.output_dir/"run.json", report)
    outputs = {str(path.relative_to(args.output_dir)): {"sha256": sha256(path), "bytes": path.stat().st_size}
               for path in args.output_dir.rglob("*") if path.is_file()}
    dump(args.output_dir/"output_manifest.json", outputs)
    print(json.dumps({"stage": "complete", **report}), flush=True)


if __name__ == "__main__":
    main()

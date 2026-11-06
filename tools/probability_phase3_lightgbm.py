"""Resumable frozen LightGBM search and ablations using phase-2 common samples."""

import argparse
import gc
import importlib.metadata
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from src.probability_phase3_lightgbm import arm_columns, load_xy, annual_brier, select_candidate
from tools.probability_phase2_features import dump, sha256, peak_memory_bytes
from tools.probability_phase2_baselines import year_data, write_prediction, save_tables


def fixed_params():
    return {"objective": "binary", "metric": "binary_logloss", "learning_rate": .05,
            "num_threads": 1, "deterministic": True, "seed": 20261105,
            "force_col_wise": True, "force_row_wise": False, "max_bin": 255,
            "min_data_in_bin": 3, "bin_construct_sample_cnt": 200000,
            "data_random_seed": 20261105, "feature_pre_filter": False,
            "max_depth": -1, "feature_fraction": 1., "bagging_fraction": 1.,
            "bagging_freq": 0, "verbosity": -1}


def pilot(args):
    import lightgbm as lgb
    if args.output_dir.exists():
        raise ValueError("Pilot output directory must be new")
    started = time.perf_counter()
    registry = json.loads((args.feature_dir/"2000/feature_registry.json").read_text())
    x, y, counts = load_xy(args.feature_dir, "temporal", 1, range(2000, 2015), np.arange(117))
    loaded = time.perf_counter()
    params = dict(fixed_params(), num_leaves=7, min_data_in_leaf=100, lambda_l2=1.)
    dataset = lgb.Dataset(x, label=y, params=params, feature_name=registry["names"],
                          categorical_feature=[], free_raw_data=True)
    dataset.construct()
    del x
    gc.collect()
    constructed = time.perf_counter()
    model = lgb.train(params, dataset, num_boost_round=100)
    finished = time.perf_counter()
    report = {"purpose": "training_only_resource_pilot_not_a_search_candidate", "formal_fit_count": 0,
              "resource_pilot_fits": 1, "lightgbm": importlib.metadata.version("lightgbm"),
              "params": params, "scope": "temporal", "lag": 1, "arm": "F4",
              "train_years": list(counts), "training_rows": len(y), "features": 117,
              "feature_dtype": "float32", "validation_data_opened": False,
              "rounds": model.current_iteration(), "loading_seconds": loaded-started,
              "dataset_seconds": constructed-loaded, "fit_seconds": finished-constructed,
              "total_seconds": finished-started, "peak_working_set_bytes": peak_memory_bytes(),
              "linear_projection_58_times_2000_round_seconds": (finished-constructed)*20*58,
              "projection_caveat": "Not a promise: 15-leaf trees can cost more, early stopping and narrower arms can cost less; dataset/loading overhead excluded",
              "predictions_retained": False, "command": sys.argv}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dump(args.output_dir/"resource_pilot.json", report)
    print(json.dumps(report), flush=True)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def heartbeat(label):
    start = time.perf_counter()
    def callback(env):
        if (env.iteration+1) % 100 == 0:
            print(json.dumps({"stage": "boosting", "fit": label, "round": env.iteration+1,
                              "elapsed_seconds": time.perf_counter()-start}), flush=True)
    callback.order = 15
    return callback


def candidate_grid(config):
    grid = config["lightgbm"]["candidate_grid"]
    return [{"num_leaves": leaf, "min_child_samples": child, "reg_lambda": lam}
            for leaf, child, lam in itertools.product(grid["num_leaves"], grid["min_data_in_leaf"], grid["lambda_l2"])]


def candidate_id(candidate):
    return f"leaves{candidate['num_leaves']}_child{candidate['min_child_samples']}_lambda{candidate['reg_lambda']}"


def native_params(config, candidate):
    return dict(config["lightgbm"]["params"], num_leaves=candidate["num_leaves"],
                min_data_in_leaf=candidate["min_child_samples"], lambda_l2=candidate["reg_lambda"])


def checked_record(path, binding):
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if record.get("fingerprint") != fingerprint(binding):
        raise ValueError(f"Resume configuration differs: {path}")
    return record


def may_start_fit(args):
    """Resource-only deadline; never truncate a fit that has already started."""
    if not args.batch_plan.exists():
        return True
    plan = json.loads(args.batch_plan.read_text())["lightgbm"]
    if time.time() < plan["deadline_unix"]:
        return True
    record = {"status": "resource_budget_reached", "batch_plan": str(args.batch_plan),
              "deadline_unix": plan["deadline_unix"], "observed_unix": time.time(),
              "scientific_grid_unchanged": True, "pending_not_failed": True}
    if not (args.output_dir/"resource_stop.json").exists():
        dump(args.output_dir/"resource_stop.json", record)
    print(json.dumps(record), flush=True)
    return False


def fit_or_restore(directory, binding, params, dataset, *, rounds, validation=None, stop=None):
    """Save fitted text/state before any potentially interrupted prediction work."""
    import lightgbm as lgb
    directory.mkdir(parents=True, exist_ok=True)
    trained = checked_record(directory/"trained.json", binding)
    if trained:
        if sha256(directory/"model.txt") != trained["model_sha256"]:
            raise ValueError("Saved model hash mismatch")
        return lgb.Booster(model_file=str(directory/"model.txt")), trained
    history = {}
    callbacks = [heartbeat(str(directory))]
    if stop:
        callbacks += [lgb.early_stopping(stop["stopping_rounds"], first_metric_only=stop["first_metric_only"],
                                         verbose=False, min_delta=stop["min_delta"]), lgb.record_evaluation(history)]
    started = time.perf_counter()
    print(json.dumps({"stage": "fit_start", "fit": str(directory), "rounds_requested": rounds}), flush=True)
    model = lgb.train(params, dataset, num_boost_round=rounds,
                      valid_sets=[validation] if validation is not None else None,
                      valid_names=["validation"] if validation is not None else None, callbacks=callbacks)
    best = int(model.best_iteration or model.current_iteration())
    if best <= 0:
        raise lgb.basic.LightGBMError("Nonpositive best iteration")
    prediction_rounds = min(best if stop else rounds, model.current_iteration())
    model.save_model(str(directory/"model.txt"), num_iteration=prediction_rounds)
    record = dict(binding, fingerprint=fingerprint(binding), params=params,
                  requested_iterations=rounds, best_iteration=best,
                  actual_model_iterations=model.current_iteration(), prediction_num_iteration=prediction_rounds,
                  optimization_rounds=len(history.get("validation", {}).get("binary_logloss", [])) if stop else model.current_iteration(),
                  fit_seconds=time.perf_counter()-started, model_sha256=sha256(directory/"model.txt"),
                  peak_working_set_bytes=peak_memory_bytes(), status="fitted",
                  lightgbm=importlib.metadata.version("lightgbm"))
    if history:
        dump(directory/"learning_curve.json", history)
    dump(directory/"trained.json", record)
    print(json.dumps({"stage": "fit_saved", "fit": str(directory), "best_iteration": best,
                      "actual_model_iterations": record["actual_model_iterations"], "fit_seconds": record["fit_seconds"]}), flush=True)
    return model, record


def search_scope(args, config, registry, scope, context):
    import lightgbm as lgb
    candidates = candidate_grid(config)
    directory = args.output_dir/"search"/scope
    records = []
    for candidate in candidates:
        binding = dict(context, scope=scope, lag=1, arm="F4", operation="search", **candidate)
        done = checked_record(directory/candidate_id(candidate)/"result.json", binding)
        if done:
            records.append(done)
    if len(records) != len(candidates):
        params = config["lightgbm"]["params"]
        x, y, train_counts = load_xy(args.feature_dir, scope, 1, config["input"]["train_years"], np.arange(117))
        train_set = lgb.Dataset(x, label=y, params=params, feature_name=registry["names"], categorical_feature=[], free_raw_data=True)
        train_set.construct()
        del x
        gc.collect()
        vx, vy, valid_counts = load_xy(args.feature_dir, scope, 1, config["input"]["validation_years"], np.arange(117))
        if len(set(valid_counts.values())) != 1:
            raise ValueError("Pooled early stopping requires equal annual validation counts")
        valid_set = lgb.Dataset(vx, label=vy, reference=train_set, params=params,
                                feature_name=registry["names"], categorical_feature=[], free_raw_data=True)
        valid_set.construct()
        records = []
        for candidate in candidates:
            fit_dir = directory/candidate_id(candidate)
            binding = dict(context, scope=scope, lag=1, arm="F4", operation="search", **candidate)
            done = checked_record(fit_dir/"result.json", binding)
            if done:
                records.append(done)
                continue
            if not (fit_dir/"trained.json").exists() and not may_start_fit(args):
                break
            try:
                model, trained = fit_or_restore(fit_dir, binding, native_params(config, candidate), train_set,
                    rounds=config["lightgbm"]["search"]["max_boost_rounds"], validation=valid_set,
                    stop=config["lightgbm"]["search"]["early_stopping"])
                probability = model.predict(vx, num_iteration=trained["prediction_num_iteration"], num_threads=1)
                scores = annual_brier(vy, probability, valid_counts)
                np.savez_compressed(fit_dir/"validation_probabilities.npz", p_raw=probability, y=vy,
                                    years=np.asarray(list(valid_counts)), annual_counts=np.asarray(list(valid_counts.values())))
                done = dict(trained, status="complete", annual_validation_brier=scores,
                            mean_annual_validation_brier=float(np.mean(list(scores.values()))),
                            train_rows=len(y), validation_rows=len(vy), annual_validation_counts=valid_counts,
                            dataset_bins_from="training_only", validation_reference="matching_training_dataset",
                            exact_features=registry["names"])
            except lgb.basic.LightGBMError as error:
                done = dict(binding, fingerprint=fingerprint(binding), status="failed", failure=str(error))
            dump(fit_dir/"result.json", done)
            records.append(done)
            print(json.dumps({"stage": "candidate_complete", "scope": scope, "candidate": candidate,
                              "status": done["status"], "best_iteration": done.get("best_iteration"),
                              "mean_annual_validation_brier": done.get("mean_annual_validation_brier")}), flush=True)
            update_registry(args, config)
        del train_set, valid_set, vx, vy, y
        gc.collect()
    if len(records) != len(candidates):
        dump(args.output_dir/f"search_partial_{scope}.json", {
            "status": "pending_full_grid", "completed_or_failed": len(records),
            "planned_candidates": len(candidates), "winner_frozen": False,
            "reason": "resource budget, not scores"})
        return None
    winner = select_candidate(records)
    selection = {"scope": scope, "run_config_sha256": context["run_config_sha256"],
                 "candidate_records": records, "winner": winner,
                 "selected_model": str(directory/candidate_id(winner)/"model.txt"),
                 "selection_rule": "mean annual raw Brier at logloss-best iteration; frozen complexity ties",
                 "reuse_rule": "same params and best iteration for every other arm and lag"}
    dump(args.output_dir/f"selection_{scope}.json", selection)
    return selection


def model_complete_path(args, scope, lag, arm):
    return args.output_dir/"models"/scope/f"lag{lag}_lightgbm_{arm}.json"


def export_model(args, config, registry, scope, lag, arm, columns, model, trained,
                 train_x, train_counts, valid_x, valid_counts):
    path = model_complete_path(args, scope, lag, arm)
    if path.exists():
        return
    records, reliability = [], []
    grid = pd.read_csv(args.grid_registry)
    for matrix, counts in ((train_x, train_counts), (valid_x, valid_counts)):
        offset = 0
        for year, count in counts.items():
            x = np.take(matrix[offset:offset+count], columns, axis=1)
            probability = model.predict(x, num_iteration=trained["prediction_num_iteration"], num_threads=1)
            data = year_data(args.feature_dir, scope, year, grid)
            write_prediction(args.output_dir, scope, lag, f"lightgbm_{arm}", year, data, probability, records, reliability)
            offset += count
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path.with_suffix(".txt")), num_iteration=trained["prediction_num_iteration"])
    pd.DataFrame(records).to_csv(path.with_suffix(".metrics.csv"), index=False)
    pd.DataFrame(reliability).to_csv(path.with_suffix(".reliability.csv"), index=False)
    state = dict(trained, status="complete", scope=scope, lag=lag, arm=arm,
                 exact_features=[registry["names"][index] for index in columns],
                 global_feature_indices=columns.tolist(), exported_model_sha256=sha256(path.with_suffix(".txt")),
                 prediction_years=list(range(2000, 2018)), calibration_years_opened=[], final_years_opened=[])
    dump(path, state)
    refresh_tables(args)


def execute_arms(args, config, registry, scope, arms, context, tasks=None):
    import lightgbm as lgb
    selection_path = args.output_dir/f"selection_{scope}.json"
    if not selection_path.exists():
        raise ValueError("Finish this scope's fixed candidate search before ablation batches")
    selection = json.loads(selection_path.read_text())
    winner = selection["winner"]
    groups = {arm: spec["groups"]+(["P"] if spec["exact_names"] else [])
              for arm, spec in config["features"]["arms"].items()}
    all_columns = arm_columns(registry["names"], registry["groups"], groups)
    if tasks is None:
        tasks = [(lag, arm) for lag in config["input"]["lags"] for arm in arms]
    for lag, group in itertools.groupby(tasks, key=lambda item: item[0]):
        group_arms = [item[1] for item in group]
        pending = [arm for arm in group_arms if not model_complete_path(args, scope, lag, arm).exists()
                   and not (args.output_dir/"reuse"/scope/f"lag{lag}_{arm}"/"failed.json").exists()]
        if not pending:
            continue
        recoverable = any((arm == "F4" and lag == 1) or
                          (args.output_dir/"reuse"/scope/f"lag{lag}_{arm}"/"trained.json").exists() for arm in pending)
        if not recoverable and not may_start_fit(args):
            return
        x, y, counts = load_xy(args.feature_dir, scope, lag, config["input"]["train_years"], np.arange(117))
        vx, vy, valid_counts = load_xy(args.feature_dir, scope, lag, config["input"]["validation_years"], np.arange(117))
        for arm in pending:
            columns = all_columns[arm]
            if len(columns) != config["features"]["arms"][arm]["n_features"]:
                raise ValueError("Frozen feature count mismatch")
            if arm == "F4" and lag == 1:
                trained = winner
                if sha256(Path(selection["selected_model"])) != winner["model_sha256"]:
                    raise ValueError("Selected model hash mismatch")
                model = lgb.Booster(model_file=selection["selected_model"])
            else:
                binding = dict(context, scope=scope, lag=lag, arm=arm, operation="fixed_reuse",
                               selected_candidate=candidate_id(winner), requested_iterations=winner["best_iteration"],
                               exact_features=[registry["names"][index] for index in columns])
                directory = args.output_dir/"reuse"/scope/f"lag{lag}_{arm}"
                previous = checked_record(directory/"trained.json", binding)
                dataset = None
                if not previous:
                    if not may_start_fit(args):
                        return
                    subset = np.take(x, columns, axis=1)
                    params = native_params(config, winner)
                    dataset = lgb.Dataset(subset, label=y, params=params,
                                          feature_name=binding["exact_features"], categorical_feature=[], free_raw_data=True)
                    dataset.construct()
                    del subset
                    gc.collect()
                try:
                    model, trained = fit_or_restore(directory, binding, native_params(config, winner), dataset,
                                                     rounds=winner["best_iteration"])
                except lgb.basic.LightGBMError as error:
                    dump(directory/"failed.json", dict(binding, fingerprint=fingerprint(binding), status="failed", failure=str(error)))
                    del dataset
                    continue
                del dataset
                gc.collect()
            export_model(args, config, registry, scope, lag, arm, columns, model, trained, x, counts, vx, valid_counts)
            update_registry(args, config)
            print(json.dumps({"stage": "model_exported", "scope": scope, "lag": lag, "arm": arm,
                              "iterations": trained["prediction_num_iteration"]}), flush=True)
        del x, y, vx, vy
        gc.collect()


def refresh_tables(args):
    records, reliability = [], []
    for path in (args.output_dir/"models").rglob("*.json"):
        records.extend(pd.read_csv(path.with_suffix(".metrics.csv")).to_dict("records"))
        reliability.extend(pd.read_csv(path.with_suffix(".reliability.csv")).to_dict("records"))
    if records:
        save_tables(args.output_dir, records, reliability)


def update_registry(args, config):
    fits, models = [], []
    interrupted_directories = set()
    for receipt in (args.output_dir/"interrupted_invocations").glob("*.json"):
        record = json.loads(receipt.read_text())
        interrupted_directories.update(Path(path).as_posix() for path in record["interrupted_fit_directories"])
    for scope in config["input"]["scopes"]:
        for candidate in candidate_grid(config):
            path = args.output_dir/"search"/scope/candidate_id(candidate)/"result.json"
            result = json.loads(path.read_text()) if path.exists() else {}
            fallback = "interrupted" if path.parent.as_posix() in interrupted_directories else "not_executed"
            fits.append({"scope": scope, "operation": "search", "candidate": candidate,
                         "status": result.get("status", fallback), "path": str(path)})
        for lag in config["input"]["lags"]:
            for arm in config["features"]["arms"]:
                path = model_complete_path(args, scope, lag, arm)
                failed_path = args.output_dir/"reuse"/scope/f"lag{lag}_{arm}"/"failed.json"
                models.append({"scope": scope, "lag": lag, "arm": arm,
                               "status": "complete" if path.exists() else ("failed" if failed_path.exists() else "not_executed"), "path": str(path)})
                if arm == "F4" and lag == 1:
                    continue
                directory = args.output_dir/"reuse"/scope/f"lag{lag}_{arm}"
                status = "fitted" if (directory/"trained.json").exists() else (
                    "failed" if (directory/"failed.json").exists() else "not_executed")
                fits.append({"scope": scope, "lag": lag, "arm": arm, "operation": "fixed_reuse", "status": status})
    dump(args.output_dir/"execution_registry.json", {
        "planned_formal_fits": 58, "planned_models": 44,
        "resource_pilot_excluded_from_formal_count": True,
        "completed_or_fitted": sum(item["status"] in ("complete", "fitted") for item in fits),
        "failed": sum(item["status"] == "failed" for item in fits),
        "interrupted": sum(item["status"] == "interrupted" for item in fits),
        "not_executed": sum(item["status"] == "not_executed" for item in fits),
        "pending_total": sum(item["status"] in ("not_executed", "interrupted") for item in fits),
        "exported_models": sum(item["status"] == "complete" for item in models),
        "pending_reason": "Awaiting scheduled continuation; interrupted fits remain pending and are not numerical failures",
        "fits": fits, "models": models})


def formal(args):
    config = json.loads(args.run_config.read_text())
    registry = json.loads((args.feature_dir/"2000/feature_registry.json").read_text())
    if sha256(args.feature_dir/"2000/feature_registry.json") != config["input"]["feature_registry_sha256"]:
        raise ValueError("Feature registry differs from the scientific lock")
    if importlib.metadata.version("lightgbm") != config["lightgbm"]["library_version"]:
        raise ValueError("LightGBM version differs from the execution lock")
    runs = {str(year): sha256(args.feature_dir/str(year)/"run.json") for year in range(2000, 2018)}
    context = {"run_config_sha256": sha256(args.run_config), "annual_input_runs_sha256": runs,
               "feature_registry_sha256": config["input"]["feature_registry_sha256"]}
    active_source = {"runner": sha256(Path(__file__)), "helpers": sha256(Path("src/probability_phase3_lightgbm.py"))}
    tasks = None
    if args.tasks:
        tasks = [(int(value.split(":", 1)[0]), value.split(":", 1)[1]) for value in args.tasks]
        if len(set(tasks)) != len(tasks) or any(lag not in (1, 2) or arm not in config["features"]["arms"] for lag, arm in tasks):
            raise ValueError("Tasks must be unique frozen lag:arm pairs")
    if args.stage == "search" and (args.tasks or args.arms):
        raise ValueError("The complete eight-candidate search cannot be arm-filtered")
    if args.output_dir.exists() and not args.resume:
        raise ValueError("Existing formal directory requires explicit --resume")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = args.output_dir/"execution_lock.json"
    if lock_path.exists():
        if json.loads(lock_path.read_text())["context"] != context:
            raise ValueError("Existing run has different frozen inputs or config")
    else:
        dump(lock_path, {"context": context, "config": config,
                         "source_sha256": active_source,
                         "packages": {name: importlib.metadata.version(name) for name in ("lightgbm", "numpy", "pandas", "scipy")}})
        schema = json.loads(Path("data/probability_phase2/experiments/run1/prediction_schema.json").read_text())
        schema.update(run_config_sha256=sha256(args.run_config), model_algorithm="LightGBM", calibration_performed=False)
        dump(args.output_dir/"prediction_schema.json", schema)
    update_registry(args, config)
    started = time.perf_counter()
    for scope in args.scopes:
        if args.stage == "search":
            search_scope(args, config, registry, scope, context)
        else:
            arms = args.arms or config["resources"]["core_arms_first" if args.stage == "core" else "remaining_arms"]
            if any(arm not in config["features"]["arms"] for arm in arms):
                raise ValueError("Unknown frozen feature arm")
            execute_arms(args, config, registry, scope, arms, context, tasks=tasks)
    update_registry(args, config)
    refresh_tables(args)
    invocations = args.output_dir/"invocations"
    invocations.mkdir(exist_ok=True)
    index = len(list(invocations.glob("*.json")))+1
    dump(invocations/f"{index:03d}.json", {"stage": args.stage, "scopes": args.scopes, "command": sys.argv,
                                           "runtime_seconds": time.perf_counter()-started,
                                           "peak_working_set_bytes": peak_memory_bytes(),
                                           "config_sha256": context["run_config_sha256"],
                                           "source_sha256_at_invocation_start": active_source,
                                           "requested_tasks": args.tasks, "requested_arms": args.arms,
                                           "resource_batch_plan": str(args.batch_plan),
                                           "model_years_opened": list(range(2000, 2018)),
                                           "calibration_or_final_years_opened": []})
    print(json.dumps({"stage": "batch_complete", "batch": args.stage, "scopes": args.scopes,
                      "runtime_seconds": time.perf_counter()-started}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=Path("data/probability_phase2/features"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--run-config", type=Path, default=Path("configs/probability_phase3_run.json"))
    parser.add_argument("--grid-registry", type=Path, default=Path("data/probability_phase1/samples/grid_registry.csv"))
    parser.add_argument("--stage", choices=["search", "core", "remaining"], default="search")
    parser.add_argument("--scopes", nargs="+", choices=["temporal", "strict"], default=["temporal"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-plan", type=Path, default=Path("data/probability_phase3/execution_batches.json"))
    filters = parser.add_mutually_exclusive_group()
    filters.add_argument("--arms", nargs="+")
    filters.add_argument("--tasks", nargs="+", help="Explicit fixed execution order, e.g. 1:F0 1:F1 2:F4")
    args = parser.parse_args(argv)
    if args.pilot:
        pilot(args)
    else:
        formal(args)


if __name__ == "__main__":
    main()

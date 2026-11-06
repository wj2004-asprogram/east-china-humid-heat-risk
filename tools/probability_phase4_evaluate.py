"""One externally sealed final evaluation of every prespecified model instance."""
import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.probability_metrics import reliability_counts
from src.probability_phase4_calibration import PlattMapping, calibration_diagnostic
from src.probability_phase4_evaluation import (score_probability, risk_masks, alert_metrics,
    grid_alert_days, daily_brier_sufficient, bootstrap_draws, resampled_brier,
    annual_brier, comparison_point, bootstrap_comparison, fixed_contrasts, summarize_annual)
from src.probability_phase4_data import sha256
from tools.probability_phase4_calibrate import (write_json, write_csv, read_json, load_run,
    start_execution, read_prediction, read_geometry, check_spatial, enrich,
    checkpoint_path, load_checkpoint, save_checkpoint)


RISK_SETS = ("all", "onset", "persistence_or_clearance")
WEIGHTINGS = ("equal_grid", "coslat")
PROBABILITIES = ("raw", "calibrated")


def probability_distribution(p):
    if not len(p):
        return {"probability_mean": None, "probability_std_ddof0": None,
                **{f"probability_q{q}": None for q in (0, 25, 50, 75, 100)}}
    quantiles = np.quantile(p, [0, .25, .5, .75, 1])
    return {"probability_mean": float(np.mean(p)), "probability_std_ddof0": float(np.std(p)),
            **{f"probability_q{q}": float(v) for q, v in zip((0, 25, 50, 75, 100), quantiles)}}


def summarize_reliability(rows, n_years):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in ("scope", "lag", "model_id", "probability", "risk_set", "bin"))].append(row)
    result = []
    for key, annual in groups.items():
        metadata = dict(zip(("scope", "lag", "model_id", "probability", "risk_set", "bin"), key))
        mass = sum(r["n"] / r["annual_risk_n"] / n_years if r["annual_risk_n"] else 0 for r in annual)
        mean_sum = sum((r["mean_probability"] or 0) * r["n"] / r["annual_risk_n"] / n_years if r["annual_risk_n"] else 0 for r in annual)
        event_sum = sum(r["events"] / r["annual_risk_n"] / n_years if r["annual_risk_n"] else 0 for r in annual)
        complete = all(r["annual_risk_n"] for r in annual) and len(annual) == n_years
        result.append({**metadata, "n": sum(r["n"] for r in annual), "events": sum(r["events"] for r in annual),
                       "equal_year_weighted_mass": mass, "complete_year_coverage": complete,
                       "mean_probability": mean_sum / mass if mass and complete else None,
                       "event_fraction": event_sum / mass if mass and complete else None})
    return result


def summarize_alerts(rows):
    groups = defaultdict(list)
    fields = ("scope", "lag", "model_id", "risk_set", "rule")
    rates = ("recall", "precision", "far", "fpr", "miss_rate", "alert_fraction")
    for row in rows:
        groups[tuple(row[k] for k in fields)].append(row)
    result = []
    for key, annual in groups.items():
        record = dict(zip(fields, key))
        record.update({"threshold": annual[0]["threshold"], "years": len(annual)})
        for name in ("n", "tp", "fp", "fn", "tn"):
            record["pooled_" + name] = sum(r[name] for r in annual)
        for name in rates:
            values = [r[name] for r in annual]
            record[name + "_years_available"] = sum(v is not None for v in values)
            record["mean_annual_" + name] = float(np.mean(values)) if all(v is not None for v in values) else None
        result.append(record)
    return result


def compare_outputs(config, score_rows, daily, draws_by_block):
    """Return all paired points, yearly differences, bootstrap intervals and LOYO."""
    years = config["splits"]["final_test"]
    arms = list(config["features"]["arms"])
    pairs = fixed_contrasts(arms)
    score_index = {(r["scope"], r["lag"], r["model_id"], r["year"], r["probability"], r["risk_set"]): r for r in score_rows}
    risks = list(dict.fromkeys(r["risk_set"] for r in score_rows))
    points, annual_pairs, intervals, loyo = [], [], [], []
    for scope in config["scopes"]:
        for lag in config["sample_policy"]["lags"]:
            model_ids = [m["model_id"] for m in config["models"] if m["scope"] == scope and m["lag"] == lag]
            comparisons = [dict(pair, candidate_probability=kind, reference_probability=kind)
                           for pair in pairs for kind in PROBABILITIES]
            comparisons += [{"candidate": model, "reference": model, "roles": "calibration_change",
                             "candidate_probability": "calibrated", "reference_probability": "raw"} for model in model_ids]
            # Every model/type/risk/weight is resampled once per shared draw plan.
            boot = {}
            for block, plan in draws_by_block.items():
                for model in model_ids:
                    for kind in PROBABILITIES:
                        for risk in RISK_SETS:
                            for weighting in WEIGHTINGS:
                                key = (scope, lag, model, kind, risk, weighting)
                                boot[(block, model, kind, risk, weighting)] = resampled_brier(daily[key], plan)
            for pair in comparisons:
                for risk in risks:
                    for weighting in WEIGHTINGS:
                        metric = "brier" if weighting == "equal_grid" else "brier_coslat"
                        def annual(model, kind):
                            values = [score_index[(scope, lag, model, year, kind, risk)][metric] for year in years]
                            return np.array([np.nan if v is None else v for v in values])
                        c = annual(pair["candidate"], pair["candidate_probability"])
                        r = annual(pair["reference"], pair["reference_probability"])
                        metadata = {"scope": scope, "lag": lag, **pair, "risk_set": risk, "weighting": weighting}
                        points.append({**metadata, **comparison_point(c, r)})
                        for i, year in enumerate(years):
                            annual_pairs.append({**metadata, "year": year, **comparison_point(c[i:i+1], r[i:i+1])})
                            loyo.append({**metadata, "omitted_year": year,
                                         **comparison_point(np.delete(c, i), np.delete(r, i))})
                        if risk in RISK_SETS:
                            for block, plan in draws_by_block.items():
                                ck = (block, pair["candidate"], pair["candidate_probability"], risk, weighting)
                                rk = (block, pair["reference"], pair["reference_probability"], risk, weighting)
                                intervals.append({**metadata, "block_days": block, "draw_sha256": plan["sha256"],
                                    **bootstrap_comparison(boot[ck], boot[rk], config["uncertainty"]["ci_minimum_valid_replicates"])})
            print(json.dumps({"comparison_scope_completed": scope, "lag": lag}), flush=True)
    return points, annual_pairs, intervals, loyo


def application_evidence(config, points, intervals, loyo, mappings):
    """Mechanical fixed-cell evidence; no test-driven model replacement."""
    result = []
    gate = config["application_gate"]
    for scope in config["scopes"]:
        cells, reasons = [], []
        for cell in gate["required_cells"]:
            lag_name, risk = cell.split("/")
            lag = int(lag_name[3:])
            def match(row, candidate, reference, cp="raw", rp="raw"):
                return (row["scope"] == scope and row["lag"] == lag and row["risk_set"] == risk
                        and row["weighting"] == "equal_grid" and row["candidate"] == candidate
                        and row["reference"] == reference and row["candidate_probability"] == cp and row["reference_probability"] == rp)
            selected = next(r for r in points if match(r, gate["candidate"], gate["reference"]))
            cis = [r for r in intervals if match(r, gate["candidate"], gate["reference"])]
            deleted = [r for r in loyo if match(r, gate["candidate"], gate["reference"])]
            failed = []
            if selected["delta_brier"] is None or selected["delta_brier"] >= 0:
                failed.append("nonnegative_or_unavailable_mean_delta")
            if selected["years_improving"] < 3:
                failed.append("fewer_than_3_years_improving")
            for ci in cis:
                if ci["delta_brier_upper"] is None or ci["delta_brier_upper"] >= 0:
                    failed.append(f"block_{ci['block_days']}_interval_does_not_exclude_zero")
                if ci["delta_brier_invalid_replicates"]:
                    failed.append(f"block_{ci['block_days']}_incomplete_replicates")
            if any(r["delta_brier"] is None or r["delta_brier"] >= 0 for r in deleted):
                failed.append("leave_one_year_out_not_consistently_negative")
            changes = []
            for model in (gate["candidate"], gate["reference"]):
                mapping = mappings[(scope, lag, model)]
                if mapping["mapping"]["status"] == "identity":
                    failed.append(model + "_calibration_fallback")
                change = next(r for r in intervals if match(r, model, model, "calibrated", "raw") and r["block_days"] == 14)
                changes.append(change)
                if change["delta_brier_lower"] is not None and change["delta_brier_lower"] > 0:
                    failed.append(model + "_calibration_degradation")
            cells.append({"cell": cell, "point": selected, "intervals": cis, "leave_one_year_out": deleted,
                          "calibration_changes": changes, "passed": not failed, "failure_reasons": failed})
            reasons += [cell + ":" + failure for failure in failed]
        result.append({"scope": scope, "candidate": gate["candidate"], "reference": gate["reference"],
                       "passed": not reasons, "failure_reasons": reasons, "cells": cells,
                       "interpretation": "retrospective_hazard_evidence_only_no_deployment_permission"})
    return result


def evaluate_model(root, out, config, gates, ledger, model, mapping_record, geometry, index_registry):
    rows = {name: [] for name in ("scores", "reliability", "alerts", "diagnostics", "distributions", "grids")}
    daily, artifacts = defaultdict(list), []
    scope, lag, model_id = model["scope"], model["lag"], model["model_id"]
    mapping = PlattMapping(**mapping_record["mapping"])
    threshold = mapping_record["budget"]["threshold"]
    for year in config["splits"]["final_test"]:
        arrays, hashes = read_prediction(root, "final_test", model, year, gates, ledger, index_registry)
        artifacts.extend(hashes)
        selected = check_spatial(arrays, geometry[year], scope, "final_test")
        y, p = arrays["y"], arrays["p_raw"]
        lat, lon = selected.latitude.to_numpy(), selected.longitude.to_numpy()
        coslat = np.cos(np.deg2rad(lat))
        calibrated = mapping.apply(p)
        masks = risk_masks(y, arrays["evaluation_previous_y"], lon, lat)
        meta = {"scope": scope, "lag": lag, "model_id": model_id, "year": year}
        for kind, pp in [("raw", p), ("calibrated", calibrated)]:
            rows["diagnostics"].append({**meta, "probability": kind, **calibration_diagnostic(y, pp, np.repeat(year, len(y)))})
            for risk, mask in masks.items():
                info = {**meta, "probability": kind, "risk_set": risk}
                rows["scores"].append({**info, **score_probability(y[mask], pp[mask], lat[mask])})
                rows["distributions"].append({**info, "n": int(mask.sum()), "n_positive": int(y[mask].sum()), **probability_distribution(pp[mask])})
                for row in reliability_counts(y[mask], pp[mask]):
                    rows["reliability"].append({**info, "annual_risk_n": int(mask.sum()), **row})
                if risk in RISK_SETS:
                    for weighting, weight in [("equal_grid", None), ("coslat", coslat)]:
                        daily[(scope, lag, model_id, kind, risk, weighting)].append(
                            daily_brier_sufficient(y, pp, arrays["day_index"], mask, weights=weight))
        for rule, cut in [("primary_0.5", .5), ("calibration_budget_10pct", threshold)]:
            for risk, mask in masks.items():
                rows["alerts"].append({**meta, "risk_set": risk, "rule": rule, **alert_metrics(y[mask], calibrated[mask], cut)})
            grids = geometry[year]
            if scope == "strict":
                grids = grids[grids.spatial_role.eq("holdout")]
            counts = grid_alert_days(arrays["grid_index"], calibrated, cut, grids.index.to_numpy())
            rows["grids"].extend({**meta, "rule": rule, "threshold": cut, **r} for r in counts)
        artifacts.append(enrich(root, "final_test", model, year, arrays, calibrated, threshold, config["protocol_id"]))
    stem = checkpoint_path(out, model).with_suffix("")
    daily_file, grid_file = stem.with_suffix(".npz"), stem.with_suffix(".csv")
    daily_file.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(daily)
    np.savez_compressed(daily_file, keys=np.array([json.dumps(k) for k in keys]), values=np.array([daily[k] for k in keys]))
    write_csv(grid_file, rows.pop("grids"))
    artifacts.extend([daily_file, grid_file])
    return rows, daily, artifacts, grid_file, daily_file


def run(root, config_path):
    root, config_path = Path(root), Path(config_path)
    config, gates, ledger = load_run(root, config_path, "final_test")
    calibrators = read_json(root / "calibration" / "calibrators.json", ledger, "sealed_calibration_parameters")
    if calibrators["freeze_config_sha256"] != sha256(config_path):
        raise ValueError("Calibrators refer to a different scientific configuration")
    mappings = {(r["scope"], r["lag"], r["model_id"]): r for r in calibrators["models"]}
    expected = {(m["scope"], m["lag"], m["model_id"]) for m in config["models"]}
    if set(mappings) != expected or len(calibrators["models"]) != len(expected):
        raise ValueError("Calibrator records do not match all frozen model instances")
    out = root / "evaluation"
    state = start_execution(out, gates, config_path)
    years = config["splits"]["final_test"]
    score_rows, reliability, alerts, diagnostics, distributions = [], [], [], [], []
    index_registry, daily, aggregate_groups = {}, defaultdict(list), defaultdict(list)
    try:
        geometry = {year: read_geometry(root, year, ledger) for year in years}
        block_lengths = [config["uncertainty"]["primary_block_days"], *config["uncertainty"]["sensitivity_block_days"]]
        plans = {block: bootstrap_draws(len(years), config["sample_policy"]["days_per_year"], block,
                                       config["uncertainty"]["replicates"], config["uncertainty"]["seed"]) for block in block_lengths}
        for block, plan in plans.items():
            np.savez_compressed(out / f"bootstrap_draws_{block}.npz", years=plan["years"], dates=plan["dates"], sha256=np.array(plan["sha256"]))
        grid_files = []
        for model_no, model in enumerate(config["models"], 1):
            checkpoint_file = checkpoint_path(out, model)
            checkpoint = load_checkpoint(checkpoint_file, state, ledger)
            if checkpoint is None:
                mapping_record = mappings[(model["scope"], model["lag"], model["model_id"])]
                rows, model_daily, artifacts, grid_file, daily_file = evaluate_model(
                    root, out, config, gates, ledger, model, mapping_record, geometry, index_registry)
                save_checkpoint(checkpoint_file, state, artifacts, rows=rows, grid_file=str(grid_file), daily_file=str(daily_file),
                    index_registry=[[scope, year, digest] for (scope, year), digest in index_registry.items() if scope == model["scope"]])
            else:
                rows = checkpoint["rows"]
                grid_file = Path(checkpoint["grid_file"])
                with np.load(checkpoint["daily_file"], allow_pickle=False) as stored:
                    model_daily = {tuple(json.loads(k)): v for k, v in zip(stored["keys"], stored["values"])}
                for scope, year, digest in checkpoint["index_registry"]:
                    if index_registry.setdefault((scope, year), digest) != digest:
                        raise ValueError("Checkpoint common index mismatch")
            score_rows.extend(rows["scores"])
            reliability.extend(rows["reliability"])
            alerts.extend(rows["alerts"])
            diagnostics.extend(rows["diagnostics"])
            distributions.extend(rows["distributions"])
            daily.update(model_daily)
            for row in rows["scores"]:
                aggregate_groups[tuple(row[k] for k in ("scope", "lag", "model_id", "probability", "risk_set"))].append(row)
            grid_files.append(grid_file)
            print(json.dumps({"evaluated_or_resumed_instances": model_no, "total": len(config["models"]),
                "scope": model["scope"], "lag": model["lag"], "model_id": model["model_id"]}), flush=True)
        with (out / "grid_alert_days.csv").open("w", encoding="utf-8", newline="") as combined:
            for i, path in enumerate(grid_files):
                with path.open(encoding="utf-8", newline="") as source:
                    if i:
                        next(source)
                    for line in source:
                        combined.write(line)
        aggregate = [{**dict(zip(("scope", "lag", "model_id", "probability", "risk_set"), key)), **summarize_annual(rows)}
                     for key, rows in aggregate_groups.items()]
        # Preserve sufficient statistics so the uncertainty calculation is auditable.
        daily_keys = sorted(daily)
        np.savez_compressed(out / "daily_brier_sufficient.npz", values=np.array([daily[k] for k in daily_keys]),
                            keys=np.array([json.dumps(k) for k in daily_keys]), years=np.array(years))
        points, annual_pairs, intervals, loyo = compare_outputs(config, score_rows, daily, plans)
        reports = {"annual_scores.csv": score_rows, "aggregate_scores.csv": aggregate,
                   "reliability.csv": reliability, "reliability_equal_year.csv": summarize_reliability(reliability, len(years)),
                   "alert_metrics.csv": alerts, "aggregate_alert_metrics.csv": summarize_alerts(alerts),
                   "calibration_diagnostics.csv": diagnostics, "probability_distributions.csv": distributions,
                   "comparisons.csv": points, "annual_comparisons.csv": annual_pairs,
                   "bootstrap_intervals.csv": intervals, "leave_one_year_out.csv": loyo}
        for name, rows in reports.items():
            write_csv(out / name, rows)
        write_json(out / "application_gate.json", application_evidence(config, points, intervals, loyo, mappings))
        state.update(status="completed", completed_utc=datetime.now(timezone.utc).isoformat(), model_instances=len(config["models"]),
                     model_year_bundles=len(config["models"])*len(years), common_indices=len(index_registry),
                     bootstrap_draws={str(k): {"sha256": v["sha256"], "replicates": v["replicates"]} for k, v in plans.items()},
                     report_rows={name: len(rows) for name, rows in reports.items()},
                     interpretation="single_frozen_final_evaluation_no_model_or_threshold_selection")
        state["outputs_sha256"] = {str(path): sha256(path) for path in sorted(out.iterdir()) if path.is_file() and path.name not in ("execution_state.json", "run_summary.json")}
        write_json(out / "run_summary.json", state)
    except Exception as exc:
        state.update(status="failed_requires_documented_bug_reopen", error=repr(exc), annual_rows_before_failure=len(score_rows))
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

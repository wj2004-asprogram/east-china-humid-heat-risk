"""Materialize a bounded, auditable CSV view of factorized raw predictions."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=["temporal", "strict"], required=True)
    parser.add_argument("--lag", type=int, choices=[1, 2], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--year", type=int, choices=range(2000, 2018), required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.limit <= 0 or Path(args.model).name != args.model:
        parser.error("Use a new output path, positive limit and a plain model identifier")
    schema = json.loads((args.experiment_dir/"prediction_schema.json").read_text())
    registry = pd.read_csv(schema["grid_registry"])
    path = args.experiment_dir/"predictions"/args.scope/f"lag{args.lag}"/args.model/f"{args.year}.npz"
    with np.load(path, allow_pickle=False) as source:
        values = {name: source[name][:args.limit] for name in source.files}
    frame = registry.iloc[values["grid_index"]].reset_index(drop=True)
    dates = pd.DatetimeIndex(pd.Timestamp(args.year, 6, 9)+pd.to_timedelta(values["day_index"], unit="D"))
    frame["sample_id"] = frame.grid_id+"|"+dates.strftime("%Y-%m-%d")
    frame["target_date_bjt"] = dates.strftime("%Y-%m-%d")
    frame["issue_time_bjt"] = dates.strftime("%Y-%m-%dT00:00:00+08:00")
    frame["issue_time_utc"] = (dates-pd.Timedelta(hours=8)).strftime("%Y-%m-%dT%H:%M:%SZ")
    frame["target_start_bjt"] = frame.issue_time_bjt
    frame["target_start_utc"] = frame.issue_time_utc
    frame["target_end_bjt_exclusive"] = (dates+pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00+08:00")
    frame["target_end_utc_exclusive"] = (dates+pd.Timedelta(hours=16)).strftime("%Y-%m-%dT%H:%M:%SZ")
    frame["max_allowed_input_valid_time_utc"] = (dates-pd.Timedelta(days=args.lag)+pd.Timedelta(hours=15)).strftime("%Y-%m-%dT%H:%M:%SZ")
    frame["scope"], frame["model_id"], frame["lag_days"] = args.scope, args.model, args.lag
    frame["protocol_sha256"] = schema["protocol_sha256"]
    frame["p_raw"], frame["y_D"] = values["p_raw"], values["y"]
    frame["evaluation_y_D_minus1"] = values["evaluation_previous_y"]
    frame["p_calibrated"] = np.nan
    frame["raw_research_alert_threshold"] = .5
    frame["sample_eligibility"] = "common_all_features_both_lags"
    frame["split"] = "train_resubstitution" if args.year <= 2014 else "validation"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame)} ordered preview rows to {args.output}")


if __name__ == "__main__":
    main()

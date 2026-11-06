"""Read one training summer and produce isolated hourly-feature pilot artifacts."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import pandas as pd
import xarray as xr

from src.config import load_config
from src.era5_utils import detect_time_coord, detect_t2m_var, detect_d2m_var
from src.probability_features import DAILY_SPECS, NIGHT_SPECS, daily_features, assemble_pilot_samples


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-project-root", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2000, choices=range(2000, 2015))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("output-dir must be new; existing outputs will not be overwritten")
    cfg = load_config(args.config, project_root=args.source_project_root)
    if cfg.utc_offset_hours != 8 or cfg.months != (6, 7, 8) or args.year not in cfg.years:
        parser.error("Requires a BJT JJA config containing the selected training year")
    raw_files = [raw.path for raw in cfg.raw_files if raw.year == args.year]
    if not raw_files:
        parser.error("No configured raw files for year")
    with ExitStack() as stack:
        parts = []
        for path in raw_files:
            ds = stack.enter_context(xr.open_dataset(path))
            parts.append(ds[[detect_t2m_var(ds), detect_d2m_var(ds)]])
        time_name = detect_time_coord(parts[0])
        hourly = xr.concat(parts, dim=time_name, data_vars="all", coords="minimal",
                           compat="equals", join="exact").sortby(time_name).load()
    start, end = pd.Timestamp(args.year, 6, 1), pd.Timestamp(args.year, 8, 31)
    daily = daily_features(hourly, start, end)
    points = [(35., 115.), (33., 116.), (27., 115.)]
    targets = pd.date_range(pd.Timestamp(args.year, 6, 9), pd.Timestamp(args.year, 6, 15))
    samples = assemble_pilot_samples(daily, hourly, targets, points)
    metadata = {
        "schema_version": 1, "qualification": "COMPUTATIONAL_PILOT_NOT_STATISTICALLY_INDEPENDENT_SAMPLES",
        "year": args.year, "split": "train", "config": str(args.config.resolve()),
        "raw_sources_read_only": [str(path.resolve()) for path in raw_files],
        "daily_variables": {name: {"hourly_source": source, "reduction": how,
                                   "units": "%" if name.endswith("pct") else "degC"}
                            for name, (source, how) in DAILY_SPECS.items()},
        "night_variables": {name: {"hourly_source": source, "reduction": how,
                                   "units": "%" if name.endswith("pct") else "degC"}
                            for name, (source, how) in NIGHT_SPECS.items()},
        "daily_window": "BJT [00:00,next 00:00); exactly 24 unique hours",
        "night_window": "BJT [D-lag-1 20:00,D-lag 08:00); exactly 12 unique hours; last hour 07:00",
        "history_window": "Seven calendar dates [D-(lag+6), D-lag], same-year JJA",
        "lag_columns": "day_lagN means D-N; columns outside selected mode are structural blanks, not missing feature inputs",
        "sample_identity": "sample_id shared between modes; CSV row key is (sample_id,cutoff_lag_days)",
        "missingness": "Missing hourly timestamps raise; nonfinite hourly values propagate to NaN; skipna=False",
        "sample_points_lat_lon": points, "sample_rows": len(samples),
        "finite_feature_rows": int(samples.finite_feature_inputs.sum()),
        "limits": ["No training, calibration or model evaluation", "Only training-year raw files read",
                   "No labels included in sample CSV; himax history is a lagged feature",
                   "Full-region daily NetCDF includes sea cells; formal samples still need land masking",
                   "Pilot does not certify all-years feature completeness or statistical independence"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    daily.to_netcdf(args.output_dir / f"daily_features_{args.year}_bjt.nc",
                    encoding={name: {"dtype": "float32", "zlib": True, "complevel": 4} for name in daily})
    samples.to_csv(args.output_dir / "lagged_feature_pilot.csv", index=False)
    (args.output_dir / "schema.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"year": args.year, "daily_days": daily.sizes["date"],
                      "sample_rows": len(samples), "finite_feature_rows": int(samples.finite_feature_inputs.sum()),
                      "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()

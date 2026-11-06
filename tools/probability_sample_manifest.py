"""Build a factorized structural manifest without reading target label values.

Example: python -m tools.probability_sample_manifest --help
"""

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from src.probability_samples import build_date_manifest, build_grid_registry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates-netcdf", type=Path, required=True)
    parser.add_argument("--time-name", default="time")
    parser.add_argument("--mask-netcdf", type=Path, required=True)
    parser.add_argument("--mask-variable", default="lsm")
    parser.add_argument("--latitude-name", default="latitude")
    parser.add_argument("--longitude-name", default="longitude")
    parser.add_argument("--land-threshold", type=float, default=0.5)
    parser.add_argument("--holdout-bounds", nargs=4, type=float, default=[118, 121, 29, 32],
                        metavar=("WEST", "EAST", "SOUTH", "NORTH"))
    parser.add_argument("--buffer-degrees", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not np.isfinite(args.land_threshold) or not 0 <= args.land_threshold <= 1:
        parser.error("land-threshold must be finite and between 0 and 1")
    # Only the date coordinate is materialized, never the target data array.
    with xr.open_dataset(args.dates_netcdf) as ds:
        dates = ds[args.time_name].values.copy()
        date_lat = ds[args.latitude_name].values.copy()
        date_lon = ds[args.longitude_name].values.copy()
    with xr.open_dataset(args.mask_netcdf) as ds:
        field = ds[args.mask_variable]
        for dim in set(field.dims) - {args.latitude_name, args.longitude_name}:
            if field.sizes[dim] != 1:
                raise ValueError(f"Static mask has nonsingleton extra dimension: {dim}")
            field = field.isel({dim: 0}, drop=True)
        field = field.transpose(args.latitude_name, args.longitude_name)
        # Require exact grid membership; never interpolate a static mask.
        mask_lat = field[args.latitude_name].values
        mask_lon = field[args.longitude_name].values
        if not (np.array_equal(np.sort(date_lat), np.sort(mask_lat)) and
                np.array_equal(np.sort(date_lon), np.sort(mask_lon))):
            raise ValueError("Date source and land mask grids differ")
        values = field.values
        land = np.isfinite(values) & (values >= args.land_threshold)
        grid = build_grid_registry(mask_lat, mask_lon, land,
                                   holdout_bounds=args.holdout_bounds,
                                   buffer_degrees=args.buffer_degrees)
    manifest = build_date_manifest(dates)
    counts = manifest.groupby(["year", "split"])[
        ["lag1_eligible", "lag2_eligible", "common_eligible"]].sum().reset_index()
    for mode in ("lag1", "lag2", "common"):
        counts[f"{mode}_structural_cell_days"] = counts[f"{mode}_eligible"] * len(grid)
    role_counts = grid["spatial_role"].value_counts().to_dict()
    for role in ("train", "buffer", "holdout"):
        counts[f"common_{role}_cell_days"] = counts["common_eligible"] * role_counts.get(role, 0)
    metadata = {
        "schema_version": 1,
        "qualification": "STRUCTURAL_ONLY_NOT_FEATURE_COMPLETE",
        "limitations": ["No hourly-derived night or daily mean dewpoint completeness checked",
                        "No target label values read; no model fitted or evaluated",
                        "Formal comparisons must additionally apply one finite full-feature-union mask"],
        "date_source": str(args.dates_netcdf.resolve()),
        "static_mask_source": str(args.mask_netcdf.resolve()),
        "land_threshold": args.land_threshold,
        "holdout_bounds_west_east_south_north": args.holdout_bounds,
        "buffer_degrees": args.buffer_degrees,
        "spatial_geometry": "Closed core rectangle; closed expanded rectangle minus core is buffer",
        "spatial_grid_counts": {str(k): int(v) for k, v in role_counts.items()},
        "grid_count": len(grid),
        "sample_id_rule": "grid_id + '|' + YYYY-MM-DD; independent of model and lag",
        "formal_sample_rule": "Cartesian product grid_registry x date_manifest[common_eligible]",
        "strict_spatial_training_rule": "Use only spatial_role=train in training years; exclude buffer and holdout from fitting, climatology and historical-label priors",
        "target_threshold_hi_c": 40.6,
        "history_days": 7,
        "cutoff_lag_days": [1, 2],
        "source_utc_semantics": "end_exclusive bounds hourly input; last_hour is end_exclusive minus one hour",
        "missingness_rule": "finite target AND finite frozen union of all models' features; previous-day grouping label is not part of common mask",
        "splits": {"train": [2000, 2014], "validation": [2015, 2017],
                   "calibration": [2018, 2019], "final_test": [2020, 2024]},
        "columns": {"grid_registry.csv": list(grid.columns),
                    "date_manifest.csv": list(manifest.columns),
                    "yearly_counts.csv": list(counts.columns)},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("grid_registry", grid), ("date_manifest", manifest), ("yearly_counts", counts)):
        frame.to_csv(args.output_dir / f"{name}.csv", index=False)
    (args.output_dir / "schema.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"grid_count": len(grid), "date_count": len(manifest),
                      "common_date_count": int(manifest.common_eligible.sum()),
                      "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()

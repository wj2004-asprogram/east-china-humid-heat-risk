"""Freeze geometry-only neighborhood sources; never read weather or labels."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    grid = pd.read_csv(args.grid_registry)
    if grid.grid_id.duplicated().any() or grid[["latitude", "longitude"]].duplicated().any():
        raise ValueError("Grid keys must be unique")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    lat, lon = grid.latitude.to_numpy(), grid.longitude.to_numpy()
    roles = grid.spatial_role.to_numpy()
    if not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise ValueError("Nonfinite coordinates")
    # Fixed 0.25 degree grid; four-neighbor connectivity uses exact integer cells.
    if not np.allclose(lat * 4, np.rint(lat * 4)) or not np.allclose(lon * 4, np.rint(lon * 4)):
        raise ValueError("Expected quarter-degree grid")
    remaining = {(int(round(y*4)), int(round(x*4))) for y, x in zip(lat[roles == "holdout"], lon[roles == "holdout"])}
    components = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        stack, size = [seed], 0
        while stack:
            y, x = stack.pop()
            size += 1
            for neighbor in ((y-1,x), (y+1,x), (y,x-1), (y,x+1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        components.append(size)
    registry, counts = [], []
    for i in range(len(grid)):
        nearby = (np.abs(lat-lat[i]) <= .5) & (np.abs(lon-lon[i]) <= .5)
        nearby[i] = False
        regular = np.flatnonzero(nearby).tolist()
        strict = np.flatnonzero(nearby & (roles == roles[i])).tolist() if roles[i] in ("train", "holdout") else []
        registry.append({"grid_row_index": i, "ordinary_source_row_indices": regular, "strict_source_row_indices": strict})
        counts.append({"grid_id": grid.grid_id.iloc[i], "spatial_role": roles[i],
                       "ordinary_neighbor_count": len(regular), "strict_neighbor_count": len(strict),
                       "ordinary_geometry_eligible": len(regular) >= 3,
                       "strict_geometry_eligible": len(strict) >= 3 and roles[i] in ("train", "holdout")})
    count_table = pd.DataFrame(counts)
    counts_by_role = Counter(roles)
    strict_counts = count_table[count_table.strict_geometry_eligible].spatial_role.value_counts().to_dict()
    neighbor_json = json.dumps(registry, separators=(",", ":")) + "\n"
    summary = {"scope": "geometry_only_no_weather_or_labels", "radius_degrees_each_axis": .5,
               "exclude_center": True, "minimum_sources": 3, "source_weight": "equal", "spatial_std_ddof": 0,
               "grid_counts": dict(counts_by_role), "holdout_four_neighbor_component_sizes": sorted(components, reverse=True),
               "holdout_largest_component_fraction": max(components)/counts_by_role["holdout"],
               "ordinary_eligible_grids": int(count_table.ordinary_geometry_eligible.sum()),
               "ordinary_excluded_grids": int((~count_table.ordinary_geometry_eligible).sum()),
               "strict_eligible_grids_by_role": {k: int(v) for k,v in strict_counts.items()},
               "strict_rule": "train queries source train only; holdout queries source holdout only; buffer excluded",
               "grid_registry_sha256": hashlib.sha256(args.grid_registry.read_bytes()).hexdigest(),
               "neighbor_sources_sha256": hashlib.sha256(neighbor_json.encode()).hexdigest(),
               "source_index_meaning": "Zero-based row in exact grid_registry.csv; hash binds row order",
               "limitation": "Geometric eligibility only; hourly/feature finite-value masks still required"}
    summary["geometry_gate_passed"] = counts_by_role["holdout"] >= 50 and counts_by_role["train"] >= 500 and summary["holdout_largest_component_fraction"] >= .9 and strict_counts.get("holdout", 0) > 0 and strict_counts.get("train", 0) > 0
    count_table.to_csv(args.output_dir / "neighborhood_counts.csv", index=False)
    (args.output_dir / "neighbor_sources.json").write_text(neighbor_json, encoding="utf-8", newline="\n")
    (args.output_dir / "geometry_audit.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

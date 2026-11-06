"""Read-only JJA site screening using previously processed ERA5 heat index.

This is an environmental exposure screening tool, not an occupational risk
assessment. It does not download ERA5, infer workforce exposure or rank ESG.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.risk import CATEGORY_THRESHOLDS_C


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def nearest_cell(latitudes, longitudes, latitude, longitude):
    """Nearest coordinate on a regular grid, without domain extrapolation."""
    if not np.isfinite([latitude, longitude]).all():
        raise ValueError("Site coordinates must be finite WGS84 degrees")
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError("Invalid WGS84 coordinate range")
    if not (min(latitudes) <= latitude <= max(latitudes)
            and min(longitudes) <= longitude <= max(longitudes)):
        raise ValueError("Site is outside the climate grid centre bounds")
    i = int(np.argmin(abs(np.asarray(latitudes) - latitude)))
    j = int(np.argmin(abs(np.asarray(longitudes) - longitude)))
    phi1, phi2 = np.radians([latitude, latitudes[i]])
    dlon = np.radians(longitudes[j] - longitude)
    a = np.sin((phi2 - phi1) / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlon / 2) ** 2
    distance = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return i, j, float(distance)


def summer_metrics(dates, values, threshold_c):
    """Count days and runs in complete JJA summers; withhold incomplete years."""
    dates = pd.DatetimeIndex(dates)
    values = np.asarray(values, dtype=float)
    if len(dates) != len(values) or not len(dates):
        raise ValueError("Dates and values must have the same nonzero length")
    if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Dates must be unique and increasing")
    if not dates.equals(dates.normalize()) or not dates.month.isin([6, 7, 8]).all():
        raise ValueError("Only midnight-labelled JJA daily values are supported")
    if not np.isfinite(threshold_c):
        raise ValueError("Threshold must be finite")
    rows = []
    for year in range(dates.year.min(), dates.year.max() + 1):
        expected = pd.date_range(f"{year}-06-01", f"{year}-08-31")
        aligned = pd.Series(values, index=dates).reindex(expected).to_numpy()
        valid = np.isfinite(aligned)
        complete = bool(valid.all())
        run = longest = 0
        for exceedance in valid & (aligned >= threshold_c):
            run = run + 1 if exceedance else 0
            longest = max(longest, run)
        rows.append({"year": year, "expected_days": len(expected),
                     "valid_days": int(valid.sum()), "complete_summer": complete,
                     "days_ge_threshold": int((aligned >= threshold_c).sum()) if complete else None,
                     "max_consecutive_days": longest if complete else None})
    return rows


def screen(sites, hi, land_mask, threshold_c):
    """Return site summaries and annual rows; preserve site provenance fields."""
    if not np.isfinite(threshold_c):
        raise ValueError("Threshold must be finite")
    required = {"site_id", "name", "latitude", "longitude", "source_url", "coordinate_method"}
    if not required.issubset(sites.columns):
        raise ValueError(f"Missing site columns: {sorted(required - set(sites.columns))}")
    if sites.empty or sites.site_id.isna().any() or sites.site_id.duplicated().any():
        raise ValueError("Sites must have nonempty, unique IDs")
    if any(sites[col].isna().any() or sites[col].astype(str).str.strip().eq("").any()
           for col in required):
        raise ValueError("Required site fields cannot be missing")
    hi = hi.transpose("date", "latitude", "longitude")
    land_mask = land_mask.transpose("latitude", "longitude")
    for coord in ("latitude", "longitude"):
        if not np.array_equal(hi[coord].values, land_mask[coord].values):
            raise ValueError("Heat-index and land-mask grids must match exactly")
        coordinates = hi[coord].values
        if not np.isfinite(coordinates).all() or len(np.unique(coordinates)) != len(coordinates):
            raise ValueError("Grid coordinates must be finite and unique")
    summaries, annual = [], []
    for site in sites.to_dict("records"):
        row = {**site, "threshold_c": float(threshold_c)}
        try:
            i, j, distance = nearest_cell(hi.latitude.values, hi.longitude.values,
                                          float(site["latitude"]), float(site["longitude"]))
        except (ValueError, TypeError) as error:
            row.update(screening_status="invalid_coordinate", screening_note=str(error))
            summaries.append(row)
            continue
        row.update(grid_latitude=float(hi.latitude[i]), grid_longitude=float(hi.longitude[j]),
                   grid_id=f"{float(hi.latitude[i]):.4f},{float(hi.longitude[j]):.4f}",
                   grid_distance_km=round(distance, 3))
        if float(land_mask.values[i, j]) != 1:
            row.update(screening_status="non_land_grid", screening_note="No substitute land grid selected")
            summaries.append(row)
            continue
        metrics = summer_metrics(hi.date.values, hi.values[:, i, j], threshold_c)
        annual.extend({"site_id": site["site_id"], "threshold_c": float(threshold_c), **year}
                      for year in metrics)
        complete = [year for year in metrics if year["complete_summer"]]
        row.update(screening_status="ok" if len(complete) == len(metrics) else "incomplete_climate_data",
                   total_summers=len(metrics), complete_summers=len(complete),
                   valid_days=sum(year["valid_days"] for year in metrics),
                   mean_days_ge_threshold_per_complete_summer=(
                       round(float(np.mean([year["days_ge_threshold"] for year in complete])), 2)
                       if complete else None),
                   max_consecutive_days=(max(year["max_consecutive_days"] for year in complete)
                                         if complete else None))
        summaries.append(row)
    return pd.DataFrame(summaries), pd.DataFrame(annual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", type=Path, required=True)
    parser.add_argument("--heat-index", type=Path, required=True)
    parser.add_argument("--risk", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold-c", type=float, default=CATEGORY_THRESHOLDS_C[3])
    args = parser.parse_args()
    inputs = {"sites": args.sites, "heat_index": args.heat_index, "risk": args.risk}
    outputs = [args.output / name for name in
               ["site_summary.csv", "site_annual.csv", "screening_manifest.json"]]
    if {path.resolve() for path in inputs.values()} & {path.resolve() for path in outputs}:
        raise ValueError("Output files must not overwrite any input")
    hashes = {name: sha256(path) for name, path in inputs.items()}
    with xr.open_dataset(args.heat_index) as heat, xr.open_dataset(args.risk) as risk:
        day_definition = heat.attrs.get("daily_aggregation", "")
        if day_definition != "local calendar days, UTC+8":
            raise ValueError("Pilot requires explicitly labelled Beijing-time daily heat index")
        hi = heat.daily_max_heat_index_c.load()
        land_mask = risk.land_mask.load()
        sites = pd.read_csv(args.sites, keep_default_na=False)
        summaries, annual = screen(sites, hi, land_mask, args.threshold_c)
        dates = pd.DatetimeIndex(hi.date.values)
    if any(sha256(inputs[name]) != digest for name, digest in hashes.items()):
        raise RuntimeError("An input file changed during screening")
    args.output.mkdir(parents=True, exist_ok=True)
    summaries.to_csv(args.output / "site_summary.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(args.output / "site_annual.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "input_sha256": hashes, "day_definition": day_definition,
        "first_date": str(dates.min().date()), "last_date": str(dates.max().date()),
        "threshold_c": args.threshold_c, "comparison": ">=",
        "sites": len(sites), "status_counts": summaries.screening_status.value_counts().to_dict(),
        "unique_land_grids": int(summaries.loc[summaries.screening_status == "ok", "grid_id"].nunique())
                              if "grid_id" in summaries else 0,
        "coordinate_match": "nearest coordinate on regular grid; ties follow file coordinate order",
        "metric_policy": "JJA only; exclude incomplete summers from aggregate metrics; no ocean substitution",
        "interpretation": "Historical environmental screening at current candidate locations; not workforce exposure, occupational compliance or ESG ratings",
        "outputs_sha256": {name: sha256(args.output / name)
                           for name in ["site_summary.csv", "site_annual.csv"]},
    }
    (args.output / "screening_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

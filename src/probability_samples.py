"""Structural sample eligibility for the frozen probability hindcast protocol.

No fitting or label file access occurs here. Date/grid eligibility is NOT proof
that the hourly-derived dewpoint and completed-night features are available.
"""

from collections.abc import Mapping, Sequence
import numpy as np
import pandas as pd


def calendar_dates(values) -> pd.DatetimeIndex:
    """Require unique, timezone-naive local calendar dates; never deduplicate."""
    dates = pd.DatetimeIndex(values)
    if dates.hasnans or dates.tz is not None or not dates.equals(dates.normalize()):
        raise ValueError("Expected finite, timezone-naive midnight calendar dates")
    if dates.has_duplicates:
        raise ValueError("Duplicate calendar dates")
    return dates.sort_values()


def year_split(year: int) -> str:
    for start, end, name in ((2000, 2014, "train"), (2015, 2017, "validation"),
                             (2018, 2019, "calibration"), (2020, 2024, "final_test")):
        if start <= year <= end:
            return name
    raise ValueError(f"Year outside frozen protocol: {year}")


def required_history(target, cutoff_lag_days: int) -> pd.DatetimeIndex:
    """Seven complete local days ending D-lag, within the target's JJA."""
    if cutoff_lag_days not in (1, 2):
        raise ValueError("cutoff_lag_days must be 1 or 2")
    day = calendar_dates([target])[0]
    end = day - pd.Timedelta(days=cutoff_lag_days)
    history = pd.date_range(end - pd.Timedelta(days=6), end, freq="D")
    if day.month not in (6, 7, 8) or any(
        d.year != day.year or d.month not in (6, 7, 8) for d in history
    ):
        raise ValueError("History must be seven complete dates in the same JJA")
    return history


def assert_causal_history(target, feature_dates, cutoff_lag_days: int) -> None:
    """Reject missing days, duplicate dates and any D-1 input in D-2 mode."""
    actual = calendar_dates(feature_dates)
    if not actual.equals(required_history(target, cutoff_lag_days)):
        raise ValueError("Feature dates do not equal the required causal history")


def build_date_manifest(available_dates) -> pd.DataFrame:
    """Include dates eligible for either mode; flag the formal common cohort."""
    dates = calendar_dates(available_dates)
    available = set(dates)
    rows = []
    for day in dates:
        split = year_split(day.year)
        if day.month not in (6, 7, 8):
            continue
        issue = day - pd.Timedelta(hours=8)
        row = {"date": day.date().isoformat(), "year": day.year, "split": split,
               "issue_utc": issue.isoformat() + "Z",
               "target_start_bjt": day.isoformat() + "+08:00",
               "target_end_bjt_exclusive": (day + pd.Timedelta(days=1)).isoformat() + "+08:00",
               "target_start_utc": issue.isoformat() + "Z",
               "target_end_utc_exclusive": (issue + pd.Timedelta(days=1)).isoformat() + "Z"}
        for lag in (1, 2):
            prefix = f"lag{lag}_"
            try:
                history = required_history(day, lag)
                eligible = set(history).issubset(available)
            except ValueError:
                history, eligible = None, False
            row[prefix + "eligible"] = eligible
            row[prefix + "feature_start"] = history[0].date().isoformat() if history is not None else None
            row[prefix + "feature_end"] = history[-1].date().isoformat() if history is not None else None
            # Last hourly observation and exclusive boundary are both explicit.
            boundary = day - pd.Timedelta(days=lag - 1, hours=8)
            row[prefix + "source_utc_end_exclusive"] = boundary.isoformat() + "Z"
            row[prefix + "source_utc_last_hour"] = (boundary - pd.Timedelta(hours=1)).isoformat() + "Z"
        row["common_eligible"] = row["lag1_eligible"] and row["lag2_eligible"]
        if row["lag1_eligible"] or row["lag2_eligible"]:
            rows.append(row)
    if not rows:
        raise ValueError("No structurally eligible target dates")
    return pd.DataFrame(rows)


def spatial_partition(longitude, latitude, holdout_bounds, buffer_degrees):
    """Closed rectangle holdout, closed expanded rectangle buffer, then train.

    Bounds are (west, east, south, north). The buffer is rectangular in degrees,
    not a geodesic distance. Holdout takes precedence on shared boundaries.
    """
    west, east, south, north = holdout_bounds
    if not np.isfinite([west, east, south, north, buffer_degrees]).all() or not (
        west <= east and south <= north and buffer_degrees >= 0
    ):
        raise ValueError("Invalid holdout bounds or buffer")
    lon, lat = np.broadcast_arrays(np.asarray(longitude), np.asarray(latitude))
    if not (np.isfinite(lon).all() and np.isfinite(lat).all()):
        raise ValueError("Grid coordinates must be finite")
    core = (lon >= west) & (lon <= east) & (lat >= south) & (lat <= north)
    b = buffer_degrees
    expanded = (lon >= west-b) & (lon <= east+b) & (lat >= south-b) & (lat <= north+b)
    return np.where(core, "holdout", np.where(expanded, "buffer", "train"))


def build_grid_registry(latitudes, longitudes, land_mask, *, holdout_bounds,
                        buffer_degrees) -> pd.DataFrame:
    """Register finite land cells; IDs derive from unique coordinate pairs."""
    lat, lon = np.asarray(latitudes, dtype=float), np.asarray(longitudes, dtype=float)
    if lat.ndim != 1 or lon.ndim != 1 or len(set(lat)) != len(lat) or len(set(lon)) != len(lon):
        raise ValueError("Expected unique one-dimensional grid axes")
    if not (np.isfinite(lat).all() and np.isfinite(lon).all()):
        raise ValueError("Grid coordinates must be finite")
    mask = np.asarray(land_mask)
    if mask.dtype != np.bool_ or mask.shape != (len(lat), len(lon)):
        raise ValueError("land_mask must be boolean with shape (latitude, longitude)")
    rows = []
    # Sorted coordinates make IDs independent of NetCDF axis direction.
    cells = sorted((float(lat[i]), float(lon[j])) for i, j in zip(*np.where(mask)))
    for latitude, longitude in cells:
        grid_id = f"lat{latitude.hex()}_lon{longitude.hex()}"
        rows.append({"grid_id": grid_id, "latitude": latitude, "longitude": longitude,
                     "spatial_role": str(spatial_partition(longitude, latitude, holdout_bounds, buffer_degrees))})
    return pd.DataFrame(rows, columns=["grid_id", "latitude", "longitude", "spatial_role"])


def sample_id(grid_id: str, target) -> str:
    """One ID per coordinate pair and target day, shared across all models."""
    if not grid_id or "|" in grid_id:
        raise ValueError("Invalid grid_id")
    return grid_id + "|" + calendar_dates([target])[0].date().isoformat()


def common_finite_mask(features: Mapping[str, np.ndarray], required_variables: Sequence[str],
                       target_hi_c, structural_mask=None) -> np.ndarray:
    """One complete-case mask from the frozen union of all model variables.

    Each feature must have sample axes equal to target_hi_c, optionally followed
    by history/component axes. Previous-day labels are NOT required inputs.
    Caller must supply the protocol's full feature union, including ablations.
    """
    if not required_variables or len(set(required_variables)) != len(required_variables):
        raise ValueError("Provide a nonempty unique frozen variable union")
    target = np.asarray(target_hi_c, dtype=float)
    mask = np.isfinite(target)
    for name in required_variables:
        if name not in features:
            raise ValueError(f"Missing required variable: {name}")
        values = np.asarray(features[name], dtype=float)
        if values.ndim < target.ndim or values.shape[:target.ndim] != target.shape:
            raise ValueError(f"Sample axes mismatch: {name}")
        valid = np.isfinite(values)
        if values.ndim > target.ndim:
            valid = valid.all(axis=tuple(range(target.ndim, values.ndim)))
        mask &= valid
    if structural_mask is not None:
        structural = np.asarray(structural_mask)
        if structural.dtype != np.bool_ or structural.shape != target.shape:
            raise ValueError("Structural mask shape/type mismatch")
        mask &= structural
    return mask


def event_labels(hi_c, threshold_c=40.6):
    """Return 0/1 or NaN; missing/infinite HI never becomes a non-event."""
    if not np.isfinite(threshold_c):
        raise ValueError("Threshold must be finite")
    values = np.asarray(hi_c, dtype=float)
    return np.where(np.isfinite(values), (values >= threshold_c).astype(float), np.nan)


def posthoc_event_groups(previous_hi_c, target_hi_c, threshold_c=40.6):
    """Evaluation-only grouping; never feed this or D-1 labels to D-2 features."""
    previous, target = event_labels(previous_hi_c, threshold_c), event_labels(target_hi_c, threshold_c)
    if previous.shape != target.shape:
        raise ValueError("Previous and target label shapes must match")
    groups = np.full(target.shape, "missing", dtype="<U12")
    groups[(previous == 0) & (target == 1)] = "new_onset"
    groups[(previous == 1) & (target == 1)] = "persistent"
    groups[(previous == 1) & (target == 0)] = "resolved"
    groups[(previous == 0) & (target == 0)] = "non_event"
    return groups

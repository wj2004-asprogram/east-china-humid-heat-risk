"""Frozen LightGBM arm mapping, loading and selection helpers (no file writes)."""

from pathlib import Path

import numpy as np


def arm_columns(names, groups, arms):
    """P is exactly the existing numerical y_anchor column, not the H group."""
    names, groups = np.asarray(names), np.asarray(groups)
    if len(names) != len(groups) or len(set(names)) != len(names):
        raise ValueError("Invalid feature registry")
    anchor = np.flatnonzero(names == "y_anchor")
    if len(anchor) != 1 or groups[anchor[0]] != "H":
        raise ValueError("Require exactly one H-group y_anchor feature")
    result = {}
    for arm, selected_groups in arms.items():
        unknown = set(selected_groups)-set(groups)-{"P"}
        if unknown:
            raise ValueError(f"Unknown feature group: {unknown}")
        selected = np.isin(groups, [group for group in selected_groups if group != "P"])
        if "P" in selected_groups:
            selected[anchor[0]] = True
        result[arm] = np.flatnonzero(selected)
    return result


def load_xy(feature_dir, scope, lag, years, columns):
    """Load only explicit train/validation years, preserving row and column order."""
    years = list(years)
    if not years or len(set(years)) != len(years) or any(year not in range(2000, 2018) for year in years):
        raise ValueError("Only unique 2000-2017 training/validation years may be loaded")
    if scope not in ("temporal", "strict") or lag not in (1, 2):
        raise ValueError("Invalid scope or lag")
    counts = {}
    for year in years:
        with np.load(Path(feature_dir)/str(year)/f"{scope}.npz", allow_pickle=False) as source:
            counts[year] = len(source["y"])
    x = np.empty((sum(counts.values()), len(columns)), dtype=np.float32)
    y = np.empty(sum(counts.values()), dtype=np.int8)
    offset = 0
    for year, count in counts.items():
        with np.load(Path(feature_dir)/str(year)/f"{scope}.npz", allow_pickle=False) as source:
            values = source[f"X{lag}"]
            x[offset:offset+count] = values[:, columns]
            y[offset:offset+count] = source["y"]
        offset += count
    if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Frozen common samples contain nonfinite features or invalid labels")
    return x, y, counts


def annual_brier(y, probability, counts):
    if len(y) != sum(counts.values()) or len(probability) != len(y):
        raise ValueError("Annual boundaries do not match predictions")
    result, offset = {}, 0
    for year, count in counts.items():
        result[int(year)] = float(np.mean((probability[offset:offset+count]-y[offset:offset+count])**2))
        offset += count
    return result


def select_candidate(records):
    """Mean annual raw Brier, then fewer leaves / larger child minimum / lambda."""
    valid = [row for row in records if row.get("status") == "complete"]
    if not valid:
        raise ValueError("No successful frozen LightGBM candidate")
    return min(valid, key=lambda row: (row["mean_annual_validation_brier"], row["num_leaves"],
                                       -row["min_child_samples"], -row["reg_lambda"]))

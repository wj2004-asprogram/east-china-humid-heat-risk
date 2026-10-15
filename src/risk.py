"""
risk.py

Heat risk categories and cos(latitude)-weighted area fractions
(optionally over land only).

Categories (based on NOAA heat index thresholds):
    0 = No Risk:          HI < 26.7 °C
    1 = Caution:          26.7 °C <= HI < 32.2 °C
    2 = Extreme Caution:  32.2 °C <= HI < 40.6 °C
    3 = Danger:           40.6 °C <= HI < 54.4 °C
    4 = Extreme Danger:   HI >= 54.4 °C
   -1 = Missing:          HI is missing (NaN); never counted as No Risk
"""

import numpy as np
import pandas as pd
import xarray as xr


CATEGORY_LABELS = {
    0: "No Risk",
    1: "Caution",
    2: "Extreme Caution",
    3: "Danger",
    4: "Extreme Danger",
}

# Lower heat index bound (°C) of categories 1-4.
CATEGORY_THRESHOLDS_C = {
    1: 26.7,
    2: 32.2,
    3: 40.6,
    4: 54.4,
}

CATEGORY_COLORS = {
    "No Risk":          "#2ecc71",  # green
    "Caution":          "#f1c40f",  # yellow
    "Extreme Caution":  "#e67e22",  # orange
    "Danger":           "#e74c3c",  # red
    "Extreme Danger":   "#8e44ad",  # purple
}

# Grid cells (or whole days) without a daily maximum heat index.
MISSING_CATEGORY = -1
MISSING_LABEL = "Missing"
MISSING_COLOR = "#555555"  # dark grey, distinct from the categories and the greyed-out sea

# Area-fraction table column: share (%) of the (land) area with valid data.
VALID_COVERAGE_LABEL = "Valid Land Coverage"


def classify_heat_risk(hi: xr.DataArray) -> xr.DataArray:
    """
    Classify heat index values (°C) into integer risk categories 0-4, and
    MISSING_CATEGORY (-1) where the heat index is missing (NaN).

    Built from the lowest to the highest threshold, so each grid cell ends
    up with the highest category whose threshold it reaches. NaN reaches no
    threshold, so it is set to Missing explicitly instead of No Risk.
    """
    category = xr.full_like(hi, 0, dtype=np.int8)  # default: No Risk

    for cat_val, threshold in CATEGORY_THRESHOLDS_C.items():
        category = xr.where(hi >= threshold, cat_val, category)

    category = xr.where(hi.isnull(), MISSING_CATEGORY, category)
    return category.astype(np.int8)


def area_weights(grid: xr.DataArray, lat_name: str, lon_name: str, land_mask=None) -> xr.DataArray:
    """
    Area weight of each cell of a regular lat/lon grid.

    The weight is cos(latitude), proportional to the true surface area of
    the cell. With a `land_mask` (bool, True = land) the weight of sea cells
    is 0, so area fractions refer to the land area only.

    Returns
    -------
    xr.DataArray
        Weights on the (lat, lon) grid, not normalised.
    """
    cos_lat = np.cos(np.deg2rad(grid[lat_name]))
    weights = cos_lat * xr.ones_like(grid[lon_name], dtype=float)
    if land_mask is not None:
        weights = weights.where(land_mask, 0.0)
    if float(weights.sum()) <= 0.0:
        raise ValueError("No grid cell with a positive area weight (empty land mask?).")
    return weights


def compute_area_fractions(
    category: xr.DataArray,
    lat_name: str,
    lon_name: str,
    date_name: str,
    land_mask=None,
) -> pd.DataFrame:
    """
    Compute the area fraction (in %) of each risk category per day.

    Uses cos(latitude) as the area weight so that grid cells at higher
    latitudes are not over-weighted relative to their true surface area.
    With a `land_mask` (bool, True = land) only land cells are counted, so
    the fractions are fractions of the land area.

    Cells with missing data (MISSING_CATEGORY) are left out: the five
    category fractions are fractions of the valid (land) area of that day
    and sum to 100 %. The VALID_COVERAGE_LABEL column gives that valid area
    as a percentage of the whole (land) area. On a day without any valid
    cell the category fractions are NaN and the coverage is 0.

    Returns
    -------
    pd.DataFrame
        Columns: date, No Risk, Caution, Extreme Caution, Danger,
        Extreme Danger, Valid Land Coverage. All in percent (0-100).
    """
    weights = area_weights(category, lat_name, lon_name, land_mask)
    total_weight = float(weights.sum())

    rows = []
    for i in range(category.sizes[date_name]):
        day_slice = category.isel({date_name: i})
        date_val = day_slice[date_name].values
        date_str = pd.Timestamp(date_val).strftime("%Y-%m-%d")

        # Sum of the area weights over cells with valid data
        valid_weight = float(((day_slice != MISSING_CATEGORY) * weights).sum())

        row = {"date": date_str}
        for cat_val, cat_label in CATEGORY_LABELS.items():
            mask = (day_slice == cat_val)
            # Sum of the area weights over grid cells where mask is True
            weighted_sum = float((mask * weights).sum())
            row[cat_label] = weighted_sum / valid_weight * 100.0 if valid_weight > 0.0 else np.nan
        row[VALID_COVERAGE_LABEL] = valid_weight / total_weight * 100.0

        rows.append(row)

    return pd.DataFrame(rows)


def find_hottest_risk_day(
    category: xr.DataArray,
    date_name: str,
    lat_name: str,
    lon_name: str,
    land_mask=None,
) -> tuple:
    """
    Find the day with the largest area-weighted Danger + Extreme Danger fraction.

    With a `land_mask` (bool, True = land) only land cells are counted.
    Cells with missing data are ignored: the fraction is taken over the
    valid (land) area of each day, as in compute_area_fractions, and days
    without any valid cell are skipped.

    Returns
    -------
    target_date_str : str
        Date string (YYYY-MM-DD) of the selected day.
    danger_frac : float
        Area-weighted fraction of the valid cells with category >= 3 (0-1).
    day_index : int
        Index of the selected day along the date dimension.
    """
    weights = area_weights(category, lat_name, lon_name, land_mask)

    best_idx = 0
    best_frac = -1.0

    for i in range(category.sizes[date_name]):
        day_slice = category.isel({date_name: i})
        valid_weight = float(((day_slice != MISSING_CATEGORY) * weights).sum())
        if valid_weight <= 0.0:
            continue
        frac = float(((day_slice >= 3) * weights).sum()) / valid_weight

        if frac > best_frac:
            best_frac = frac
            best_idx = i

    if best_frac < 0.0:
        raise ValueError("No day with valid data: every grid cell is missing.")

    best_date_str = str(category.isel({date_name: best_idx})[date_name].values)[:10]
    return best_date_str, best_frac, best_idx

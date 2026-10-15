"""
risk.py

Heat risk categories and cos(latitude)-weighted area fractions.

Categories (based on NOAA heat index thresholds):
    0 = No Risk:          HI < 26.7 °C
    1 = Caution:          26.7 °C <= HI < 32.2 °C
    2 = Extreme Caution:  32.2 °C <= HI < 40.6 °C
    3 = Danger:           40.6 °C <= HI < 54.4 °C
    4 = Extreme Danger:   HI >= 54.4 °C
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


def classify_heat_risk(hi: xr.DataArray) -> xr.DataArray:
    """
    Classify heat index values (°C) into integer risk categories 0-4.

    Built from the lowest to the highest threshold, so each grid cell ends
    up with the highest category whose threshold it reaches.
    """
    category = xr.full_like(hi, 0, dtype=np.int8)  # default: No Risk

    for cat_val, threshold in CATEGORY_THRESHOLDS_C.items():
        category = xr.where(hi >= threshold, cat_val, category)

    return category


def cos_lat_weights(category: xr.DataArray, lat_name: str, lon_name: str):
    """
    Cosine-latitude area weights for a regular lat/lon grid.

    Returns
    -------
    weights_1d : xr.DataArray
        cos(lat) for each latitude row (not normalised).
    total_weight : float
        Sum of the weights over all grid cells = sum(cos(lat)) * n_lon.
    """
    lat_rad = np.deg2rad(category[lat_name])
    weights_1d = np.cos(lat_rad)  # shape (n_lat,)

    n_lon = category.sizes[lon_name]
    total_weight = float(weights_1d.sum()) * n_lon
    return weights_1d, total_weight


def compute_area_fractions(
    category: xr.DataArray,
    lat_name: str,
    lon_name: str,
    date_name: str,
) -> pd.DataFrame:
    """
    Compute the area fraction (in %) of each risk category per day.

    Uses cos(latitude) as the area weight so that grid cells at higher
    latitudes are not over-weighted relative to their true surface area.

    Returns
    -------
    pd.DataFrame
        Columns: date, No Risk, Caution, Extreme Caution, Danger, Extreme Danger.
        Area fractions in percent (0-100).
    """
    weights_1d, total_weight = cos_lat_weights(category, lat_name, lon_name)

    rows = []
    for i in range(category.sizes[date_name]):
        day_slice = category.isel({date_name: i})
        date_val = day_slice[date_name].values
        date_str = pd.Timestamp(date_val).strftime("%Y-%m-%d")

        row = {"date": date_str}
        for cat_val, cat_label in CATEGORY_LABELS.items():
            mask = (day_slice == cat_val)
            # Sum of cos(lat) over grid cells where mask is True
            weighted_sum = float((mask * weights_1d).sum())
            frac = weighted_sum / total_weight * 100.0
            row[cat_label] = frac

        rows.append(row)

    return pd.DataFrame(rows)


def find_hottest_risk_day(
    category: xr.DataArray,
    date_name: str,
    lat_name: str,
    lon_name: str,
) -> tuple:
    """
    Find the day with the largest area-weighted Danger + Extreme Danger fraction.

    Returns
    -------
    target_date_str : str
        Date string (YYYY-MM-DD) of the selected day.
    danger_frac : float
        Area-weighted fraction of grid cells with category >= 3 (0-1).
    day_index : int
        Index of the selected day along the date dimension.
    """
    weights_1d, total_weight = cos_lat_weights(category, lat_name, lon_name)

    best_idx = 0
    best_frac = -1.0

    for i in range(category.sizes[date_name]):
        mask = category.isel({date_name: i}) >= 3
        frac = float((mask * weights_1d).sum()) / total_weight

        if frac > best_frac:
            best_frac = frac
            best_idx = i

    best_date_str = str(category.isel({date_name: best_idx})[date_name].values)[:10]
    return best_date_str, best_frac, best_idx

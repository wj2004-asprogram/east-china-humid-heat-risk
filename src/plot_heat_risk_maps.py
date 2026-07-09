"""
plot_heat_risk_maps.py

Read the heat risk categories NetCDF file, automatically select the day
with the largest Danger + Extreme Danger area fraction, and produce two
spatial maps:

  1. Continuous heat index map (daily_max_heat_index_c)
  2. Discrete heat risk category map (heat_risk_category)

Usage:
    python src/plot_heat_risk_maps.py
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FixedLocator
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Coordinate / dimension detection helpers
# ---------------------------------------------------------------------------

def detect_lat_name(ds: xr.Dataset) -> str:
    """Detect the latitude coordinate name (prefers 'latitude' over 'lat')."""
    for candidate in ["latitude", "lat"]:
        if candidate in ds.coords:
            return candidate
    raise KeyError(
        f"Cannot find latitude coordinate (tried 'latitude', 'lat'). "
        f"Available coords: {list(ds.coords)}"
    )


def detect_lon_name(ds: xr.Dataset) -> str:
    """Detect the longitude coordinate name (prefers 'longitude' over 'lon')."""
    for candidate in ["longitude", "lon"]:
        if candidate in ds.coords:
            return candidate
    raise KeyError(
        f"Cannot find longitude coordinate (tried 'longitude', 'lon'). "
        f"Available coords: {list(ds.coords)}"
    )


def detect_date_name(ds: xr.Dataset) -> str:
    """
    Detect the daily time dimension name.

    Prefers 'date' over 'time'.  Checks against dataset dimensions rather
    than coordinates because the classification output uses 'date' as a
    dimension coordinate.
    """
    for candidate in ["date", "time"]:
        if candidate in ds.dims:
            return candidate
    raise KeyError(
        f"Cannot find a daily time dimension (tried 'date', 'time'). "
        f"Dataset dims: {list(ds.dims)}"
    )


# ---------------------------------------------------------------------------
# 2. Date selection
# ---------------------------------------------------------------------------

def find_hottest_risk_day(
    ds: xr.Dataset,
    category_var: str,
    date_name: str,
    lat_name: str,
    lon_name: str,
) -> tuple:
    """
    Find the day with the largest area-weighted Danger + Extreme Danger fraction.

    Danger     = category 3
    Extreme Danger = category 4

    Uses cos(latitude) as the area weight, consistent with
    ``compute_area_fractions`` in ``classify_heat_risk.py``, so grid cells
    at higher latitudes are not over-weighted.

    Parameters
    ----------
    ds : xr.Dataset
        Must contain `category_var` with dimensions (date, lat, lon).
    category_var : str
        Variable name for the risk category.
    date_name : str
        Name of the date/time dimension.
    lat_name : str
        Name of the latitude coordinate.
    lon_name : str
        Name of the longitude coordinate.

    Returns
    -------
    target_date_str : str
        Date string (YYYY-MM-DD) of the hottest risk day.
    danger_frac : float
        Area-weighted fraction of grid cells with category >= 3 (0–1).
    day_index : int
        Integer index of the selected day along the date dimension.
    """
    cat = ds[category_var]

    # Cosine-latitude area weights — consistent with classify_heat_risk.py
    lat_rad = np.deg2rad(ds[lat_name])
    weights_1d = np.cos(lat_rad)  # shape (n_lat,)

    n_lon = cat.sizes[lon_name]
    total_weight = float(weights_1d.sum()) * n_lon

    n_days = cat.sizes[date_name]
    best_idx = 0
    best_frac = -1.0

    for i in range(n_days):
        day_slice = cat.isel({date_name: i})
        mask = day_slice >= 3
        weighted_sum = float((mask * weights_1d).sum())
        frac = weighted_sum / total_weight

        if frac > best_frac:
            best_frac = frac
            best_idx = i

    # Get the date string for the best day
    best_slice = cat.isel({date_name: best_idx})
    best_date_str = str(best_slice[date_name].values)[:10]

    return best_date_str, best_frac, best_idx


# ---------------------------------------------------------------------------
# 3. Plotting
# ---------------------------------------------------------------------------

# City marker definitions
CITIES = {
    "Nanjing":  (32.06, 118.79),
    "Shanghai": (31.23, 121.47),
    "Hangzhou": (30.27, 120.15),
    "Huai'an":  (33.61, 119.02),
}

# Risk category colour map (discrete)
CATEGORY_COLORS = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]
CATEGORY_LABELS = {
    0: "No Risk",
    1: "Caution",
    2: "Extreme Caution",
    3: "Danger",
    4: "Extreme Danger",
}


def _add_city_markers(ax, lon_name, lat_name):
    """Add city scatter markers and text labels to an axis."""
    for city_name, (lat, lon) in CITIES.items():
        ax.scatter(lon, lat, color="black", s=40, zorder=5, edgecolors="white", linewidths=0.5)
        # Slight offset so text does not sit on top of the marker
        ax.text(lon + 0.3, lat + 0.15, city_name, fontsize=8, color="black",
                fontweight="bold", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="none"))


def plot_heat_index_map(
    da_hi: xr.DataArray,
    lat_name: str,
    lon_name: str,
    target_date_str: str,
    output_path: Path,
):
    """
    Plot a continuous filled-contour map of daily maximum heat index.

    Parameters
    ----------
    da_hi : xr.DataArray
        2D slice (lat, lon) for the selected day.
    lat_name, lon_name : str
        Coordinate names.
    target_date_str : str
        Date string for the title (YYYY-MM-DD).
    output_path : Path
        Output PNG path.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    lon = da_hi[lon_name].values
    lat = da_hi[lat_name].values
    values = da_hi.values

    # pcolormesh: centres of grid cells
    pcm = ax.pcolormesh(
        lon, lat, values,
        cmap="YlOrRd",
        shading="auto",
        vmin=values.min(),
        vmax=values.max(),
    )

    cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Daily Maximum Heat Index (°C)")

    _add_city_markers(ax, lon_name, lat_name)

    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(
        f"Daily Maximum Heat Index over East China\n"
        f"Selected hottest risk day: {target_date_str}"
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {output_path}")
    plt.close(fig)


def plot_risk_category_map(
    da_cat: xr.DataArray,
    lat_name: str,
    lon_name: str,
    target_date_str: str,
    output_path: Path,
):
    """
    Plot a discrete-colour map of heat risk categories.

    Parameters
    ----------
    da_cat : xr.DataArray
        2D slice (lat, lon) of integer categories 0–4 for the selected day.
    lat_name, lon_name : str
        Coordinate names.
    target_date_str : str
        Date string for the title (YYYY-MM-DD).
    output_path : Path
        Output PNG path.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    lon = da_cat[lon_name].values
    lat = da_cat[lat_name].values
    values = da_cat.values.astype(int)

    # Discrete colour map with boundaries at half-integers between 0 and 4
    cmap = mcolors.ListedColormap(CATEGORY_COLORS)
    norm = mcolors.BoundaryNorm(
        boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
        ncolors=5,
    )

    pcm = ax.pcolormesh(lon, lat, values, cmap=cmap, norm=norm, shading="auto")

    cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04,
                        ticks=[0, 1, 2, 3, 4])
    cbar.ax.set_yticklabels([
        "No Risk",
        "Caution",
        "Extreme Caution",
        "Danger",
        "Extreme Danger",
    ])

    _add_city_markers(ax, lon_name, lat_name)

    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(
        f"Heat Risk Category over East China\n"
        f"Selected hottest risk day: {target_date_str}"
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def main():
    # --- Paths ---
    project_root = Path(__file__).resolve().parent.parent
    input_path = project_root / "data" / "processed" / "heat_risk_categories_2022_07_01_07.nc"

    fig_dir = project_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    output_hi = fig_dir / "east_china_heat_index_map_hottest_day_2022_07_01_07.png"
    output_cat = fig_dir / "east_china_heat_risk_category_map_hottest_day_2022_07_01_07.png"

    # =======================================================================
    # Step 1: Load data
    # =======================================================================
    print(f"Input file:              {input_path}")

    ds = xr.open_dataset(input_path)

    # Detect coordinate names
    lat_name = detect_lat_name(ds)
    lon_name = detect_lon_name(ds)
    date_name = detect_date_name(ds)

    print(f"Latitude coordinate:     '{lat_name}'")
    print(f"Longitude coordinate:    '{lon_name}'")
    print(f"Date/time dimension:     '{date_name}'")

    # Verify required variables exist
    var_hi = "daily_max_heat_index_c"
    var_cat = "heat_risk_category"

    for v in [var_hi, var_cat]:
        if v not in ds.data_vars:
            raise KeyError(
                f"Variable '{v}' not found. Available: {list(ds.data_vars)}"
            )

    # =======================================================================
    # Step 2: Select the hottest risk day
    # =======================================================================
    target_date_str, danger_frac, day_index = find_hottest_risk_day(
        ds, var_cat, date_name, lat_name, lon_name,
    )

    print(f"Selected target date:    {target_date_str}")
    print(f"Weighted Danger + Extreme Danger area fraction: {danger_frac * 100:.2f}%")

    # Extract 2D slices for the target day
    da_hi_day = ds[var_hi].isel({date_name: day_index})
    da_cat_day = ds[var_cat].isel({date_name: day_index})

    # =======================================================================
    # Step 3: Plot heat index map
    # =======================================================================
    plot_heat_index_map(da_hi_day, lat_name, lon_name, target_date_str, output_hi)

    # =======================================================================
    # Step 4: Plot risk category map
    # =======================================================================
    plot_risk_category_map(da_cat_day, lat_name, lon_name, target_date_str, output_cat)

    print("\nDone.")


if __name__ == "__main__":
    main()

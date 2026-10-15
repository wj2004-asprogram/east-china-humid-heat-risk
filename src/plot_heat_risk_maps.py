"""
plot_heat_risk_maps.py

Read the heat risk categories NetCDF file, select the day with the
largest Danger + Extreme Danger area fraction, and produce two maps:

  1. Continuous heat index map (daily_max_heat_index_c)
  2. Discrete heat risk category map (heat_risk_category)
"""

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import xarray as xr

from src.config import RunConfig
from src.era5_utils import (
    detect_date_dim,
    detect_lat_coord,
    detect_lon_coord,
    load_dataset,
    require_var,
)
from src.risk import CATEGORY_COLORS, CATEGORY_LABELS, find_hottest_risk_day


def _add_city_markers(ax, cities):
    """Add city scatter markers and text labels to an axis."""
    for city in cities:
        ax.scatter(city.lon, city.lat, color="black", s=40, zorder=5,
                   edgecolors="white", linewidths=0.5)
        # Slight offset so text does not sit on top of the marker
        ax.text(city.lon + 0.3, city.lat + 0.15, city.name, fontsize=8, color="black",
                fontweight="bold", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75,
                          edgecolor="none"))


def plot_heat_index_map(
    da_hi: xr.DataArray,
    lat_name: str,
    lon_name: str,
    cities,
    target_date_str: str,
    output_path: Path,
):
    """Plot a continuous map of daily maximum heat index for one day."""
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

    _add_city_markers(ax, cities)

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
    cities,
    target_date_str: str,
    output_path: Path,
):
    """Plot a discrete-colour map of heat risk categories (0-4) for one day."""
    fig, ax = plt.subplots(figsize=(10, 8))

    lon = da_cat[lon_name].values
    lat = da_cat[lat_name].values
    values = da_cat.values.astype(int)

    # Discrete colour map with boundaries at half-integers between 0 and 4
    cmap = mcolors.ListedColormap(list(CATEGORY_COLORS.values()))
    norm = mcolors.BoundaryNorm(
        boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
        ncolors=5,
    )

    pcm = ax.pcolormesh(lon, lat, values, cmap=cmap, norm=norm, shading="auto")

    cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04,
                        ticks=list(CATEGORY_LABELS.keys()))
    cbar.ax.set_yticklabels(list(CATEGORY_LABELS.values()))

    _add_city_markers(ax, cities)

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


def run(cfg: RunConfig) -> None:
    input_path = cfg.paths.heat_risk_nc
    output_hi = cfg.paths.fig_heat_index_map
    output_cat = cfg.paths.fig_category_map
    output_hi.parent.mkdir(parents=True, exist_ok=True)

    # =======================================================================
    # Step 1: Load data
    # =======================================================================
    print(f"Input file:              {input_path}")
    ds = load_dataset(input_path)

    da_hi = require_var(ds, "daily_max_heat_index_c")
    da_cat = require_var(ds, "heat_risk_category")

    lat_name = detect_lat_coord(ds)
    lon_name = detect_lon_coord(ds)
    date_name = detect_date_dim(da_cat)

    # =======================================================================
    # Step 2: Select the hottest risk day
    # =======================================================================
    target_date_str, danger_frac, day_index = find_hottest_risk_day(
        da_cat, date_name, lat_name, lon_name,
    )

    print(f"Selected target date:    {target_date_str}")
    print(f"Weighted Danger + Extreme Danger area fraction: {danger_frac * 100:.2f}%")

    # =======================================================================
    # Step 3: Plot both maps for the selected day
    # =======================================================================
    plot_heat_index_map(
        da_hi.isel({date_name: day_index}), lat_name, lon_name,
        cfg.cities, target_date_str, output_hi,
    )
    plot_risk_category_map(
        da_cat.isel({date_name: day_index}), lat_name, lon_name,
        cfg.cities, target_date_str, output_cat,
    )

    print("\nDone.")

"""
plot_heat_risk_maps.py

Read the heat risk categories NetCDF file, select the day with the
largest Danger + Extreme Danger land area fraction, and produce two maps:

  1. Continuous heat index map (daily_max_heat_index_c)
  2. Discrete heat risk category map (heat_risk_category)

Sea grid cells (land_mask == 0) are kept on the maps but greyed out,
because they are not counted in the area fractions. Cells with missing
data are drawn in a separate colour, never as No Risk.
"""

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.patches import Patch

from src.config import RunConfig
from src.era5_utils import (
    detect_date_dim,
    detect_lat_coord,
    detect_lon_coord,
    load_dataset,
    require_var,
)
from src.risk import (
    CATEGORY_COLORS,
    CATEGORY_LABELS,
    MISSING_CATEGORY,
    MISSING_COLOR,
    MISSING_LABEL,
    find_hottest_risk_day,
)


SEA_COLOR = "lightgrey"


def _grey_out_sea(ax, land: xr.DataArray, lat_name: str, lon_name: str):
    """Cover sea cells (land == False) with a semi-transparent grey layer."""
    sea = np.ma.masked_where(land.values, np.ones(land.shape))  # only sea cells are drawn
    ax.pcolormesh(
        land[lon_name].values, land[lat_name].values, sea,
        cmap=mcolors.ListedColormap([SEA_COLOR]), alpha=0.85, shading="auto", zorder=2,
    )


def _draw_missing(ax, lon, lat, missing: np.ndarray):
    """Fill grid cells with missing data (missing == True) in MISSING_COLOR."""
    cells = np.ma.masked_where(~missing, np.ones(missing.shape))  # only missing cells are drawn
    ax.pcolormesh(lon, lat, cells, cmap=mcolors.ListedColormap([MISSING_COLOR]),
                  shading="auto", zorder=1.5)


def _add_legend(ax, show_missing: bool):
    """Legend for the greyed-out sea and, if any, the missing-data cells."""
    handles = [Patch(facecolor=SEA_COLOR, edgecolor="grey",
                     label="Sea (not counted in area fractions)")]
    if show_missing:
        handles.append(Patch(facecolor=MISSING_COLOR, edgecolor="grey",
                             label=f"{MISSING_LABEL} data (not counted)"))
    ax.legend(handles=handles, loc="lower right", fontsize=8)


def color_limits(values: np.ndarray, land: np.ndarray) -> tuple:
    """
    Colour-scale limits (vmin, vmax) for a heat index map: the range over
    valid land cells, or over all valid cells if no land cell is valid.
    """
    valid = ~np.isnan(values)
    scale_values = values[valid & land]
    if scale_values.size == 0:
        scale_values = values[valid]
    return float(scale_values.min()), float(scale_values.max())


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
    land: xr.DataArray,
    lat_name: str,
    lon_name: str,
    cities,
    target_date_str: str,
    output_path: Path,
):
    """
    Plot a continuous map of daily maximum heat index for one day, sea
    greyed out and missing cells in MISSING_COLOR.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    lon = da_hi[lon_name].values
    lat = da_hi[lat_name].values
    values = da_hi.values
    missing = np.isnan(values)
    vmin, vmax = color_limits(values, land.values)

    # pcolormesh: centres of grid cells
    pcm = ax.pcolormesh(
        lon, lat, values,
        cmap="YlOrRd",
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )

    cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Daily Maximum Heat Index (°C)")

    if missing.any():
        _draw_missing(ax, lon, lat, missing)
    _grey_out_sea(ax, land, lat_name, lon_name)
    _add_legend(ax, show_missing=bool(missing.any()))
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
    land: xr.DataArray,
    lat_name: str,
    lon_name: str,
    cities,
    target_date_str: str,
    output_path: Path,
):
    """
    Plot a discrete-colour map of heat risk categories (0-4) for one day,
    sea greyed out and Missing (-1) cells in MISSING_COLOR.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    lon = da_cat[lon_name].values
    lat = da_cat[lat_name].values
    values = da_cat.values.astype(int)

    # Discrete colour map with boundaries at half-integers between -1 and 4
    codes = [MISSING_CATEGORY, *CATEGORY_LABELS.keys()]
    cmap = mcolors.ListedColormap([MISSING_COLOR, *CATEGORY_COLORS.values()])
    norm = mcolors.BoundaryNorm(
        boundaries=[c - 0.5 for c in codes] + [codes[-1] + 0.5],
        ncolors=len(codes),
    )

    pcm = ax.pcolormesh(lon, lat, values, cmap=cmap, norm=norm, shading="auto")

    cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04, ticks=codes)
    cbar.ax.set_yticklabels([MISSING_LABEL, *CATEGORY_LABELS.values()])

    _grey_out_sea(ax, land, lat_name, lon_name)
    _add_legend(ax, show_missing=False)  # Missing is in the colour bar
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
    land = require_var(ds, "land_mask").astype(bool)

    lat_name = detect_lat_coord(ds)
    lon_name = detect_lon_coord(ds)
    date_name = detect_date_dim(da_cat)

    # =======================================================================
    # Step 2: Select the hottest risk day (largest Danger + Extreme Danger land area)
    # =======================================================================
    target_date_str, danger_frac, day_index = find_hottest_risk_day(
        da_cat, date_name, lat_name, lon_name, land,
    )

    print(f"Selected target date:    {target_date_str}")
    print(f"Weighted Danger + Extreme Danger share of the valid land area: {danger_frac * 100:.2f}%")

    # =======================================================================
    # Step 3: Plot both maps for the selected day
    # =======================================================================
    plot_heat_index_map(
        da_hi.isel({date_name: day_index}), land, lat_name, lon_name,
        cfg.cities, target_date_str, output_hi,
    )
    plot_risk_category_map(
        da_cat.isel({date_name: day_index}), land, lat_name, lon_name,
        cfg.cities, target_date_str, output_cat,
    )

    print("\nDone.")

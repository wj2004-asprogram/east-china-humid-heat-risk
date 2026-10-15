"""
classify_heat_risk.py

Read the processed humid-heat-risk NetCDF file, classify daily maximum
heat index into heat risk categories, compute daily area fractions, and
produce a stacked bar chart.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from src.config import RunConfig
from src.era5_utils import detect_date_dim, detect_lat_coord, detect_lon_coord, require_var
from src.risk import (
    CATEGORY_COLORS,
    CATEGORY_LABELS,
    CATEGORY_THRESHOLDS_C,
    classify_heat_risk,
    compute_area_fractions,
)


def plot_stacked_bar(
    df: pd.DataFrame,
    output_path: Path,
    start_date: str,
    end_date: str,
):
    """
    Draw a stacked bar chart of daily heat risk area fractions.

    Parameters
    ----------
    df : pd.DataFrame
        Columns: date, No Risk, Caution, Extreme Caution, Danger, Extreme Danger.
    output_path : Path
        Where to save the PNG figure.
    start_date, end_date : str
        Date strings for the title (e.g. "2022-07-01").
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    bottom = np.zeros(len(df))
    x = np.arange(len(df))

    for cat in CATEGORY_LABELS.values():
        values = df[cat].values
        ax.bar(
            x, values, bottom=bottom,
            color=CATEGORY_COLORS[cat],
            label=cat,
            width=0.6,
        )
        bottom += values

    # --- Labels & title ---
    ax.set_xticks(x)
    ax.set_xticklabels(df["date"], rotation=45, ha="right")
    ax.set_xlabel("Date")
    ax.set_ylabel("Area Fraction (%)")
    ax.set_title(
        f"Daily Heat Risk Area Fraction over East China\n"
        f"{start_date} to {end_date}"
    )
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, alpha=0.3)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=9,
        title="Risk Category",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {output_path}")
    plt.close(fig)


def category_attrs() -> dict:
    """NetCDF attributes describing the integer risk categories."""
    t = CATEGORY_THRESHOLDS_C
    return {
        "units": "category",
        "description": "Heat risk category based on daily maximum heat index",
        "category_0": f"No Risk: heat index < {t[1]}°C",
        "category_1": f"Caution: {t[1]}°C <= heat index < {t[2]}°C",
        "category_2": f"Extreme Caution: {t[2]}°C <= heat index < {t[3]}°C",
        "category_3": f"Danger: {t[3]}°C <= heat index < {t[4]}°C",
        "category_4": f"Extreme Danger: heat index >= {t[4]}°C",
    }


def run(cfg: RunConfig) -> None:
    input_path = cfg.paths.humid_heat_nc
    output_nc = cfg.paths.heat_risk_nc
    output_csv = cfg.paths.area_fraction_csv
    output_fig = cfg.paths.fig_area_fraction

    output_nc.parent.mkdir(parents=True, exist_ok=True)
    output_fig.parent.mkdir(parents=True, exist_ok=True)

    # =======================================================================
    # Step 1: Load data
    # =======================================================================
    print(f"Input file:    {input_path}")
    ds = xr.open_dataset(input_path)

    var_name = "daily_max_heat_index_c"
    da = require_var(ds, var_name)
    lat_name = detect_lat_coord(da)
    lon_name = detect_lon_coord(da)
    date_name = detect_date_dim(da)

    print(f"Variable:      {var_name}")
    print(f"Date dim:      '{date_name}'")
    print(f"Latitude:      '{lat_name}'")
    print(f"Longitude:     '{lon_name}'")

    # =======================================================================
    # Step 2: Classify heat risk
    # =======================================================================
    heat_risk_category = classify_heat_risk(da)
    heat_risk_category.attrs = category_attrs()

    # =======================================================================
    # Step 3: Save NetCDF
    # =======================================================================
    ds_out = xr.Dataset(
        {
            "daily_max_heat_index_c": da,
            "heat_risk_category": heat_risk_category,
        },
        attrs={
            "title": "Heat risk categories for East China",
            "period": cfg.period_label,
            "source": "Based on ERA5 hourly single-level reanalysis",
            "created_with": "src/classify_heat_risk.py",
        },
    )
    ds_out.to_netcdf(output_nc)
    print(f"\nNetCDF saved:  {output_nc}")

    # =======================================================================
    # Step 4: Compute area fractions
    # =======================================================================
    df = compute_area_fractions(heat_risk_category, lat_name, lon_name, date_name)

    print("\n=== Daily heat risk area fractions (%) ===")
    print(df.to_string(index=False))

    # Sanity check: the categories cover every grid cell, so each day sums to 100 %
    daily_sums = df[list(CATEGORY_LABELS.values())].sum(axis=1)
    print("\nDaily sum check (should be ~100%):")
    for date_str, total in zip(df["date"], daily_sums):
        print(f"  {date_str}: {total:.2f}%")

    # =======================================================================
    # Step 5: Save CSV
    # =======================================================================
    df.to_csv(output_csv, index=False)
    print(f"\nCSV saved:     {output_csv}")

    # =======================================================================
    # Step 6: Plot
    # =======================================================================
    plot_stacked_bar(df, output_fig, df["date"].iloc[0], df["date"].iloc[-1])

    print("\nDone.")

"""
classify_heat_risk.py

Read the processed humid-heat-risk NetCDF file, classify daily maximum
heat index into heat risk categories, compute daily area fractions over
land (ERA5 land-sea mask >= the configured threshold), and produce a
stacked bar chart.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from src.config import RunConfig
from src.era5_utils import (
    detect_date_dim,
    detect_lat_coord,
    detect_lon_coord,
    load_dataset,
    require_var,
)
from src.land_sea_mask import load_land_mask
from src.risk import (
    CATEGORY_COLORS,
    CATEGORY_LABELS,
    CATEGORY_THRESHOLDS_C,
    MISSING_CATEGORY,
    MISSING_LABEL,
    VALID_COVERAGE_LABEL,
    classify_heat_risk,
    compute_area_fractions,
)


# Coverage below 100 % minus this (percentage points) counts as partial;
# float sums can end just below 100 when nothing is missing.
COVERAGE_TOLERANCE = 1e-9

# Longer runs print a summary instead of every day.
MAX_DAYS_PRINTED = 62


def is_partial_coverage(coverage_pct: pd.Series) -> pd.Series:
    """True for days whose valid land coverage (%) is below 100 %."""
    return coverage_pct < 100.0 - COVERAGE_TOLERANCE


def season_means(df: pd.DataFrame, seasons) -> pd.DataFrame:
    """
    Mean of the daily area fractions over each season, labelled with the
    season's year in the "date" column (for the multi-year bar chart).

    Category columns are means over the days with valid data; the
    coverage column is the mean daily valid land coverage.
    """
    dates = pd.to_datetime(df["date"])
    rows = []
    for first, last in seasons:
        days = df[(dates >= pd.Timestamp(first)) & (dates <= pd.Timestamp(last))]
        row = {"date": str(first.year)}
        for col in [*CATEGORY_LABELS.values(), VALID_COVERAGE_LABEL]:
            row[col] = days[col].mean()  # skips NaN (days without valid data)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_stacked_bar(
    df: pd.DataFrame,
    output_path: Path,
    period: str,
    x_label: str = "Date",
    what: str = "Daily Heat Risk Area Fraction",
):
    """
    Draw a stacked bar chart of heat risk land area fractions.

    Fractions are shares of the valid land area. Bars whose valid land
    coverage is below 100 % are marked with "*"; bars without valid data
    are empty.

    Parameters
    ----------
    df : pd.DataFrame
        Columns: date (bar label), No Risk, Caution, Extreme Caution,
        Danger, Extreme Danger, Valid Land Coverage.
    output_path : Path
        Where to save the PNG figure.
    period : str
        Period for the title (e.g. "2022-07-01 to 2022-07-31").
    x_label, what : str
        Axis label of the bars and first title line.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    bottom = np.zeros(len(df))
    x = np.arange(len(df))

    for cat in CATEGORY_LABELS.values():
        values = np.nan_to_num(df[cat].values)  # no bar on days without valid data
        ax.bar(
            x, values, bottom=bottom,
            color=CATEGORY_COLORS[cat],
            label=cat,
            width=0.6,
        )
        bottom += values

    # --- Labels & title ---
    partial = is_partial_coverage(df[VALID_COVERAGE_LABEL])
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{d}*" if p else d for d, p in zip(df["date"], partial)], rotation=45, ha="right",
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Share of Valid Land Area (%)")
    title = f"{what} over East China (land only)\n{period}"
    if partial.any():
        title += "\n* valid land coverage below 100 % (missing data)"
    ax.set_title(title)
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
        "category_missing": (
            f"{MISSING_CATEGORY} = {MISSING_LABEL}: daily maximum heat index is "
            "missing (NaN); not counted in the area fractions"
        ),
    }


def land_mask_attrs(threshold: float) -> dict:
    """NetCDF attributes describing the land mask."""
    return {
        "units": "1",
        "long_name": "land mask",
        "description": (
            f"1 = land (ERA5 land_sea_mask >= {threshold:g}), 0 = sea; "
            "risk area fractions count land cells only"
        ),
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
    ds = load_dataset(input_path)

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

    land_mask = load_land_mask(
        cfg.paths.raw_land_sea_mask, cfg.land_sea_mask_threshold, da, lat_name, lon_name,
    )
    print(f"Land mask:     {int(land_mask.sum())} of {land_mask.size} grid cells are land "
          f"(land_sea_mask >= {cfg.land_sea_mask_threshold:g})")

    # =======================================================================
    # Step 3: Save NetCDF
    # =======================================================================
    ds_out = xr.Dataset(
        {
            "daily_max_heat_index_c": da,
            "heat_risk_category": heat_risk_category,
            "land_mask": land_mask.astype(np.int8).assign_attrs(
                land_mask_attrs(cfg.land_sea_mask_threshold)
            ),
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
    # Step 4: Compute area fractions over land
    # =======================================================================
    df = compute_area_fractions(heat_risk_category, lat_name, lon_name, date_name, land_mask)

    print("\n=== Daily heat risk land area fractions (%) ===")
    print(df.to_string(index=False) if len(df) <= MAX_DAYS_PRINTED else df.to_string(index=False, max_rows=10))

    # Sanity check: the categories cover every valid land cell, so each day
    # with valid data sums to 100 %
    daily_sums = df[list(CATEGORY_LABELS.values())].sum(axis=1, min_count=1)
    print("\nDaily sum check (should be ~100% where coverage > 0):")
    if len(df) <= MAX_DAYS_PRINTED:
        for date_str, total, coverage in zip(df["date"], daily_sums, df[VALID_COVERAGE_LABEL]):
            print(f"  {date_str}: {total:.2f}%  (valid land coverage {coverage:.2f}%)")
    else:
        print(f"  {len(df)} days: sums {daily_sums.min():.2f} to {daily_sums.max():.2f}%, "
              f"valid land coverage {df[VALID_COVERAGE_LABEL].min():.2f} to "
              f"{df[VALID_COVERAGE_LABEL].max():.2f}%")

    # =======================================================================
    # Step 5: Save CSV
    # =======================================================================
    df.to_csv(output_csv, index=False)
    print(f"\nCSV saved:     {output_csv}")

    # =======================================================================
    # Step 6: Plot
    # =======================================================================
    if cfg.is_multi_year:
        # One bar per season: thousands of daily bars would be unreadable.
        plot_stacked_bar(
            season_means(df, cfg.seasons), output_fig, cfg.period_label,
            x_label="Year", what="Mean Daily Heat Risk Area Fraction per Season",
        )
    else:
        plot_stacked_bar(df, output_fig, f"{df['date'].iloc[0]} to {df['date'].iloc[-1]}")

    print("\nDone.")

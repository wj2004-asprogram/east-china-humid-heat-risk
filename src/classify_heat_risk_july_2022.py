"""
classify_heat_risk.py

Read the processed humid-heat-risk NetCDF file, classify daily maximum
heat index into heat risk categories, compute daily area fractions, and
produce a stacked bar chart.

Usage:
    python src/classify_heat_risk.py
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Coordinate / dimension detection helpers
# ---------------------------------------------------------------------------

def detect_lat_name(ds: xr.Dataset) -> str:
    """Detect the latitude coordinate name."""
    for candidate in ["latitude", "lat"]:
        if candidate in ds.coords:
            return candidate
    raise KeyError(
        f"Cannot find latitude coordinate (tried 'latitude', 'lat'). "
        f"Available coords: {list(ds.coords)}"
    )


def detect_lon_name(ds: xr.Dataset) -> str:
    """Detect the longitude coordinate name."""
    for candidate in ["longitude", "lon"]:
        if candidate in ds.coords:
            return candidate
    raise KeyError(
        f"Cannot find longitude coordinate (tried 'longitude', 'lon'). "
        f"Available coords: {list(ds.coords)}"
    )


def detect_date_name(da: xr.DataArray) -> str:
    """
    Detect the daily time dimension name from a DataArray.

    Prefers 'date' over 'time'.
    """
    for candidate in ["date", "time"]:
        if candidate in da.dims:
            return candidate
    raise KeyError(
        f"Cannot find a daily time dimension (tried 'date', 'time'). "
        f"Array dims: {list(da.dims)}"
    )


# ---------------------------------------------------------------------------
# 2. Heat risk classification
# ---------------------------------------------------------------------------

def classify_heat_risk(hi: xr.DataArray) -> xr.DataArray:
    """
    Classify heat index values into risk categories.

    Categories (based on NOAA heat index thresholds):
        0 = No Risk:          HI < 26.7 °C
        1 = Caution:          26.7 °C <= HI < 32.2 °C
        2 = Extreme Caution:  32.2 °C <= HI < 40.6 °C
        3 = Danger:           40.6 °C <= HI < 54.4 °C
        4 = Extreme Danger:   HI >= 54.4 °C

    Parameters
    ----------
    hi : xr.DataArray
        Heat index values in °C.

    Returns
    -------
    xr.DataArray
        Integer risk categories (0-4) with the same shape as `hi`.
    """
    category = xr.zeros_like(hi, dtype=np.int8)

    category = category.where(~(hi >= 26.7), 1)   # Caution
    category = category.where(~(hi >= 32.2), 2)   # Extreme Caution
    category = category.where(~(hi >= 40.6), 3)   # Danger
    category = category.where(~(hi >= 54.4), 4)   # Extreme Danger

    # For values already set to 0 (HI < 26.7), xr.where leaves them at 0
    # because the first condition (hi >= 26.7) is False for those.
    # However, xr.where always replaces where condition is True, so we need
    # to build from the top down to avoid overwriting.
    #
    # Let's use a cleaner approach: build from highest to lowest.

    category = xr.full_like(hi, 0, dtype=np.int8)  # default: No Risk

    category = xr.where(hi >= 26.7, 1, category)
    category = xr.where(hi >= 32.2, 2, category)
    category = xr.where(hi >= 40.6, 3, category)
    category = xr.where(hi >= 54.4, 4, category)

    return category


# ---------------------------------------------------------------------------
# 3. Area fraction calculation
# ---------------------------------------------------------------------------

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

    Parameters
    ----------
    category : xr.DataArray
        Integer risk categories with dimensions (date, lat, lon).
    lat_name : str
        Name of the latitude coordinate.
    lon_name : str
        Name of the longitude coordinate.
    date_name : str
        Name of the date/time dimension.

    Returns
    -------
    pd.DataFrame
        Columns: date, No Risk, Caution, Extreme Caution, Danger, Extreme Danger.
        Area fractions in percent (0-100).
    """
    # Cosine latitude weights (raw, not normalised yet)
    lat_rad = np.deg2rad(category[lat_name])
    weights_1d = np.cos(lat_rad)  # shape (n_lat,)

    # Total weight across all grid cells  =  sum(cos(lat)) * n_lon
    n_lon = category.sizes[lon_name]
    total_weight = float(weights_1d.sum()) * n_lon

    category_labels = {
        0: "No Risk",
        1: "Caution",
        2: "Extreme Caution",
        3: "Danger",
        4: "Extreme Danger",
    }

    rows = []
    for i in range(category.sizes[date_name]):
        day_slice = category.isel({date_name: i})
        date_val = day_slice[date_name].values
        date_str = pd.Timestamp(date_val).strftime("%Y-%m-%d")

        row = {"date": date_str}
        for cat_val, cat_label in category_labels.items():
            mask = (day_slice == cat_val)
            # Sum of cos(lat) over grid cells where mask is True
            weighted_sum = float((mask * weights_1d).sum())
            frac = weighted_sum / total_weight * 100.0
            row[cat_label] = frac

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# 4. Plotting
# ---------------------------------------------------------------------------

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
    start_date : str
        Start date string for the title (e.g. "2022-07-01").
    end_date : str
        End date string for the title (e.g. "2022-07-07").
    """
    category_colors = {
        "No Risk":          "#2ecc71",  # green
        "Caution":          "#f1c40f",  # yellow
        "Extreme Caution":  "#e67e22",  # orange
        "Danger":           "#e74c3c",  # red
        "Extreme Danger":   "#8e44ad",  # purple
    }

    category_order = [
        "No Risk",
        "Caution",
        "Extreme Caution",
        "Danger",
        "Extreme Danger",
    ]

    fig, ax = plt.subplots(figsize=(12, 6))

    bottom = np.zeros(len(df))
    x = np.arange(len(df))

    for cat in category_order:
        values = df[cat].values
        ax.bar(
            x, values, bottom=bottom,
            color=category_colors[cat],
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


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main():
    # --- Paths ---
    project_root = Path(__file__).resolve().parent.parent
    input_path = project_root / "data" / "processed" / "humid_heat_2022_07.nc"
    output_nc = project_root / "data" / "processed" / "heat_risk_categories_2022_07.nc"
    output_csv = project_root / "data" / "processed" / "daily_heat_risk_area_fraction_2022_07.csv"
    output_fig = project_root / "figures" / "east_china_heat_risk_area_fraction_2022_07.png"

    # Ensure output directories exist
    output_nc.parent.mkdir(parents=True, exist_ok=True)
    output_fig.parent.mkdir(parents=True, exist_ok=True)

    # =======================================================================
    # Step 1: Load data
    # =======================================================================
    print(f"Input file:    {input_path}")
    ds = xr.open_dataset(input_path)

    var_name = "daily_max_heat_index_c"
    if var_name not in ds.data_vars:
        raise KeyError(
            f"Variable '{var_name}' not found. "
            f"Available: {list(ds.data_vars)}"
        )

    da = ds[var_name]
    lat_name = detect_lat_name(da)
    lon_name = detect_lon_name(da)
    date_name = detect_date_name(da)

    print(f"Variable:      {var_name}")
    print(f"Date dim:      '{date_name}'")
    print(f"Latitude:      '{lat_name}'")
    print(f"Longitude:     '{lon_name}'")

    # =======================================================================
    # Step 2: Classify heat risk
    # =======================================================================
    heat_risk_category = classify_heat_risk(da)

    # Add metadata
    heat_risk_category.attrs = {
        "units": "category",
        "description": "Heat risk category based on daily maximum heat index",
        "category_0": "No Risk: heat index < 26.7°C",
        "category_1": "Caution: 26.7°C <= heat index < 32.2°C",
        "category_2": "Extreme Caution: 32.2°C <= heat index < 40.6°C",
        "category_3": "Danger: 40.6°C <= heat index < 54.4°C",
        "category_4": "Extreme Danger: heat index >= 54.4°C",
    }

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
            "period": "2022-07-01 to 2022-07-31",
            "source": "Based on ERA5 hourly single-level reanalysis",
            "created_with": "classify_heat_risk.py",
        },
    )
    ds_out.to_netcdf(output_nc)
    print(f"\nNetCDF saved:  {output_nc}")

    # =======================================================================
    # Step 4: Compute area fractions
    # =======================================================================
    df = compute_area_fractions(heat_risk_category, lat_name, lon_name, date_name)

    # Print per-day fractions
    print("\n=== Daily heat risk area fractions (%) ===")
    print(df.to_string(index=False))

    # Verify sums are close to 100%
    sum_cols = ["No Risk", "Caution", "Extreme Caution", "Danger", "Extreme Danger"]
    daily_sums = df[sum_cols].sum(axis=1)
    print(f"\nDaily sum check (should be ~100%):")
    for _, row in df.iterrows():
        total = sum(row[col] for col in sum_cols)
        print(f"  {row['date']}: {total:.2f}%")

    # =======================================================================
    # Step 5: Save CSV
    # =======================================================================
    df.to_csv(output_csv, index=False)
    print(f"\nCSV saved:     {output_csv}")

    # =======================================================================
    # Step 6: Plot
    # =======================================================================
    dates_list = df["date"].tolist()
    start_date = dates_list[0]
    end_date = dates_list[-1]

    plot_stacked_bar(df, output_fig, start_date, end_date)

    print("\nDone.")


if __name__ == "__main__":
    main()

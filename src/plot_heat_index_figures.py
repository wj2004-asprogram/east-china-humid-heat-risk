"""
plot_heat_index_figures.py

Read processed humid-heat-risk NetCDF data and produce:

1. A regional-mean daily maximum heat index time series for East China.
2. A four-city comparison of daily maximum heat index at the nearest
   ERA5 grid point.

Also exports a CSV table of the city-level daily heat index values.

Usage:
    python src/plot_heat_index_figures.py
"""

import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Helper functions — auto-detect dataset metadata
# ---------------------------------------------------------------------------

def detect_lat_name(ds: xr.Dataset) -> str:
    """
    Detect the latitude coordinate name.

    Supports both ERA5-style "latitude" and shortened "lat".
    """
    for candidate in ["latitude", "lat"]:
        if candidate in ds.coords:
            return candidate
    raise KeyError(
        f"Cannot find a latitude coordinate (tried 'latitude', 'lat'). "
        f"Available coords: {list(ds.coords)}"
    )


def detect_lon_name(ds: xr.Dataset) -> str:
    """
    Detect the longitude coordinate name.

    Supports both ERA5-style "longitude" and shortened "lon".
    """
    for candidate in ["longitude", "lon"]:
        if candidate in ds.coords:
            return candidate
    raise KeyError(
        f"Cannot find a longitude coordinate (tried 'longitude', 'lon'). "
        f"Available coords: {list(ds.coords)}"
    )


def detect_date_name(ds: xr.Dataset, da: xr.DataArray) -> str:
    """
    Detect the daily time coordinate name from a DataArray's dimensions.

    Prefers 'date' over 'time' so that daily-resolution coordinates
    are picked up before sub-daily ones.

    Parameters
    ----------
    ds : xr.Dataset
        Opened dataset.
    da : xr.DataArray
        The target data array (used to check which dims are present).

    Returns
    -------
    str
        Name of the time/date coordinate.
    """
    for candidate in ["date", "time"]:
        if candidate in da.dims and candidate in ds.coords:
            return candidate
    raise KeyError(
        f"Cannot find a time/date coordinate in the data array. "
        f"Array dims: {list(da.dims)}, dataset coords: {list(ds.coords)}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- City definitions ---
    cities = {
        "Nanjing":  (32.06, 118.79),
        "Shanghai": (31.23, 121.47),
        "Hangzhou": (30.27, 120.15),
        "Huai'an":  (33.61, 119.02),
    }

    # --- Paths ---
    project_root = Path(__file__).resolve().parent.parent
    input_file = project_root / "data" / "processed" / "humid_heat_2022_07_01_07.nc"

    output_fig1 = project_root / "figures" / "east_china_mean_daily_heat_index_2022_07_01_07.png"
    output_fig2 = project_root / "figures" / "east_china_city_daily_heat_index_2022_07_01_07.png"
    output_csv  = project_root / "data" / "processed" / "city_daily_heat_index_2022_07_01_07.csv"

    # Ensure output directories exist
    output_fig1.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    print(f"Loading {input_file} ...")
    ds = xr.open_dataset(input_file)
    print(ds)

    # --- Auto-detect coordinates ---
    lat_name = detect_lat_name(ds)
    lon_name = detect_lon_name(ds)

    var_name = "daily_max_heat_index_c"
    if var_name not in ds.data_vars:
        raise KeyError(
            f"Variable '{var_name}' not found in dataset. "
            f"Available: {list(ds.data_vars)}"
        )

    da = ds[var_name]
    date_name = detect_date_name(ds, da)

    print(f"Detected — lat: '{lat_name}', lon: '{lon_name}', "
          f"date: '{date_name}', var: '{var_name}'")

    # ======================================================================
    # Figure 1 — Regional-mean daily maximum heat index over East China
    # ======================================================================

    regional_mean = da.mean(dim=[lat_name, lon_name])

    fig1, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(
        regional_mean[date_name], regional_mean,
        marker="o", linestyle="-", linewidth=1.5, color="darkred",
    )

    ax1.set_title(
        "Mean Daily Maximum Heat Index over East China\n"
        "2022-07-01 to 2022-07-07",
        fontsize=13,
    )
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Daily Maximum Heat Index (°C)")
    ax1.grid(True, alpha=0.3)
    fig1.autofmt_xdate()

    fig1.tight_layout()
    fig1.savefig(output_fig1, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_fig1}")
    plt.close(fig1)

    # ======================================================================
    # Figure 2 — Four-city daily maximum heat index comparison
    # ======================================================================

    colors = {
        "Nanjing":  "#d62728",  # red
        "Shanghai": "#1f77b4",  # blue
        "Hangzhou": "#2ca02c",  # green
        "Huai'an":  "#ff7f0e",  # orange
    }

    # Collect city data for CSV (wide format)
    city_series = {}   # city -> list of (date, value)
    dates_for_csv = None

    fig2, ax2 = plt.subplots(figsize=(12, 6))

    for city, (lat, lon) in cities.items():
        # Select nearest grid point
        ds_city = da.sel({
            lat_name: lat,
            lon_name: lon,
        }, method="nearest")

        # Report actual matched coordinates
        actual_lat = float(ds_city[lat_name].values)
        actual_lon = float(ds_city[lon_name].values)
        print(f"{city:>10s}: requested ({lat:.2f}°N, {lon:.2f}°E)  "
              f"→ matched ({actual_lat:.2f}°N, {actual_lon:.2f}°E)")

        # Extract time series
        time_vals = ds_city[date_name].values
        heat_vals = ds_city.values

        if dates_for_csv is None:
            dates_for_csv = [pd.Timestamp(t).strftime("%Y-%m-%d") for t in time_vals]

        city_series[city] = [float(v) for v in heat_vals]

        # Plot
        ax2.plot(
            ds_city[date_name], ds_city,
            marker="o", linestyle="-", linewidth=1.5,
            color=colors[city], label=city,
            markersize=4,
        )

    # --- Style the plot ---
    ax2.set_title(
        "Daily Maximum Heat Index — East China Cities\n"
        "2022-07-01 to 2022-07-07 (ERA5 nearest grid point)",
        fontsize=13,
    )
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Daily Maximum Heat Index (°C)")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    fig2.autofmt_xdate()

    fig2.tight_layout()
    fig2.savefig(output_fig2, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure: {output_fig2}")
    plt.close(fig2)

    # ======================================================================
    # CSV export — wide format: date, Nanjing, Shanghai, Hangzhou, Huai'an
    # ======================================================================

    df = pd.DataFrame({"date": dates_for_csv})
    for city in cities:
        df[city] = city_series[city]

    df.to_csv(output_csv, index=False)
    print(f"Saved CSV:   {output_csv}")
    print(f"  Rows: {len(df)}  |  Cities: {len(cities)}")


if __name__ == "__main__":
    main()

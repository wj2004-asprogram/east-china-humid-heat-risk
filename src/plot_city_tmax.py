"""
plot_city_tmax.py

Extract daily maximum 2m temperature at the nearest ERA5 grid point
for four East China cities, plot a comparison time series, and export
a CSV table.

Cities:
    - Nanjing   (32.06°N, 118.79°E)
    - Shanghai  (31.23°N, 121.47°E)
    - Hangzhou  (30.27°N, 120.15°E)
    - Huai'an   (33.61°N, 119.02°E)

Usage:
    python src/plot_city_tmax.py
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


def detect_time_name(ds: xr.Dataset) -> str:
    """
    Detect the time coordinate name.

    ERA5 data may use "time" (classic CDS API) or "valid_time" (CDS Beta).
    """
    for candidate in ["time", "valid_time"]:
        if candidate in ds.coords:
            return candidate
    raise KeyError(
        f"Cannot find a time coordinate (tried 'time', 'valid_time'). "
        f"Available coords: {list(ds.coords)}"
    )


def detect_temperature_var(ds: xr.Dataset, lat_name: str, lon_name: str) -> str:
    """
    Auto-detect the daily maximum temperature variable.

    Priority order: daily_tmax, tmax, t2m.  Falls back to the first data
    variable that has both latitude and longitude dimensions.
    """
    candidates = ["daily_tmax", "tmax", "t2m"]

    for var_name in candidates:
        if var_name in ds.data_vars:
            da = ds[var_name]
            if lat_name in da.dims and lon_name in da.dims:
                return var_name

    # Fallback: first spatial data variable
    for var_name in ds.data_vars:
        da = ds[var_name]
        if lat_name in da.dims and lon_name in da.dims:
            return var_name

    raise ValueError(
        f"No suitable spatial data variable found. "
        f"Data vars: {list(ds.data_vars)}"
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
    input_file = project_root / "data" / "processed" / "daily_tmax_2022_07_01_07.nc"
    output_fig = project_root / "figures" / "east_china_city_daily_tmax_2022_07_01_07.png"
    output_csv = project_root / "data" / "processed" / "city_daily_tmax_2022_07_01_07.csv"

    output_fig.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    print(f"Loading {input_file} ...")
    ds = xr.open_dataset(input_file)
    print(ds)

    # --- Auto-detect coordinates and variable ---
    lat_name = detect_lat_name(ds)
    lon_name = detect_lon_name(ds)
    time_name = detect_time_name(ds)
    var_name = detect_temperature_var(ds, lat_name, lon_name)

    print(f"Detected — lat: '{lat_name}', lon: '{lon_name}', "
          f"time: '{time_name}', var: '{var_name}'")

    # --- Extract city time series ---
    records = []  # for CSV

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {
        "Nanjing":  "#d62728",  # red
        "Shanghai": "#1f77b4",  # blue
        "Hangzhou": "#2ca02c",  # green
        "Huai'an":  "#ff7f0e",  # orange
    }

    for city, (lat, lon) in cities.items():
        # Select nearest grid point
        ds_city = ds.sel({
            lat_name: lat,
            lon_name: lon,
        }, method="nearest")

        # Report actual matched coordinates
        actual_lat = float(ds_city[lat_name].values)
        actual_lon = float(ds_city[lon_name].values)
        print(f"{city:>10s}: requested ({lat:.2f}°N, {lon:.2f}°E)  "
              f"→ matched ({actual_lat:.2f}°N, {actual_lon:.2f}°E)")

        # Extract the temperature time series
        da_city = ds_city[var_name]
        temps = da_city.values
        time_vals = da_city[time_name].values

        # Convert to pandas-friendly format for CSV
        for t, t_val in zip(time_vals, temps):
            records.append({
                "city": city,
                "time": pd.Timestamp(t),
                "tmax_C": float(t_val),
                "era5_lat": actual_lat,
                "era5_lon": actual_lon,
            })

        # Plot
        ax.plot(
            da_city[time_name], da_city,
            marker="o", linestyle="-", linewidth=1.5,
            color=colors[city], label=city,
            markersize=4,
        )

    # --- Style the plot ---
    ax.set_title(
        "Daily Maximum 2m Temperature — East China Cities\n"
        "2022-07-01 to 2022-07-07 (ERA5 nearest grid point)",
        fontsize=13,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Maximum 2m Temperature (°C)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(output_fig, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure: {output_fig}")
    plt.close(fig)

    # --- Save CSV ---
    df = pd.DataFrame(records)
    df = df.sort_values(["city", "time"]).reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved CSV:   {output_csv}")
    print(f"  Rows: {len(df)}  |  Cities: {df['city'].nunique()}")


if __name__ == "__main__":
    main()

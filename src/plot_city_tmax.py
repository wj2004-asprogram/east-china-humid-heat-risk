"""
plot_city_tmax.py

Extract daily maximum 2m temperature at the nearest ERA5 grid point for
the configured cities, plot a comparison time series, and export a CSV
table (long format: one row per city and day).

The CSV column `date` is the calendar day (YYYY-MM-DD) in the config's
day definition (UTC or local time, e.g. UTC+8), as in the other CSVs;
the run's run_info JSON file records which one.
"""

import matplotlib.pyplot as plt
import pandas as pd

from src.config import RunConfig
from src.era5_utils import (
    detect_lat_coord,
    detect_lon_coord,
    detect_t2m_var,
    detect_time_coord,
    load_dataset,
    select_city_points,
)


def run(cfg: RunConfig) -> None:
    input_file = cfg.paths.daily_tmax_nc
    output_fig = cfg.paths.fig_city_tmax
    output_csv = cfg.paths.city_tmax_csv

    output_fig.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    print(f"Loading {input_file} ...")
    ds = load_dataset(input_file)

    lat_name = detect_lat_coord(ds)
    lon_name = detect_lon_coord(ds)
    time_name = detect_time_coord(ds)
    var_name = detect_t2m_var(ds)

    print(f"Detected — lat: '{lat_name}', lon: '{lon_name}', "
          f"time: '{time_name}', var: '{var_name}'")

    # --- Extract city time series ---
    records = []  # for CSV

    fig, ax = plt.subplots(figsize=(12, 6))

    for city, ds_city, actual_lat, actual_lon in select_city_points(
        ds, cfg.cities, lat_name, lon_name
    ):
        da_city = ds_city[var_name]
        temps = da_city.values
        time_vals = da_city[time_name].values

        for t, t_val in zip(time_vals, temps):
            records.append({
                "city": city.name,
                "date": pd.Timestamp(t).strftime("%Y-%m-%d"),
                "tmax_C": float(t_val),
                "era5_lat": actual_lat,
                "era5_lon": actual_lon,
            })

        ax.plot(
            da_city[time_name], da_city,
            marker="o", linestyle="-", linewidth=1.5,
            color=city.color, label=city.name,
            markersize=4,
        )

    # --- Style the plot ---
    ax.set_title(
        "Daily Maximum 2m Temperature — East China Cities\n"
        f"{cfg.period_label} (ERA5 nearest grid point)",
        fontsize=13,
    )
    ax.set_xlabel(f"Date ({cfg.day_definition} calendar day)")
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
    df = df.sort_values(["city", "date"]).reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved CSV:   {output_csv}")
    print(f"  Rows: {len(df)}  |  Cities: {df['city'].nunique()}")

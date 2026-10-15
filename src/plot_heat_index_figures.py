"""
plot_heat_index_figures.py

Read the processed humid-heat-risk NetCDF file and produce:

1. A regional-mean daily maximum heat index time series for East China.
2. A city comparison of daily maximum heat index at the nearest ERA5
   grid point.

Also exports a CSV table of the city-level daily heat index values
(wide format: date, one column per city).
"""

import matplotlib.pyplot as plt
import pandas as pd

from src.config import RunConfig
from src.era5_utils import (
    detect_date_dim,
    detect_lat_coord,
    detect_lon_coord,
    load_dataset,
    require_var,
    select_city_points,
)


def run(cfg: RunConfig) -> None:
    input_file = cfg.paths.humid_heat_nc
    output_fig1 = cfg.paths.fig_mean_heat_index
    output_fig2 = cfg.paths.fig_city_heat_index
    output_csv = cfg.paths.city_heat_index_csv

    output_fig1.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    print(f"Loading {input_file} ...")
    ds = load_dataset(input_file)

    lat_name = detect_lat_coord(ds)
    lon_name = detect_lon_coord(ds)

    var_name = "daily_max_heat_index_c"
    da = require_var(ds, var_name)
    date_name = detect_date_dim(da)

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
        f"{cfg.period_label}",
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
    # Figure 2 — City daily maximum heat index comparison
    # ======================================================================
    city_series = {}   # city name -> list of values
    dates_for_csv = None

    fig2, ax2 = plt.subplots(figsize=(12, 6))

    for city, ds_city, _, _ in select_city_points(da, cfg.cities, lat_name, lon_name):
        time_vals = ds_city[date_name].values
        heat_vals = ds_city.values

        if dates_for_csv is None:
            dates_for_csv = [pd.Timestamp(t).strftime("%Y-%m-%d") for t in time_vals]

        city_series[city.name] = [float(v) for v in heat_vals]

        ax2.plot(
            ds_city[date_name], ds_city,
            marker="o", linestyle="-", linewidth=1.5,
            color=city.color, label=city.name,
            markersize=4,
        )

    ax2.set_title(
        "Daily Maximum Heat Index — East China Cities\n"
        f"{cfg.period_label} (ERA5 nearest grid point)",
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
    # CSV export — wide format: date, one column per city
    # ======================================================================
    df = pd.DataFrame({"date": dates_for_csv})
    for city in cfg.cities:
        df[city.name] = city_series[city.name]

    df.to_csv(output_csv, index=False)
    print(f"Saved CSV:   {output_csv}")
    print(f"  Rows: {len(df)}  |  Cities: {len(cfg.cities)}")

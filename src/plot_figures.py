"""
plot_figures.py

Time series of the regional-mean daily maximum 2m temperature over
East China.
"""

import matplotlib.pyplot as plt
from src.config import RunConfig
from src.era5_utils import (
    detect_lat_coord,
    detect_lon_coord,
    detect_t2m_var,
    detect_time_coord,
    load_dataset,
)
from src.plot_utils import break_at_gaps, line_style


def run(cfg: RunConfig) -> None:
    input_file = cfg.paths.daily_tmax_nc
    output_file = cfg.paths.fig_mean_tmax
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # --- Load data (daily Tmax in °C, written by process_temperature) ---
    print(f"Loading {input_file} ...")
    ds = load_dataset(input_file)

    var_name = detect_t2m_var(ds)
    time_name = detect_time_coord(ds)
    lat_name = detect_lat_coord(ds)
    lon_name = detect_lon_coord(ds)
    print(f"Using variable: '{var_name}', time coordinate: '{time_name}'")

    # --- Regional mean: one value per day ---
    regional_mean = ds[var_name].mean(dim=[lat_name, lon_name])
    n_days = regional_mean.sizes[time_name]

    # --- Plot (lines break between seasons) ---
    fig, ax = plt.subplots(figsize=(10, 5))

    regional_mean = break_at_gaps(regional_mean, time_name)
    ax.plot(regional_mean[time_name], regional_mean,
            linestyle="-", color="darkred", **line_style(n_days))

    ax.set_title(f"East China Mean Daily Tmax, {cfg.period_label}",
                 fontsize=13)
    ax.set_xlabel(f"Date ({cfg.day_definition} calendar day)")
    ax.set_ylabel("Daily Maximum 2m Temperature (°C)")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close(fig)

"""
plot_figures.py

Visualise ERA5 daily maximum temperature for East China.

Current functionality:
    - Time series of regional-mean daily Tmax over East China.

Usage:
    python src/plot_figures.py
"""

import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path


def detect_temperature_variable(ds: xr.Dataset) -> str:
    """
    Auto-detect a daily maximum 2m temperature variable in the dataset.

    Looks for variable names containing 't2m' (ERA5 2m temperature),
    preferring variables with 'max' in the name. Falls back to the
    first data variable with units hinting at temperature.

    Parameters
    ----------
    ds : xr.Dataset
        Opened dataset.

    Returns
    -------
    str
        Name of the detected temperature variable.
    """
    # Priority candidates: common ERA5 2m temperature variable names
    candidates = ["t2m_max", "tmax", "mx2t", "t2m"]

    for var_name in candidates:
        if var_name in ds.data_vars:
            da = ds[var_name]
            # Check that it's a spatial field, not a scalar
            if "latitude" in da.dims and "longitude" in da.dims:
                return var_name

    # Fallback: scan all data variables for temperature-like names
    for var_name in ds.data_vars:
        lower = var_name.lower()
        if "t2m" in lower or "temp" in lower or "tmax" in lower:
            da = ds[var_name]
            if "latitude" in da.dims and "longitude" in da.dims:
                return var_name

    # Last resort: return the first spatial data variable
    for var_name in ds.data_vars:
        da = ds[var_name]
        if "latitude" in da.dims and "longitude" in da.dims:
            return var_name

    raise ValueError("No suitable spatial data variable found in dataset.")


def detect_time_coord(ds: xr.Dataset) -> str:
    """
    Detect the time coordinate name in the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Opened dataset.

    Returns
    -------
    str
        Name of the time coordinate.
    """
    # Common names in order of preference
    candidates = ["valid_time", "time", "forecast_time"]
    for name in candidates:
        if name in ds.coords:
            return name

    raise KeyError(f"No time coordinate detected. Available coords: {list(ds.coords)}")


def main():
    # --- Paths ---
    project_root = Path(__file__).resolve().parent.parent
    input_file = project_root / "data" / "processed" / "daily_tmax_2022_07_01_07.nc"
    output_dir = project_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "east_china_mean_daily_tmax_2022_07_01_07.png"

    # --- Load data ---
    print(f"Loading {input_file} ...")
    ds = xr.open_dataset(input_file)
    print(ds)

    # --- Detect variable and time coordinate ---
    var_name = detect_temperature_variable(ds)
    time_name = detect_time_coord(ds)
    print(f"Using variable: '{var_name}', time coordinate: '{time_name}'")

    # --- Calculate spatial mean ---
    # Average over latitude and longitude to get a single value per time step
    regional_mean = ds[var_name].mean(dim=["latitude", "longitude"])

    # Convert from Kelvin to Celsius if needed (ERA5 t2m is in Kelvin)
    # Check units attribute or typical value range to decide
    mean_val = float(regional_mean.mean())
    if mean_val > 100:  # likely Kelvin (typical summer T ~ 300 K)
        regional_mean = regional_mean - 273.15
        unit_label = "°C"
    else:
        unit_label = "K"

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(regional_mean[time_name], regional_mean,
            marker="o", linestyle="-", linewidth=1.5, color="darkred")

    ax.set_title("East China Mean Daily Tmax, 2022-07-01 to 2022-07-07",
                 fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Maximum 2m Temperature (°C)")
    ax.grid(True, alpha=0.3)

    # Format date axis
    fig.autofmt_xdate()

    # --- Save ---
    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.close(fig)


if __name__ == "__main__":
    main()

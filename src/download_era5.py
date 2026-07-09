"""
download_era5.py

Download a small sample of ERA5 hourly single-level data for East China
from the Copernicus Climate Data Store (CDS).

Before running:
    1. Register at https://cds.climate.copernicus.eu/
    2. Set up your CDS API key (see https://cds.climate.copernicus.eu/api-how-to)
    3. Install cdsapi: pip install cdsapi

Usage:
    python src/download_era5.py
"""

import cdsapi
from pathlib import Path


def main():
    # --- Configuration ---
    # Region: East China (~112°E–123°E, ~25°N–36°N)
    area = [36, 112, 25, 123]  # [North, West, South, East]

    # Test sample: July 2022, first 7 days, all 24 hours
    year = "2022"
    month = "07"
    days = [f"{d:02d}" for d in range(1, 8)]       # 01 to 07
    times = [f"{h:02d}:00" for h in range(24)]     # 00:00 to 23:00

    # Variables
    variables = [
        "2m_temperature",
        "2m_dewpoint_temperature",
    ]

    # Output path
    output_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "era5_east_china_2022_07_01_07.nc"

    # --- Build the CDS API request ---
    client = cdsapi.Client()

    request = {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": variables,
        "year": year,
        "month": month,
        "day": days,
        "time": times,
        "area": area,
    }

    print("Downloading ERA5 sample for East China…")
    print(f"  Variables: {variables}")
    print(f"  Region:    N={area[0]}°, W={area[1]}°, S={area[2]}°, E={area[3]}°")
    print(f"  Period:    {year}-{month}-01 to {year}-{month}-07")
    print(f"  Output:    {output_file}")

    client.retrieve(
        "reanalysis-era5-single-levels",
        request,
        str(output_file),
    )

    print("Download complete.")


if __name__ == "__main__":
    main()

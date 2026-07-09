from pathlib import Path

import cdsapi


OUTPUT_FILE = Path("data/raw/era5_east_china_2022_07.nc")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

if OUTPUT_FILE.exists():
    print(f"Output file already exists: {OUTPUT_FILE}")
    print("Delete it first if you want to download again.")
    raise SystemExit(0)


client = cdsapi.Client()

request = {
    "product_type": ["reanalysis"],
    "variable": [
        "2m_temperature",
        "2m_dewpoint_temperature",
    ],
    "year": ["2022"],
    "month": ["07"],
    "day": [f"{day:02d}" for day in range(1, 32)],
    "time": [f"{hour:02d}:00" for hour in range(24)],
    "area": [36, 112, 25, 123],
    "data_format": "netcdf",
    "download_format": "unarchived",
}

print("Downloading ERA5 July 2022 data...")
print(f"Output file: {OUTPUT_FILE}")

client.retrieve(
    "reanalysis-era5-single-levels",
    request,
    str(OUTPUT_FILE),
)

print("Download completed.")


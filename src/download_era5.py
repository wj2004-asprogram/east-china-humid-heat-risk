"""
download_era5.py

Download ERA5 hourly single-level 2m temperature and 2m dewpoint
temperature for the configured region and period from the Copernicus
Climate Data Store (CDS).

Before running:
    1. Register at https://cds.climate.copernicus.eu/
    2. Set up your CDS API key (see https://cds.climate.copernicus.eu/how-to-api)
    3. Accept the ERA5 licence on the dataset page.
"""

from datetime import timedelta

from src.config import RunConfig


VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
]


def build_request(cfg: RunConfig) -> dict:
    """
    Build the CDS API request for the configured period and region.

    Only periods within a single calendar month are supported for now:
    the CDS combines the year/month/day lists, so a multi-month request
    would also fetch days outside the period. Multi-month downloads
    (one request per month) are planned for phase 2.
    """
    if (cfg.start_date.year, cfg.start_date.month) != (cfg.end_date.year, cfg.end_date.month):
        raise NotImplementedError(
            f"Period {cfg.period_label} spans several months; "
            "multi-month downloads are not supported yet (phase 2)."
        )

    n_days = (cfg.end_date - cfg.start_date).days + 1
    days = [(cfg.start_date + timedelta(days=i)).strftime("%d") for i in range(n_days)]

    return {
        "product_type": ["reanalysis"],
        "variable": VARIABLES,
        "year": [f"{cfg.start_date.year:04d}"],
        "month": [f"{cfg.start_date.month:02d}"],
        "day": days,
        "time": [f"{hour:02d}:00" for hour in range(24)],  # all 24 UTC hours
        "area": cfg.cds_area,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def run(cfg: RunConfig) -> None:
    """Download the raw ERA5 file, unless it already exists."""
    import cdsapi  # imported here so the other steps run without cdsapi

    output_file = cfg.paths.raw_era5
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.exists():
        print(f"Output file already exists: {output_file}")
        print("Delete it first if you want to download again.")
        return

    request = build_request(cfg)

    print(f"Downloading ERA5 for {cfg.period_label} ...")
    print(f"  Variables: {VARIABLES}")
    print(f"  Region:    N={cfg.north}°, W={cfg.west}°, S={cfg.south}°, E={cfg.east}°")
    print(f"  Output:    {output_file}")

    # Download to a temporary name and rename only on success, so an
    # interrupted download never leaves a truncated file at the final path.
    partial_file = output_file.with_name(output_file.name + ".part")
    partial_file.unlink(missing_ok=True)

    client = cdsapi.Client()
    client.retrieve("reanalysis-era5-single-levels", request, str(partial_file))
    partial_file.replace(output_file)

    print("Download completed.")

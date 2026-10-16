"""
run_info.py

Write a small JSON file next to the run's CSV tables that records how
their dates are defined: UTC calendar days or local calendar days (e.g.
UTC+8, Beijing time). The CSVs themselves only hold the dates
(YYYY-MM-DD), so without this file a Beijing-time table and a UTC table
look alike.
"""

import json

from src.config import RunConfig


def build_run_info(cfg: RunConfig) -> dict:
    """Run metadata derived from the config only (no time stamps, so reruns give the same file)."""
    return {
        "config": cfg.name,
        "period": {"start": cfg.start_date.isoformat(), "end": cfg.end_date.isoformat()},
        "day_definition": cfg.day_definition,
        "utc_offset_hours": cfg.utc_offset_hours,
        "day_window": f"day D runs from {cfg.day_window_utc} (24 hourly ERA5 time steps)",
        "date_column": (
            f"the 'date' column of each CSV below is a calendar day in {cfg.day_definition}, "
            "formatted YYYY-MM-DD"
        ),
        "csv_files": [
            cfg.paths.city_tmax_csv.name,
            cfg.paths.city_heat_index_csv.name,
            cfg.paths.area_fraction_csv.name,
        ],
        "land_sea_mask_threshold": cfg.land_sea_mask_threshold,
    }


def run(cfg: RunConfig) -> None:
    output_path = cfg.paths.run_info_json
    output_path.parent.mkdir(parents=True, exist_ok=True)

    info = build_run_info(cfg)
    output_path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Day definition: {info['day_definition']} ({info['day_window']})")
    print(f"Saved: {output_path}")

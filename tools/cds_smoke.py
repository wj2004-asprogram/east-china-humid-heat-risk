"""One hour, original variables/region, outside raw/. Safe to restart this CLI.

No other task is submitted. A pending smoke job retains its ID across invocations.
"""

import argparse

import pandas as pd
import xarray as xr

from src.cds_download import Downloader, StateStore
from src.config import cds_state_path, load_config
from src.download_era5 import DATASET, build_raw_file_request
from src.era5_utils import detect_d2m_var, detect_t2m_var, detect_time_coord


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    raw = cfg.raw_files[0]
    request = build_raw_file_request(cfg, raw)
    request["day"], request["time"] = [request["day"][0]], [request["time"][0]]
    state_path = cds_state_path(raw.path)
    target = state_path.parent / "smoke" / "one_hour.nc"

    def validate(path):
        with xr.open_dataset(path) as ds:
            detect_t2m_var(ds)
            detect_d2m_var(ds)
            times = pd.DatetimeIndex(ds[detect_time_coord(ds)].values)
            expected = pd.Timestamp(f"{request['year'][0]}-{request['month'][0]}-{request['day'][0]} {request['time'][0]}")
            if not times.equals(pd.DatetimeIndex([expected])):
                raise ValueError("Smoke file does not contain exactly the requested hour")
    Downloader(StateStore(state_path)).retrieve(DATASET, request, target,
                                               {"progress": False}, validator=validate)
    print(f"SMOKE PASSED: {target}", flush=True)


if __name__ == "__main__":
    main()

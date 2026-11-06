"""Hourly-derived daily and completed-night features; no fitting or evaluation."""

import numpy as np
import pandas as pd
import xarray as xr

from src.era5_utils import detect_time_coord, detect_t2m_var, detect_d2m_var
from src.indices import kelvin_to_celsius, relative_humidity, heat_index_rothfusz
from src.probability_samples import calendar_dates, required_history, sample_id, year_split
from src.time_utils import local_daily


DAILY_SPECS = {
    "tmax_c": ("temperature_c", "max"), "tmin_c": ("temperature_c", "min"),
    "tmean_c": ("temperature_c", "mean"), "tdmean_c": ("dewpoint_c", "mean"),
    "tdmax_c": ("dewpoint_c", "max"), "rhmean_pct": ("rh_pct", "mean"),
    "himax_c": ("hi_c", "max"),
}
NIGHT_SPECS = {
    "night_tmean_c": ("temperature_c", "mean"),
    "night_tmin_c": ("temperature_c", "min"),
    "night_tdmean_c": ("dewpoint_c", "mean"),
    "night_rhmean_pct": ("rh_pct", "mean"),
}


def checked_hours(hourly):
    """UTC stamps must be unique whole hours, without NaT or timezone labels."""
    name = detect_time_coord(hourly)
    times = pd.DatetimeIndex(hourly[name].values)
    if times.hasnans or times.tz is not None or times.has_duplicates or not times.equals(times.floor("h")):
        raise ValueError("Expected unique finite whole-hour UTC timestamps")
    return name, times


def hourly_fields(hourly):
    """Compute physical variables from Kelvin inputs, retaining missing values."""
    checked_hours(hourly)
    t, td = hourly[detect_t2m_var(hourly)], hourly[detect_d2m_var(hourly)]
    for field in (t, td):
        if str(field.attrs.get("units", "")).lower() not in ("k", "kelvin"):
            raise ValueError("Hourly temperature and dewpoint must declare Kelvin units")
    t = kelvin_to_celsius(t.where(np.isfinite(t)))
    td = kelvin_to_celsius(td.where(np.isfinite(td)))
    rh = relative_humidity(t, td)
    fields = xr.Dataset({"temperature_c": t, "dewpoint_c": td, "rh_pct": rh,
                         "hi_c": heat_index_rothfusz(t, rh)})
    for name in fields:
        fields[name].attrs["units"] = "%" if name.endswith("pct") else "degC"
    return fields


def daily_features(hourly, start_date, end_date):
    """Aggregate complete BJT days; a missing stamp raises, any NaN propagates."""
    time_name, _ = checked_hours(hourly)
    start, end = calendar_dates([start_date])[0], calendar_dates([end_date])[0]
    if end < start:
        raise ValueError("end_date precedes start_date")
    fields = hourly_fields(hourly)
    result = xr.Dataset({name: local_daily(fields[source], time_name, 8, how,
                                           start.date(), end.date())
                         for name, (source, how) in DAILY_SPECS.items()})
    result = result.rename({time_name: "date"})
    result.attrs.update(day_definition="BJT [00:00, next 00:00), UTC+08:00",
                        missingness="24 unique hours required; skipna=False",
                        qualification="HOURLY_DERIVED_FEATURE_PILOT_NOT_MODEL_RESULTS")
    for name in result:
        result[name].attrs["units"] = "%" if name.endswith("pct") else "degC"
    return result


def night_window(target, cutoff_lag_days):
    """D-lag-1 20 BJT through D-lag 08 BJT, upper boundary excluded."""
    day = calendar_dates([target])[0]
    required_history(day, cutoff_lag_days)
    end_bjt = day - pd.Timedelta(days=cutoff_lag_days) + pd.Timedelta(hours=8)
    start_bjt = end_bjt - pd.Timedelta(hours=12)
    return start_bjt - pd.Timedelta(hours=8), end_bjt - pd.Timedelta(hours=8)


def completed_night_features(hourly, target, cutoff_lag_days):
    """Use exactly 12 ended-night hours; no target morning or D-1 in D-2 mode."""
    time_name, times = checked_hours(hourly)
    start, end = night_window(target, cutoff_lag_days)
    expected = pd.date_range(start, end, freq="h", inclusive="left")
    if not expected.isin(times).all():
        raise ValueError("Incomplete 12-hour completed-night window")
    selected = hourly.sel({time_name: expected})
    fields = hourly_fields(selected)
    result = xr.Dataset({name: getattr(fields[source], how)(dim=time_name, skipna=False)
                         for name, (source, how) in NIGHT_SPECS.items()})
    for name in result:
        result[name].attrs["units"] = "%" if name.endswith("pct") else "degC"
    result.attrs.update(source_start_utc=start.isoformat()+"Z",
                        source_end_utc_exclusive=end.isoformat()+"Z",
                        source_last_hour_utc=(end-pd.Timedelta(hours=1)).isoformat()+"Z",
                        hour_count=12, cutoff_lag_days=cutoff_lag_days)
    return result


def assemble_pilot_samples(daily, hourly, targets, points):
    """Small, explicit grid/date/lag rows; target and previous labels omitted.

    Points are exact (latitude, longitude) pairs. Lag columns mean true calendar
    offsets, e.g. lag2 mode has columns day_lag2 through day_lag8. These columns
    are sparse across modes, so finite_feature_inputs refers only to active lags.
    """
    dates = calendar_dates(daily.date.values)
    if len(set(points)) != len(points):
        raise ValueError("Duplicate pilot grid points")
    rows = []
    for day in calendar_dates(targets):
        if year_split(day.year) != "train":
            raise ValueError("Pilot restricted to training years")
        for latitude, longitude in points:
            grid_id = f"lat{float(latitude).hex()}_lon{float(longitude).hex()}"
            point_daily = daily.sel(latitude=latitude, longitude=longitude)
            point_hourly = hourly.sel(latitude=latitude, longitude=longitude)
            for lag in (1, 2):
                history = required_history(day, lag)
                if not history.isin(dates).all():
                    raise ValueError("Incomplete seven-day daily feature history")
                night = completed_night_features(point_hourly, day, lag)
                issue = day - pd.Timedelta(hours=8)
                source_start = history[0] - pd.Timedelta(hours=8)
                source_end = history[-1] + pd.Timedelta(hours=16)
                row = {"sample_id": sample_id(grid_id, day), "grid_id": grid_id,
                       "target_date": day.date().isoformat(), "split": "train",
                       "latitude": latitude, "longitude": longitude, "cutoff_lag_days": lag,
                       "issue_utc": issue.isoformat()+"Z",
                       "target_start_bjt": day.isoformat()+"+08:00",
                       "target_end_bjt_exclusive": (day+pd.Timedelta(days=1)).isoformat()+"+08:00",
                       "feature_start_date": history[0].date().isoformat(),
                       "feature_end_date": history[-1].date().isoformat(),
                       "source_start_utc": source_start.isoformat()+"Z",
                       "source_end_utc_exclusive": source_end.isoformat()+"Z",
                       "night_start_utc": night.attrs["source_start_utc"],
                       "night_end_utc_exclusive": night.attrs["source_end_utc_exclusive"],
                       "night_last_hour_utc": night.attrs["source_last_hour_utc"]}
                features = {}
                for date in history:
                    offset = (day-date).days
                    for name in DAILY_SPECS:
                        features[f"{name}_day_lag{offset}"] = float(point_daily[name].sel(date=date).item())
                for name in NIGHT_SPECS:
                    features[name] = float(night[name].item())
                row.update(features)
                row["finite_feature_inputs"] = bool(np.isfinite(list(features.values())).all())
                rows.append(row)
    return pd.DataFrame(rows)

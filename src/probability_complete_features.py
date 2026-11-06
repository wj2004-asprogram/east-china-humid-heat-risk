"""Frozen 117-column feature registry and annual common-case assembly."""

import numpy as np
import pandas as pd
import xarray as xr

from src.probability_features import checked_hours, hourly_fields


DAILY = {
    "tmean_c": ("temperature_c", "mean"), "tmin_c": ("temperature_c", "min"),
    "tmax_c": ("temperature_c", "max"), "tdmean_c": ("dewpoint_c", "mean"),
    "tdmin_c": ("dewpoint_c", "min"), "tdmax_c": ("dewpoint_c", "max"),
    "rhmean_percent": ("rh_pct", "mean"), "rhmin_percent": ("rh_pct", "min"),
    "rhmax_percent": ("rh_pct", "max"), "himax_c": ("hi_c", "max"),
}
NIGHT = {"night_tmean_c": ("temperature_c", "mean"),
         "night_tmin_c": ("temperature_c", "min"),
         "night_tdmean_c": ("dewpoint_c", "mean"),
         "night_rhmean_percent": ("rh_pct", "mean")}


def complete_daily_night(hourly, year):
    """Exactly 92 BJT days, with night indexed by its morning's calendar date."""
    if year not in range(2000, 2020):
        raise ValueError("Feature extraction is restricted to 2000-2019")
    time_name, actual = checked_hours(hourly)
    dates = pd.date_range(f"{year}-06-01", f"{year}-08-31")
    expected = pd.date_range(dates[0]-pd.Timedelta(hours=8), periods=92*24, freq="h")
    if not expected.isin(actual).all():
        raise ValueError("Missing UTC hour in the requested complete BJT summer")
    selected = hourly.sel({time_name: expected})
    fields = hourly_fields(selected)
    shape = (92, 24, fields.sizes["latitude"], fields.sizes["longitude"])
    arrays = {name: fields[name].transpose(time_name, "latitude", "longitude").values.reshape(shape)
              for name in fields}
    values = {name: getattr(np, how)(arrays[source], axis=1) for name, (source, how) in DAILY.items()}
    for name, (source, how) in NIGHT.items():
        result = np.full((92, *shape[2:]), np.nan)
        # June 1's night predates available data and is not needed by June 9 targets.
        night = np.concatenate([arrays[source][:-1, 20:], arrays[source][1:, :8]], axis=1)
        result[1:] = getattr(np, how)(night, axis=1)
        values[name] = result
    night_t = np.concatenate([arrays["temperature_c"][:-1, 20:], arrays["temperature_c"][1:, :8]], axis=1)
    cooling = np.full((92, *shape[2:]), np.nan)
    cooling[1:] = np.where(np.isfinite(night_t).all(axis=1), night_t[:, 0]-night_t[:, -1], np.nan)
    values["night_cooling20to07_c"] = cooling
    result = xr.Dataset({name: (("date", "latitude", "longitude"), value)
                         for name, value in values.items()},
                        coords={"date": dates, "latitude": fields.latitude, "longitude": fields.longitude})
    for name in result:
        result[name].attrs["units"] = "%" if name.endswith("percent") else "degC"
    result.attrs.update(day_definition="UTC+8 complete 24-hour calendar day",
                        night_definition="[previous day 20:00, date 08:00) BJT; 12 hours",
                        night_cooling="T20 minus T07; any missing night hour makes cooling missing",
                        missingness="all reductions skipna=False; missing timestamp raises")
    return result


def land_arrays(daily, registry):
    return {name: daily[name].sel(latitude=xr.DataArray(registry.latitude.to_numpy(), dims="grid"),
                                  longitude=xr.DataArray(registry.longitude.to_numpy(), dims="grid")).values
            for name in daily}


def neighbor_statistics(values, source_lists, min_sources=3):
    """Fixed equal-weight source membership; no missing-source renormalization."""
    mean = np.full(values.shape, np.nan)
    std = np.full(values.shape, np.nan)
    for query, sources in enumerate(source_lists):
        if len(sources) >= min_sources:
            selected = values[:, sources]
            mean[:, query] = selected.mean(axis=1)
            std[:, query] = selected.std(axis=1, ddof=0)
    return mean, std


def feature_matrix(arrays, dates, registry, regional, lag):
    """Return [84 dates, grids, 117 features], names and feature-group membership."""
    if lag not in (1, 2):
        raise ValueError("lag must be 1 or 2")
    dates = pd.DatetimeIndex(dates)
    if len(dates) != 92 or not dates.equals(pd.date_range(dates[0], periods=92)) or dates[0].strftime("%m-%d") != "06-01":
        raise ValueError("Require one complete continuous JJA")
    target = np.arange(8, 92)
    anchor = target-lag
    grid_count = len(registry)
    columns, names, groups = [], [], []

    def add(name, group, values):
        columns.append(np.broadcast_to(values, (84, grid_count)))
        names.append(name)
        groups.append(group)

    add("latitude", "C", registry.latitude.to_numpy()[None, :])
    add("longitude", "C", registry.longitude.to_numpy()[None, :])
    days = dates[target].dayofyear.to_numpy() - dates[target].is_leap_year.astype(int)
    for harmonic, label in ((1, "annual"), (2, "semiannual")):
        for operation in ("sin", "cos"):
            add(f"{operation}_{label}_doy365", "C", getattr(np, operation)(2*np.pi*harmonic*days[:, None]/365))
    for group, variables in (("T", ["tmean_c", "tmin_c", "tmax_c"]),
                             ("M", ["tdmean_c", "tdmin_c", "tdmax_c", "rhmean_percent", "rhmin_percent", "rhmax_percent"]),
                             ("H", ["himax_c"])):
        for variable in variables:
            values = arrays[variable]
            for offset in (0, 1, 2):
                add(f"{variable}_anchor_minus{offset}", group, values[anchor-offset])
            for window in (3, 7):
                history = values[anchor[:, None]-np.arange(window)[None, :]]
                for statistic in ("mean", "min", "max"):
                    add(f"{variable}_last{window}_{statistic}", group, getattr(np, statistic)(history, axis=1))
    for name in ("night_tmean_c", "night_tmin_c", "night_cooling20to07_c"):
        add(name, "NT", arrays[name][anchor])
    for name in ("night_tdmean_c", "night_rhmean_percent"):
        add(name, "NM", arrays[name][anchor])
    for variable, group in (("tmean_c", "RT"), ("tdmean_c", "RM"), ("rhmean_percent", "RM")):
        mean, std = regional[variable]
        add(f"regional_{variable}_mean", group, mean[anchor])
        add(f"regional_{variable}_std_ddof0", group, std[anchor])
        for window in (3, 7):
            add(f"regional_{variable}_mean_last{window}", group,
                mean[anchor[:, None]-np.arange(window)[None, :]].mean(axis=1))
    hi = arrays["himax_c"]
    y = np.where(np.isfinite(hi), (hi >= 40.6).astype(float), np.nan)
    event_history = y[anchor[:, None]-np.arange(7)[None, :]]
    add("y_anchor", "H", y[anchor])
    for window in (3, 7):
        add(f"event_fraction_last{window}", "H", event_history[:, :window].mean(axis=1))
    duration = np.cumprod(event_history, axis=1).sum(axis=1)
    add("consecutive_event_days_capped7", "H", duration)
    return np.stack(columns, axis=-1).astype(np.float32), names, groups


def assemble_common_year(daily, registry, source_lists, *, strict=False):
    arrays = land_arrays(daily, registry)
    regional = {name: neighbor_statistics(arrays[name], source_lists)
                for name in ("tmean_c", "tdmean_c", "rhmean_percent")}
    x1, names, groups = feature_matrix(arrays, daily.date.values, registry, regional, 1)
    x2, names2, _ = feature_matrix(arrays, daily.date.values, registry, regional, 2)
    assert names == names2 and len(names) == 117
    hi = arrays["himax_c"]
    labels = np.where(np.isfinite(hi), (hi >= 40.6).astype(float), np.nan)
    geometry = np.array([len(sources) >= 3 for sources in source_lists])
    if strict:
        geometry &= registry.spatial_role.eq("train").to_numpy()
    valid_labels = np.isfinite(labels[8:]) & np.isfinite(labels[7:-1])
    valid1, valid2 = np.isfinite(x1).all(axis=2), np.isfinite(x2).all(axis=2)
    common = geometry[None, :] & valid_labels & valid1 & valid2
    day_index, grid_index = np.where(common)
    payload = {"X1": x1[common], "X2": x2[common], "day_index": day_index.astype(np.int16),
               "grid_index": grid_index.astype(np.int16), "y": labels[8:][common].astype(np.int8),
               "previous_y": labels[7:-1][common].astype(np.int8),
               "state1": labels[7:-1][common].astype(np.int8),
               "state2": labels[6:-2][common].astype(np.int8)}
    report = {"total_grid_dates": int(common.size), "geometry_eligible_grid_dates": int(geometry.sum()*84),
              "common_samples": int(common.sum()),
              "geometry_excluded": int((~geometry).sum()*84),
              "bad_labels_within_geometry": int((geometry[None, :] & ~valid_labels).sum()),
              "bad_lag1_features_within_geometry": int((geometry[None, :] & ~valid1).sum()),
              "bad_lag2_features_within_geometry": int((geometry[None, :] & ~valid2).sum()),
              "missing_counts_overlap": True}
    return payload, report, {"names": names, "groups": groups}

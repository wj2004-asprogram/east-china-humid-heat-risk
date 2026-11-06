"""Frozen probability baselines with auditable, JSON-serializable fitted state.

Inputs contain one row per (date, grid_id). Transition and persistence additionally
require state_lagN and state_date_lagN, where N is the explicitly selected lag.
No file I/O, automatic evaluation, tuning, or missing-bin fallback is performed.
"""

import numpy as np
import pandas as pd

from src.probability_samples import year_split


PRIOR_PSEUDOCOUNT = 1.0
SHRINKAGE_STRENGTH = 20.0
BIN_DAYS = 15


def _lag(lag):
    if lag not in (1, 2):
        raise ValueError("cutoff_lag_days must be 1 or 2")
    return int(lag)


def _frame(frame, *, fit=False, strict_spatial=False):
    required = {"date", "grid_id"} | ({"label", "spatial_role"} if fit else set())
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
    if frame.empty:
        raise ValueError("Empty sample frame")
    result = frame.copy()
    dates = pd.DatetimeIndex(result.date)
    if dates.hasnans or dates.tz is not None or not dates.equals(dates.normalize()):
        raise ValueError("date must contain finite local calendar dates")
    if not np.isin(dates.month, [6, 7, 8]).all():
        raise ValueError("Dates must be JJA")
    splits = [year_split(int(year)) for year in dates.year]
    if fit and any(split != "train" for split in splits):
        raise ValueError("Fit only accepts training years 2000-2014")
    result["date"] = dates
    if not result.grid_id.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("grid_id must be a nonempty string")
    if result.duplicated(["date", "grid_id"]).any():
        raise ValueError("Duplicate date/grid sample")
    if fit:
        if not result.spatial_role.isin(["train", "buffer", "holdout"]).all():
            raise ValueError("Unknown spatial role")
        if strict_spatial and not result.spatial_role.eq("train").all():
            raise ValueError("Strict spatial fit forbids buffer and holdout rows")
        if result.groupby("grid_id").spatial_role.nunique().gt(1).any():
            raise ValueError("Spatial role changes within a grid")
        _binary(result.label, "label")
    season_start = pd.to_datetime({"year": dates.year, "month": 6, "day": 1})
    result["season_bin"] = ((dates - pd.DatetimeIndex(season_start)).days // BIN_DAYS).astype(int)
    return result


def _binary(values, name):
    if values.isna().any() or not values.isin([0, 1]).all():
        raise ValueError(f"{name} must be finite binary 0/1; unknown is not zero")
    return values.astype(int)


def _states(frame, lag):
    state_name, date_name = f"state_lag{lag}", f"state_date_lag{lag}"
    if state_name not in frame or date_name not in frame:
        raise ValueError(f"Require {state_name} and {date_name}")
    dates = pd.DatetimeIndex(frame[date_name])
    expected = pd.DatetimeIndex(frame.date) - pd.Timedelta(days=lag)
    if not dates.equals(expected):
        raise ValueError(f"State date must equal target date minus {lag} days")
    if not np.isin(dates.month, [6, 7, 8]).all():
        raise ValueError("State must be in the same JJA")
    return _binary(frame[state_name], state_name)


def persistence_probabilities(frame, *, cutoff_lag_days):
    """Unsmoothed 0/1 forecast from the latest permitted observed state."""
    lag = _lag(cutoff_lag_days)
    rows = _frame(frame)
    return _states(rows, lag).to_numpy(dtype=float)


class ClimatologyBaseline:
    """Seasonal regional prior with optional per-grid shrinkage.

    Strict spatial mode always uses the training-region seasonal prior, including
    at unseen grids; it never uses grid-specific historical label statistics.
    Ordinary mode uses the same prior for unseen grids or grid/bin combinations.
    A seasonal bin unsupported by the fit raises at prediction time.
    """

    kind = "climatology"

    def __init__(self, *, cutoff_lag_days, strict_spatial=False):
        self.cutoff_lag_days = _lag(cutoff_lag_days)
        self.strict_spatial = bool(strict_spatial)
        self.state = None

    def fit(self, frame):
        rows = _frame(frame, fit=True, strict_spatial=self.strict_spatial)
        regional = []
        for season_bin, group in rows.groupby("season_bin", sort=True):
            k, n = int(group.label.sum()), len(group)
            regional.append({"season_bin": int(season_bin), "events": k, "samples": n,
                             "probability": (k+PRIOR_PSEUDOCOUNT)/(n+2*PRIOR_PSEUDOCOUNT)})
        grid_counts = []
        if not self.strict_spatial:
            for (grid_id, season_bin), group in rows.groupby(["grid_id", "season_bin"], sort=True):
                grid_counts.append({"grid_id": grid_id, "season_bin": int(season_bin),
                                    "events": int(group.label.sum()), "samples": len(group)})
        self.state = {"schema_version": 1, "kind": self.kind,
                      "cutoff_lag_days": self.cutoff_lag_days,
                      "strict_spatial": self.strict_spatial,
                      "train_years": sorted(int(y) for y in rows.date.dt.year.unique()),
                      "train_grid_ids": sorted(rows.grid_id.unique().tolist()),
                      "training_rows": len(rows), "season_bin_days": BIN_DAYS,
                      "season_origin": "June 1 = day 0; bin 6 = August 30-31",
                      "prior_pseudocount_each_class": PRIOR_PSEUDOCOUNT,
                      "shrinkage_strength": SHRINKAGE_STRENGTH,
                      "missing_bin_policy": "raise", "regional": regional,
                      "grid_counts": grid_counts}
        return self

    def to_state(self):
        """Return a detached JSON-compatible dict, including training provenance."""
        import copy
        if self.state is None:
            raise ValueError("Baseline is not fitted")
        return copy.deepcopy(self.state)

    def _prediction_rows(self, frame, lag):
        if self.state is None:
            raise ValueError("Baseline is not fitted")
        if _lag(lag) != self.state["cutoff_lag_days"]:
            raise ValueError("Prediction lag differs from fitted lag")
        rows = _frame(frame)
        priors = {item["season_bin"]: item["probability"] for item in self.state["regional"]}
        missing = set(rows.season_bin) - set(priors)
        if missing:
            raise ValueError(f"No training support for seasonal bin(s): {sorted(missing)}")
        return rows, priors

    def predict(self, frame, *, cutoff_lag_days):
        rows, priors = self._prediction_rows(frame, cutoff_lag_days)
        counts = {(item["grid_id"], item["season_bin"]): item for item in self.state["grid_counts"]}
        values = []
        for grid_id, season_bin in zip(rows.grid_id, rows.season_bin):
            prior = priors[season_bin]
            item = counts.get((grid_id, season_bin))
            values.append(prior if item is None else
                          (item["events"]+SHRINKAGE_STRENGTH*prior)/(item["samples"]+SHRINKAGE_STRENGTH))
        return np.asarray(values)


class TransitionBaseline(ClimatologyBaseline):
    """Direct lag-specific P(y(D)=1 | season bin, y(D-lag)) from training rows."""

    kind = "conditional_transition"

    def fit(self, frame):
        rows = _frame(frame, fit=True, strict_spatial=self.strict_spatial)
        rows["state"] = _states(rows, self.cutoff_lag_days)
        super().fit(frame)
        # Transitions are pooled regionally, never grid-specific.
        self.state["grid_counts"] = []
        self.state["state_source_column"] = f"state_lag{self.cutoff_lag_days}"
        self.state["state_date_column"] = f"state_date_lag{self.cutoff_lag_days}"
        self.state["transition_counts"] = [
            {"season_bin": int(season_bin), "state": int(state),
             "events": int(group.label.sum()), "samples": len(group)}
            for (season_bin, state), group in rows.groupby(["season_bin", "state"], sort=True)]
        self.state["missing_state_support_policy"] = "training regional seasonal prior"
        return self

    def predict(self, frame, *, cutoff_lag_days):
        rows, priors = self._prediction_rows(frame, cutoff_lag_days)
        states = _states(rows, self.state["cutoff_lag_days"])
        counts = {(item["season_bin"], item["state"]): item for item in self.state["transition_counts"]}
        values = []
        for season_bin, state in zip(rows.season_bin, states):
            item = counts.get((season_bin, state), {"events": 0, "samples": 0})
            values.append((item["events"]+SHRINKAGE_STRENGTH*priors[season_bin])/
                          (item["samples"]+SHRINKAGE_STRENGTH))
        return np.asarray(values)

"""Paired annual probability evaluation and whole-date block uncertainty."""
import hashlib

import numpy as np

from src.probability_metrics import probability_scores, reliability_counts
from src.probability_phase4_calibration import validate_probabilities


def score_probability(y, p, latitude, clip=1e-6):
    result = probability_scores(y, p, latitude)
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    result["brier_coslat"] = result.get("brier_coslat")
    q = np.clip(p, clip, 1 - clip)
    result["log_loss_clip_1e_6"] = float(np.mean(-y * np.log(q) - (1 - y) * np.log1p(-q))) if len(y) else None
    return result


def risk_masks(y, previous_y, longitude=None, latitude=None, lon_boundary=117.5, lat_boundary=30.5):
    y, previous_y = np.asarray(y), np.asarray(previous_y)
    if y.shape != previous_y.shape or not np.isin(previous_y, [0, 1]).all():
        raise ValueError("Previous-day evaluation labels must align and be binary")
    masks = {"all": np.ones(y.shape, dtype=bool), "onset": previous_y == 0,
             "persistence_or_clearance": previous_y == 1}
    for before, after, name in [(0, 0, "0_to_0"), (0, 1, "0_to_1_onset"),
                                (1, 0, "1_to_0_clearance"), (1, 1, "1_to_1_continuation")]:
        masks[name] = (previous_y == before) & (y == after)
    if longitude is not None and latitude is not None:
        east, north = np.asarray(longitude) >= lon_boundary, np.asarray(latitude) >= lat_boundary
        for name, mask in [("southwest", ~east & ~north), ("southeast", east & ~north),
                           ("northwest", ~east & north), ("northeast", east & north)]:
            masks[name] = mask
    return masks


def alert_metrics(y, p, threshold):
    y, p = np.asarray(y), validate_probabilities(p)
    if y.shape != p.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("Alert labels must align and be binary")
    alert, positive = p >= threshold, y == 1
    tp, fp = int((alert & positive).sum()), int((alert & ~positive).sum())
    fn, tn = int((~alert & positive).sum()), int((~alert & ~positive).sum())
    ratio = lambda a, b: a / b if b else None
    return {"threshold": float(threshold), "n": len(y), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": ratio(tp, tp + fn), "precision": ratio(tp, tp + fp),
            "far": ratio(fp, tp + fp), "fpr": ratio(fp, fp + tn),
            "miss_rate": ratio(fn, tp + fn), "alert_fraction": ratio(tp + fp, len(y))}


def grid_alert_days(grid_index, p, threshold, all_grid_indices=None):
    grid_index, p = np.asarray(grid_index), validate_probabilities(p)
    if grid_index.shape != p.shape:
        raise ValueError("Grid and probability arrays must align")
    grids = np.unique(grid_index) if all_grid_indices is None else np.asarray(all_grid_indices)
    if not np.issubdtype(grid_index.dtype, np.integer) or (grid_index < 0).any():
        raise ValueError("Grid indices must be nonnegative integers")
    size = int(max(grid_index.max(initial=0), grids.max(initial=0))) + 1
    counts = np.bincount(grid_index, minlength=size)
    alerts = np.bincount(grid_index[p >= threshold], minlength=size)
    return [{"grid_index": int(g), "valid_days": int(counts[g]),
             "alert_days": int(alerts[g])} for g in grids]


def daily_brier_sufficient(y, p, day_index, mask=None, days_per_year=84, weights=None):
    y, p = np.asarray(y), validate_probabilities(p)
    day_index = np.asarray(day_index)
    if y.shape != p.shape or day_index.shape != p.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("Daily statistic inputs must align with binary labels")
    if not np.issubdtype(day_index.dtype, np.integer) or ((day_index < 0) | (day_index >= days_per_year)).any():
        raise ValueError("Day indices must cover fixed season positions")
    mask = np.ones(len(y), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    weights = np.ones(len(y), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    if mask.shape != p.shape or weights.shape != p.shape:
        raise ValueError("Daily mask/weights must align")
    return np.stack([np.bincount(day_index[mask], weights=((p-y)**2 * weights)[mask], minlength=days_per_year),
                     np.bincount(day_index[mask], weights=weights[mask], minlength=days_per_year)], axis=-1)


def bootstrap_draws(n_years=5, days_per_year=84, block_days=14, replicates=2000, seed=20261105):
    """Paired year draws and noncircular blocks; all model pairs reuse this plan."""
    if not 0 < block_days <= days_per_year or n_years < 1 or replicates < 1:
        raise ValueError("Invalid bootstrap dimensions")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, block_days])))
    blocks = (days_per_year + block_days - 1) // block_days
    years = np.empty((replicates, n_years), dtype=np.int64)
    starts = np.empty((replicates, n_years, blocks), dtype=np.int64)
    for replicate in range(replicates):
        years[replicate] = rng.integers(0, n_years, size=n_years)
        for slot in range(n_years):
            starts[replicate, slot] = rng.integers(0, days_per_year-block_days+1, size=blocks)
    dates = (starts[..., None] + np.arange(block_days)).reshape(replicates, n_years, -1)[..., :days_per_year]
    digest = hashlib.sha256(years.astype("<i8").tobytes() + dates.astype("<i8").tobytes()).hexdigest()
    return {"years": years, "dates": dates, "sha256": digest, "block_days": block_days,
            "replicates": replicates, "seed": seed}


def resampled_brier(daily_stats, draws):
    """Exact sum/count equivalent of resampling every grid row on sampled dates.

    A sampled year with an empty risk set invalidates that replicate. It is not
    silently dropped/replaced, which would change equal-year weighting.
    """
    daily_stats = np.asarray(daily_stats, dtype=float)
    if daily_stats.ndim != 3 or daily_stats.shape[-1] != 2:
        raise ValueError("Expected [year, day, (error_sum, count)]")
    totals = daily_stats[draws["years"][..., None], draws["dates"]].sum(axis=2)
    annual = np.divide(totals[..., 0], totals[..., 1], out=np.full(totals.shape[:2], np.nan), where=totals[..., 1] > 0)
    return np.mean(annual, axis=1)


def annual_brier(daily_stats):
    totals = np.asarray(daily_stats).sum(axis=1)
    return np.divide(totals[:, 0], totals[:, 1], out=np.full(len(totals), np.nan), where=totals[:, 1] > 0)


def comparison_point(candidate, reference):
    candidate, reference = np.asarray(candidate), np.asarray(reference)
    c, r = float(np.mean(candidate)), float(np.mean(reference))
    return {"candidate_brier": c if np.isfinite(c) else None, "reference_brier": r if np.isfinite(r) else None,
            "delta_brier": c-r if np.isfinite(c+r) else None,
            "brier_skill": 1-c/r if np.isfinite(c+r) and r > 0 else None,
            "years_improving": int(np.sum(candidate < reference)), "years_available": int(np.sum(np.isfinite(candidate+reference)))}


def bootstrap_comparison(candidate_draws, reference_draws, minimum_valid=1900):
    c, r = np.asarray(candidate_draws), np.asarray(reference_draws)
    delta = c-r
    skill = np.divide(c, r, out=np.full(c.shape, np.nan), where=(r > 0) & np.isfinite(r))
    skill = 1-skill
    result = {}
    for name, values in [("candidate_brier", c), ("reference_brier", r), ("delta_brier", delta), ("brier_skill", skill)]:
        valid = values[np.isfinite(values)]
        lo, hi = np.percentile(valid, [2.5, 97.5], method="linear") if len(valid) >= minimum_valid else (None, None)
        result.update({name + "_lower": float(lo) if lo is not None else None,
                       name + "_upper": float(hi) if hi is not None else None,
                       name + "_valid_replicates": len(valid), name + "_invalid_replicates": len(values)-len(valid),
                       name + "_interval_status": "insufficient_resampling_support" if len(valid)<minimum_valid else
                           "conditional_valid_replicates" if len(valid)<len(values) else "complete"})
    result["empty_risk_slot_replicates"] = int((~np.isfinite(c) | ~np.isfinite(r)).sum())
    result["zero_reference_brier_replicates"] = int(((r == 0) & np.isfinite(c)).sum())
    return result


def fixed_contrasts(arms):
    """Prespecified increments plus all fixed simple/linear references."""
    pairs = {}
    increments = [("F1", "F0"), ("F1P", "F0P"), ("F2", "F1"), ("F3", "F2"), ("F4", "F3"),
                  ("F2", "F2_minus_NT"), ("F2", "F2_minus_NM"), ("F3", "F3_minus_RT"), ("F3", "F3_minus_RM")]
    def add(candidate, reference, role):
        if candidate != reference:
            pairs.setdefault((candidate, reference), []).append(role)
    for learner in ("logistic", "lightgbm"):
        for candidate, reference in increments:
            add(learner + "_" + candidate, learner + "_" + reference,
                "primary_moisture" if (candidate, reference) == ("F1", "F0") else "prespecified_increment")
        for arm in arms:
            candidate = learner + "_" + arm
            for reference in ("persistence", "seasonal", "seasonal_pooled", "transition", "logistic_F4"):
                add(candidate, reference, "fixed_baseline")
            if learner == "lightgbm":
                add(candidate, "logistic_" + arm, "same_arm_linear")
    return [{"candidate": c, "reference": r, "roles": ";".join(roles)} for (c, r), roles in pairs.items()]


def summarize_annual(rows, metrics=("brier", "brier_coslat", "ap", "pr_auc_trapezoid", "log_loss_clip_1e_6", "prevalence")):
    """Equal-year means; missing annual metrics remain NA with availability."""
    result = {"years": len(rows), "n": sum(r["n"] for r in rows), "events": sum(r["events"] for r in rows)}
    for metric in metrics:
        values = [r.get(metric) for r in rows]
        result[metric + "_years_available"] = sum(v is not None for v in values)
        result[metric] = float(np.mean(values)) if all(v is not None for v in values) else None
    return result

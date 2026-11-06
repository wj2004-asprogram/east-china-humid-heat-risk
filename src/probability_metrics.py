"""Raw probability metrics with explicit risk sets and tied-score PR curves."""

import numpy as np


def probability_scores(y, p, latitude=None):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if y.shape != p.shape or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Require aligned finite binary outcomes and probabilities in [0,1]")
    n = len(y)
    if not n:
        return {"n": 0, "events": 0, "prevalence": None, "brier": None, "ap": None,
                "pr_auc_trapezoid": None, "tp": 0, "fp": 0, "fn": 0, "tn": 0}
    positive = int(y.sum())
    alert = p >= .5
    tp, fp = int((alert & (y == 1)).sum()), int((alert & (y == 0)).sum())
    fn, tn = positive-tp, n-positive-fp
    result = {"n": n, "events": positive, "prevalence": positive/n,
              "brier": float(np.mean((p-y)**2)), "ap": None, "pr_auc_trapezoid": None,
              "tp": tp, "fp": fp, "fn": fn, "tn": tn,
              "recall": tp/(tp+fn) if tp+fn else None,
              "precision": tp/(tp+fp) if tp+fp else None,
              "far": fp/(tp+fp) if tp+fp else None,
              "fpr": fp/(fp+tn) if fp+tn else None,
              "miss_rate": fn/(tp+fn) if tp+fn else None,
              "alert_fraction": (tp+fp)/n}
    if latitude is not None:
        result["brier_coslat"] = float(np.average((p-y)**2, weights=np.cos(np.deg2rad(latitude))))
    if 0 < positive < n:
        order = np.argsort(-p, kind="stable")
        sorted_p, sorted_y = p[order], y[order]
        ends = np.r_[np.where(sorted_p[:-1] != sorted_p[1:])[0], n-1]
        cumulative = np.cumsum(sorted_y)[ends]
        recall = np.r_[0., cumulative/positive]
        precision = np.r_[1., cumulative/(ends+1)]
        result["ap"] = float(np.sum(np.diff(recall)*precision[1:]))
        result["pr_auc_trapezoid"] = float(np.sum(np.diff(recall)*(precision[:-1]+precision[1:])/2))
    return result


def evaluate_year(y, previous_y, p, latitude):
    """Onset includes both 00 and 01; previous outcome never routes prediction."""
    y, previous_y, p, latitude = map(np.asarray, (y, previous_y, p, latitude))
    return {name: probability_scores(y[mask], p[mask], latitude[mask]) for name, mask in (
        ("all", np.ones(len(y), dtype=bool)), ("onset", previous_y == 0),
        ("persistence_or_clearance", previous_y == 1))}


def reliability_counts(y, p):
    y, p = np.asarray(y), np.asarray(p)
    bins = np.minimum((p*10).astype(int), 9)
    rows = []
    for index in range(10):
        selected = bins == index
        n = int(selected.sum())
        rows.append({"bin": index, "left": index/10, "right": (index+1)/10, "n": n,
                     "events": int(y[selected].sum()),
                     "mean_probability": float(p[selected].mean()) if n else None,
                     "event_fraction": float(y[selected].mean()) if n else None})
    return rows

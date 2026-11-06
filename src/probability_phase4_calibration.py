"""Frozen, year-balanced probability calibration; no model selection."""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


@dataclass(frozen=True)
class PlattMapping:
    a: float = 1.0
    b: float = 0.0
    status: str = "identity"
    reason: str = ""
    objective: float | None = None
    iterations: int = 0

    def apply(self, p, clip=1e-6):
        p = validate_probabilities(p)
        if self.status != "fitted":
            return p.copy()
        q = np.clip(p, clip, 1 - clip)
        return expit(self.a * (np.log(q) - np.log1p(-q)) + self.b)

    def to_dict(self):
        return asdict(self)


def validate_probabilities(p):
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Require one-dimensional finite probabilities in [0, 1]")
    return p


def year_balanced_weights(years):
    years = np.asarray(years)
    if years.ndim != 1 or not len(years):
        raise ValueError("Require nonempty one-dimensional years")
    unique, inverse, counts = np.unique(years, return_inverse=True, return_counts=True)
    return 1.0 / (len(unique) * counts[inverse])


def _inputs(y, p, years, clip):
    p = validate_probabilities(p)
    y = np.asarray(y, dtype=np.float64)
    years = np.asarray(years)
    if y.shape != p.shape or years.shape != p.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("Outcomes and years must align with probabilities; outcomes are binary")
    weights = year_balanced_weights(years)
    q = np.clip(p, clip, 1 - clip)
    return y, np.log(q) - np.log1p(-q), weights


def fit_platt(y, p, years, *, clip=1e-6, penalty=1e-6, persistence=False):
    """Fit a>=0, b free with equal total weight for each calibration year.

    Structural errors raise. Numerical/optimizer/single-class failure returns the
    original raw probability exactly, including raw endpoints zero and one.
    """
    y, x, weights = _inputs(y, p, years, clip)
    if persistence:
        return PlattMapping(status="persistence_raw", reason="protocol_no_calibration")
    if len(np.unique(y)) < 2:
        return PlattMapping(reason="single_class_calibration")

    def objective(theta):
        a, b = theta
        z = a * x + b
        residual = weights * (expit(z) - y)
        loss = np.dot(weights, np.logaddexp(0, z) - y * z) + penalty * (a - 1) ** 2
        gradient = np.array([np.dot(residual, x) + 2 * penalty * (a - 1), residual.sum()])
        return float(loss), gradient

    try:
        result = minimize(objective, np.array([1., 0.]), jac=True, method="L-BFGS-B",
                          bounds=[(0., None), (None, None)],
                          options={"maxiter": 2000, "maxfun": 20000, "maxls": 20,
                                   "ftol": 1e-12, "gtol": 1e-8})
        if not result.success or not np.isfinite(result.x).all() or not np.isfinite(result.fun):
            return PlattMapping(reason="optimizer_failure: " + str(result.message))
        return PlattMapping(float(result.x[0]), float(result.x[1]), "fitted", "",
                            float(result.fun), int(result.nit))
    except (FloatingPointError, ValueError, RuntimeError) as exc:
        return PlattMapping(reason="optimizer_exception: " + str(exc))


def budget_threshold(probabilities_by_year, budget=.1):
    """Smallest fixed-grid threshold meeting the equally weighted annual budget.

    This function intentionally accepts no labels. Ties stay together.
    """
    annual = [validate_probabilities(p) for p in probabilities_by_year]
    if not annual or any(not len(p) for p in annual):
        raise ValueError("Budget threshold requires nonempty probabilities for every year")
    thresholds = np.r_[np.arange(101, dtype=float) / 100., 1.000001]
    for threshold in thresholds:
        annual_fractions = [float(np.mean(p >= threshold)) for p in annual]
        fraction = float(np.mean(annual_fractions))
        if fraction <= budget:
            return {"threshold": float(threshold), "calibration_alert_fraction": fraction,
                    "annual_alert_fractions": annual_fractions, "budget": budget}
    raise AssertionError("No-alert sentinel must meet nonnegative budget")


def calibration_diagnostic(y, p, years, clip=1e-6):
    """Unpenalized logistic intercept/slope diagnosis; never changes predictions.

    Slope is free here, unlike the production calibration map. Single-class,
    constant-score, separated or failed fits are reported as unidentified/NA.
    """
    y, x, weights = _inputs(y, p, years, clip)
    if len(np.unique(y)) < 2:
        return {"intercept": None, "slope": None, "status": "single_class"}
    if np.ptp(x) <= np.finfo(float).eps:
        return {"intercept": None, "slope": None, "status": "constant_score_unidentified"}
    # Complete/quasi separation admits no finite joint unpenalized MLE.
    if x[y == 0].max() <= x[y == 1].min() or x[y == 1].max() <= x[y == 0].min():
        return {"intercept": None, "slope": None, "status": "separation_no_finite_mle"}

    def objective(theta):
        intercept, slope = theta
        z = intercept + slope * x
        residual = weights * (expit(z) - y)
        return float(np.dot(weights, np.logaddexp(0, z) - y * z)), np.array([residual.sum(), np.dot(residual, x)])

    try:
        result = minimize(objective, np.array([0., 1.]), jac=True, method="L-BFGS-B",
                          options={"maxiter": 2000, "maxfun": 20000, "maxls": 20,
                                   "ftol": 1e-12, "gtol": 1e-8})
        if result.success and np.isfinite(result.x).all() and np.isfinite(result.fun):
            return {"intercept": float(result.x[0]), "slope": float(result.x[1]), "status": "fitted"}
        return {"intercept": None, "slope": None, "status": "optimizer_failure"}
    except (FloatingPointError, ValueError, RuntimeError):
        return {"intercept": None, "slope": None, "status": "optimizer_exception"}

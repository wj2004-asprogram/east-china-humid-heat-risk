import numpy as np
import pytest

from src.probability_phase4_calibration import (PlattMapping, fit_platt, budget_threshold,
    calibration_diagnostic, year_balanced_weights)


def test_year_balance_and_replication_invariance():
    y = np.array([0, 1, 0, 0, 1, 1])
    p = np.array([.1, .4, .3, .6, .7, .9])
    year = np.array([2000, 2000, 2001, 2001, 2001, 2001])
    weights = year_balanced_weights(year)
    assert weights[year == 2000].sum() == pytest.approx(.5)
    first = fit_platt(y, p, year)
    repeat = np.r_[np.repeat(np.arange(2), 7), np.arange(2, 6)]
    second = fit_platt(y[repeat], p[repeat], year[repeat])
    assert first.status == second.status == "fitted"
    assert first.a == pytest.approx(second.a, abs=1e-7)
    assert first.b == pytest.approx(second.b, abs=1e-7)


def test_nonnegative_constraint_and_persistence_exact_endpoints():
    mapping = fit_platt([1, 1, 0, 0], [.1, .2, .8, .9], [2000]*4)
    assert mapping.status == "fitted"
    assert mapping.a == pytest.approx(0, abs=1e-8)
    assert np.allclose(mapping.apply([0, 1]), [.5, .5])
    persistence = fit_platt([0, 1], [0, 1], [2000]*2, persistence=True)
    assert np.array_equal(persistence.apply([0, 1]), [0, 1])
    assert persistence.status == "persistence_raw"


def test_identity_on_single_class_and_optimizer_failure(monkeypatch):
    mapping = fit_platt([0, 0], [0, 1], [2000]*2)
    assert mapping.reason == "single_class_calibration"
    assert np.array_equal(mapping.apply([0, 1]), [0, 1])
    import src.probability_phase4_calibration as module
    def failed(*args, **kwargs):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(module, "minimize", failed)
    mapping = fit_platt([0, 1], [0, 1], [2000]*2)
    assert mapping.status == "identity"
    assert "synthetic failure" in mapping.reason
    assert np.array_equal(mapping.apply([0, 1]), [0, 1])


def test_budget_uses_equal_years_ties_and_no_alert_sentinel():
    # Pooled rate is 1/11, but equal-year rate at .5 is .5: must exceed .8.
    result = budget_threshold([np.array([.8]), np.repeat(.1, 10)])
    assert result["threshold"] == .81
    result = budget_threshold([np.array([0., 1.]), np.array([0., 1.])])
    assert result["threshold"] == 1.000001
    assert result["calibration_alert_fraction"] == 0
    assert budget_threshold([np.r_[np.repeat(.2, 9), .9]])["threshold"] == .21


def test_diagnostic_unidentified_and_negative_slope_allowed():
    assert calibration_diagnostic([0, 1], [.4, .4], [2000]*2)["slope"] is None
    assert calibration_diagnostic([0, 1], [.2, .8], [2000]*2)["status"] == "separation_no_finite_mle"
    result = calibration_diagnostic([1, 0, 1, 0, 0, 1, 0, 0], [.1,.2,.3,.4,.6,.7,.8,.9], [2000]*8)
    assert result["status"] == "fitted"
    assert result["slope"] < 0


def test_invalid_inputs_raise_instead_of_identity():
    with pytest.raises(ValueError):
        fit_platt([0, 1], [np.nan, .5], [2000]*2)
    with pytest.raises(ValueError):
        budget_threshold([np.array([])])

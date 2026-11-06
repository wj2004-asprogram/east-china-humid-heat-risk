"""Synthetic checks for the prespecified LR ablations and frozen scaler."""
import numpy as np
import pytest
from pathlib import Path

from tools.probability_phase3_lr_ablations import arm_columns, restore_scaler, fit_candidate
from tools.probability_phase2_baselines import year_data


def test_p_is_single_anchor_and_columns_keep_original_order():
    names = ["temperature", "hi", "y_anchor", "humidity", "duration", "night"]
    groups = ["T", "H", "H", "M", "H", "NT"]
    cols = arm_columns(names, groups, {"F0P": ["T", "P"], "F1P": ["T", "P", "M"]})
    assert cols["F0P"].tolist() == [0, 2]
    assert cols["F1P"].tolist() == [0, 2, 3]
    with pytest.raises(ValueError, match="y_anchor"):
        arm_columns(["temperature"], ["T"], {"F0P": ["T", "P"]})


def test_scaler_restoration_uses_frozen_training_statistics():
    state = {"features": ["a", "b"], "scope": "strict", "lag": 2,
             "train_years": list(range(2000, 2015)), "training_rows": 30, "C": .1,
             "scaler_mean": [10., 20.], "scaler_scale": [2., 4.]}
    scaler = restore_scaler(state, ["a", "b"], "strict", 2, 30)
    np.testing.assert_allclose(scaler.transform(np.array([[14., 8.]])), [[2., -3.]])
    with pytest.raises(ValueError, match="provenance"):
        restore_scaler(state, ["a", "b"], "strict", 1, 30)


@pytest.mark.parametrize("year", [1999, 2018, 2019, 2020, 2024])
def test_closed_years_rejected_before_io(year):
    with pytest.raises(ValueError, match="only training/validation"):
        year_data(Path("must_not_open"), "strict", year, None)


def test_small_fit_and_convergence_failure_returns_no_model(monkeypatch):
    import warnings
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning
    x = np.array([[-2.], [-1.], [1.], [2.]])
    model, record = fit_candidate(x, np.array([0, 0, 1, 1]))
    assert record["converged"] and model.predict_proba(x)[0, 1] < .5
    def fail(self, x, y):
        self.n_iter_ = np.array([2000])
        warnings.warn("synthetic failure", ConvergenceWarning)
    monkeypatch.setattr(LogisticRegression, "fit", fail)
    model, record = fit_candidate(x, np.array([0, 0, 1, 1]))
    assert model is None and not record["converged"] and record["iterations"] == 2000

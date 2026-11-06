import numpy as np
import pytest

from src.probability_metrics import probability_scores, evaluate_year, reliability_counts


def test_tied_constant_probability_ap_and_trapezoid():
    result = probability_scores([0, 0, 1, 1], [.5]*4)
    assert result["brier"] == .25
    assert result["ap"] == .5
    assert result["pr_auc_trapezoid"] == .75
    assert result["tp"] == result["fp"] == 2


def test_onset_risk_set_retains_negatives_and_one_class_is_na():
    result = evaluate_year([0, 1, 1, 0], [0, 0, 1, 1], [.2, .8, .8, .2], [30]*4)
    assert result["onset"]["n"] == 2
    assert result["onset"]["events"] == 1
    assert result["onset"]["brier"] == pytest.approx(.04)
    assert probability_scores([0, 0], [.1, .2])["ap"] is None
    assert reliability_counts([0, 1], [0., 1.])[9]["n"] == 1


def test_invalid_predictions_rejected():
    with pytest.raises(ValueError):
        probability_scores([1], [np.nan])

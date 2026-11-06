"""Synthetic, hand-calculated probabilities and frozen fitting boundaries."""

import json

import numpy as np
import pandas as pd
import pytest

from src.probability_baselines import ClimatologyBaseline, TransitionBaseline, persistence_probabilities


def training():
    return pd.DataFrame({"date": pd.to_datetime(["2000-06-09", "2000-06-10", "2000-06-09", "2000-06-10"]),
                         "grid_id": ["a", "a", "b", "b"], "spatial_role": ["train"]*4,
                         "label": [1, 1, 0, 0], "state_lag1": [1, 1, 0, 0], "state_lag2": [0, 0, 1, 1],
                         "state_date_lag1": pd.to_datetime(["2000-06-08", "2000-06-09", "2000-06-08", "2000-06-09"]),
                         "state_date_lag2": pd.to_datetime(["2000-06-07", "2000-06-08", "2000-06-07", "2000-06-08"])})


def test_climatology_smoothing_unseen_grid_and_serializable_provenance():
    model = ClimatologyBaseline(cutoff_lag_days=1).fit(training())
    query = pd.DataFrame({"date": ["2015-06-12"]*3, "grid_id": ["a", "b", "unseen"],
                          "label": [1, 1, 1]})  # Prediction never consumes these labels.
    # Prior=(2+1)/(4+2)=1/2; a=(2+20/2)/22; b=(0+20/2)/22.
    np.testing.assert_allclose(model.predict(query, cutoff_lag_days=1), [12/22, 10/22, 1/2])
    state = json.loads(json.dumps(model.to_state()))
    assert state["train_years"] == [2000]
    assert state["train_grid_ids"] == ["a", "b"]
    assert state["shrinkage_strength"] == 20


def test_strict_spatial_uses_only_regional_prior_and_rejects_excluded_rows():
    model = ClimatologyBaseline(cutoff_lag_days=2, strict_spatial=True).fit(training())
    query = pd.DataFrame({"date": ["2018-06-09"]*2, "grid_id": ["a", "holdout"],
                          "label": [1, 0]})
    np.testing.assert_allclose(model.predict(query, cutoff_lag_days=2), [.5, .5])
    assert model.to_state()["grid_counts"] == []
    for excluded in ("buffer", "holdout"):
        data = training()
        data.loc[0, "spatial_role"] = excluded
        with pytest.raises(ValueError, match="forbids"):
            TransitionBaseline(cutoff_lag_days=2, strict_spatial=True).fit(data)


@pytest.mark.parametrize("bad_year", [2015, 2018, 2020])
def test_nontraining_years_never_enter_fit(bad_year):
    data = training()
    data.loc[0, "date"] = pd.Timestamp(bad_year, 6, 9)
    with pytest.raises(ValueError, match="training years"):
        ClimatologyBaseline(cutoff_lag_days=1).fit(data)


def test_d2_transition_is_direct_pairing_and_persistence_uses_d2():
    data = training()
    one = TransitionBaseline(cutoff_lag_days=1).fit(data)
    two = TransitionBaseline(cutoff_lag_days=2).fit(data)
    # Identical queried state=1 but D-1 pairs have 2 events, D-2 pairs have 0.
    query = data.iloc[[0]].copy()
    query["state_lag2"] = 1
    assert one.predict(query, cutoff_lag_days=1)[0] == pytest.approx(12/22)
    assert two.predict(query, cutoff_lag_days=2)[0] == pytest.approx(10/22)
    np.testing.assert_array_equal(persistence_probabilities(data, cutoff_lag_days=2), [0, 0, 1, 1])
    with pytest.raises(ValueError, match="differs"):
        two.predict(query, cutoff_lag_days=1)
    query["state_date_lag2"] = query["state_date_lag1"]
    with pytest.raises(ValueError, match="minus 2"):
        two.predict(query, cutoff_lag_days=2)


def test_missing_state_support_shrinks_to_prior_but_unknown_state_rejected():
    data = training()
    data["state_lag2"] = 0
    model = TransitionBaseline(cutoff_lag_days=2).fit(data)
    query = data.iloc[[0]].copy()
    query["state_lag2"] = 1
    assert model.predict(query, cutoff_lag_days=2)[0] == .5
    query["state_lag2"] = np.nan
    with pytest.raises(ValueError, match="unknown"):
        model.predict(query, cutoff_lag_days=2)


def test_season_bin_boundaries_and_no_unsupported_bin_fallback():
    data = training().iloc[[0, 1]].copy()
    data["date"] = pd.to_datetime(["2000-08-29", "2000-08-30"])
    data["label"] = [0, 1]
    model = ClimatologyBaseline(cutoff_lag_days=1, strict_spatial=True).fit(data)
    query = pd.DataFrame({"date": ["2017-08-29", "2017-08-30", "2017-08-31"], "grid_id": ["a"]*3})
    np.testing.assert_allclose(model.predict(query, cutoff_lag_days=1), [1/3, 2/3, 2/3])
    assert [item["season_bin"] for item in model.to_state()["regional"]] == [5, 6]
    query.loc[0, "date"] = "2017-06-09"
    with pytest.raises(ValueError, match="No training support"):
        model.predict(query, cutoff_lag_days=1)


def test_duplicate_missing_label_and_d1_only_input_rejected():
    data = training()
    with pytest.raises(ValueError, match="Duplicate"):
        ClimatologyBaseline(cutoff_lag_days=1).fit(pd.concat([data, data.iloc[[0]]]))
    data.loc[0, "label"] = np.nan
    with pytest.raises(ValueError, match="label must"):
        ClimatologyBaseline(cutoff_lag_days=1).fit(data)
    with pytest.raises(ValueError, match="state_lag2"):
        persistence_probabilities(training().drop(columns=["state_lag2"]), cutoff_lag_days=2)

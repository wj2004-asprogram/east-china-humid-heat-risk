import numpy as np
import pytest


def test_report_requires_both_lags_and_matching_risk_counts():
    import pandas as pd
    from tools.probability_phase3_lightgbm_report import comparisons
    row = dict(scope="temporal", lag=1, year=2015, risk_set="onset", n=100, events=10)
    tree = pd.DataFrame([dict(row, model="lightgbm_F1", brier=.08)])
    lr = pd.DataFrame([dict(row, model="logistic_F1", brier=.10)])
    result = comparisons(tree, lr)
    assert result.delta_brier_tree_minus_lr.iloc[0] == pytest.approx(-.02)
    assert result.bss_tree_vs_same_arm_lr.iloc[0] == pytest.approx(.2)
    assert not result.both_lags_available.iloc[0]
    lr.loc[0, "brier"] = 0
    assert pd.isna(comparisons(tree, lr).bss_tree_vs_same_arm_lr.iloc[0])
    lr.loc[0, "events"] = 9
    with pytest.raises(ValueError, match="sample/outcome"):
        comparisons(tree, lr)

from src.probability_phase3_lightgbm import arm_columns, select_candidate, annual_brier, load_xy


def test_p_adds_only_the_anchor_state_not_other_hi_history():
    names = ["latitude", "tmean", "tdmean", "himax", "y_anchor", "event_duration"]
    groups = ["C", "T", "M", "H", "H", "H"]
    columns = arm_columns(names, groups, {"F0": ["C", "T"], "F0P": ["C", "T", "P"],
                                        "F1P": ["C", "T", "P", "M"]})
    assert columns["F0"].tolist() == [0, 1]
    assert columns["F0P"].tolist() == [0, 1, 4]
    assert columns["F1P"].tolist() == [0, 1, 2, 4]


def test_candidate_ties_and_year_weighting_are_frozen():
    records = [dict(status="complete", mean_annual_validation_brier=.1, num_leaves=leaf,
                    min_child_samples=child, reg_lambda=lam)
               for leaf, child, lam in [(15, 500, 10), (7, 100, 10), (7, 500, 1), (7, 500, 10)]]
    assert select_candidate(records) == records[-1]
    scores = annual_brier(np.array([0, 1, 1]), np.array([.5, .5, 1.]), {2015: 1, 2016: 2})
    assert scores == {2015: .25, 2016: .125}
    assert np.mean(list(scores.values())) == .1875


def test_loader_rejects_calibration_and_final_years_before_file_access(tmp_path):
    for year in (2018, 2019, 2020, 2024):
        with pytest.raises(ValueError, match="2000-2017"):
            load_xy(tmp_path, "temporal", 1, [year], [0])


def test_lightgbm_synthetic_model_survives_text_roundtrip():
    lgb = pytest.importorskip("lightgbm")
    from tools.probability_phase3_lightgbm import fixed_params
    x = np.arange(200, dtype=np.float32).reshape(100, 2)
    y = (x[:, 0] >= 100).astype(int)
    params = dict(fixed_params(), num_leaves=7, min_data_in_leaf=3)
    data = lgb.Dataset(x, label=y, params=params, categorical_feature=[])
    model = lgb.train(params, data, num_boost_round=10)
    restored = lgb.Booster(model_str=model.model_to_string())
    np.testing.assert_allclose(model.predict(x), restored.predict(x), rtol=0, atol=0)
    assert model.predict(x[:1])[0] < model.predict(x[-1:])[0]


def test_fitted_checkpoint_resumes_without_training_and_rejects_changed_binding(tmp_path, monkeypatch):
    lgb = pytest.importorskip("lightgbm")
    from tools.probability_phase3_lightgbm import fixed_params, fit_or_restore
    x = np.arange(200, dtype=np.float32).reshape(100, 2)
    y = (x[:, 0] >= 100).astype(int)
    params = dict(fixed_params(), min_data_in_leaf=3)
    train = lgb.Dataset(x, label=y, params=params, categorical_feature=[])
    valid = lgb.Dataset(x[::2], label=y[::2], reference=train, params=params, categorical_feature=[])
    binding = {"scope": "synthetic", "lag": 2, "feature_names": ["temperature", "anchor"]}
    stop = {"stopping_rounds": 3, "first_metric_only": True, "min_delta": 0}
    first, state = fit_or_restore(tmp_path/"fit", binding, params, train, rounds=12, validation=valid, stop=stop)
    assert state["best_iteration"] > 0
    monkeypatch.setattr(lgb, "train", lambda *a, **k: pytest.fail("Completed checkpoint must not refit"))
    restored, _ = fit_or_restore(tmp_path/"fit", binding, params, None, rounds=12)
    np.testing.assert_array_equal(first.predict(x, num_iteration=state["prediction_num_iteration"]), restored.predict(x))
    with pytest.raises(ValueError, match="configuration differs"):
        fit_or_restore(tmp_path/"fit", dict(binding, lag=1), params, None, rounds=12)


def test_resource_deadline_marks_pending_without_altering_scientific_grid(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from tools.probability_phase3_lightgbm import may_start_fit
    plan = tmp_path/"batch.json"
    plan.write_text(json.dumps({"lightgbm": {"deadline_unix": 200}}))
    args = SimpleNamespace(batch_plan=plan, output_dir=tmp_path)
    monkeypatch.setattr("tools.probability_phase3_lightgbm.time.time", lambda: 199)
    assert may_start_fit(args)
    monkeypatch.setattr("tools.probability_phase3_lightgbm.time.time", lambda: 200)
    assert not may_start_fit(args)
    state = json.loads((tmp_path/"resource_stop.json").read_text())
    assert state["pending_not_failed"] and state["scientific_grid_unchanged"]


def test_interruption_is_pending_not_failure_and_completed_result_supersedes_it(tmp_path):
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from tools.probability_phase3_lightgbm import update_registry, candidate_grid, candidate_id
    config = json.loads(Path("configs/probability_phase3_run.json").read_text())
    candidate = candidate_grid(config)[0]
    directory = tmp_path/"search"/"strict"/candidate_id(candidate)
    receipts = tmp_path/"interrupted_invocations"
    receipts.mkdir()
    (receipts/"receipt.json").write_text(json.dumps({"interrupted_fit_directories": [str(directory)]}))
    args = SimpleNamespace(output_dir=tmp_path)
    update_registry(args, config)
    state = json.loads((tmp_path/"execution_registry.json").read_text())
    assert (state["interrupted"], state["failed"], state["not_executed"], state["pending_total"]) == (1, 0, 57, 58)
    directory.mkdir(parents=True)
    (directory/"result.json").write_text(json.dumps({"status": "complete"}))
    update_registry(args, config)
    state = json.loads((tmp_path/"execution_registry.json").read_text())
    assert (state["completed_or_fitted"], state["interrupted"], state["pending_total"]) == (1, 0, 57)

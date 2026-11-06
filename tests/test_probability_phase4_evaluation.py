import numpy as np
import pytest

from src.probability_phase4_evaluation import (score_probability, risk_masks, alert_metrics,
    grid_alert_days, daily_brier_sufficient, bootstrap_draws, resampled_brier,
    annual_brier, comparison_point, bootstrap_comparison, fixed_contrasts, summarize_annual)


def test_score_ties_clipped_loss_single_class_and_boundaries():
    score = score_probability([0, 1], [.5, .5], [30, 30])
    assert score["brier"] == .25
    assert score["ap"] == .5
    assert score["pr_auc_trapezoid"] == .75
    assert score["log_loss_clip_1e_6"] == pytest.approx(np.log(2))
    assert np.isfinite(score_probability([0, 1], [1, 0], [30, 30])["log_loss_clip_1e_6"])
    assert score_probability([0], [.2], [30])["ap"] is None
    masks = risk_masks([0, 1, 0, 1], [0, 0, 1, 1], [117,117.5,117,117.5], [30,30,30.5,30.5])
    assert masks["onset"].sum() == 2
    assert masks["northeast"].tolist() == [False,False,False,True]
    assert all(masks[k].sum() == 1 for k in ["0_to_0","0_to_1_onset","1_to_0_clearance","1_to_1_continuation"])


def test_alert_denominators_and_grid_missing_days():
    score = alert_metrics([0, 0], [.1, .7], 1.000001)
    assert score["precision"] is None and score["recall"] is None
    assert score["fpr"] == 0
    rows = grid_alert_days([0,0,1], [.2,.8,.9], .5, [0,1,2])
    assert rows == [{"grid_index":0,"valid_days":2,"alert_days":1},
                    {"grid_index":1,"valid_days":1,"alert_days":1},
                    {"grid_index":2,"valid_days":0,"alert_days":0}]


def test_daily_sufficient_bootstrap_matches_literal_grid_date_resampling():
    # Unequal grid coverage and a day with no onset rows are retained.
    y = np.array([0,1,0,1,1,0,0])
    p = np.array([.1,.5,.2,.8,.6,.3,.4])
    day = np.array([0,0,1,2,2,3,4])
    mask = np.array([1,1,0,1,1,1,1], dtype=bool)
    annual = [daily_brier_sufficient(y,p,day,mask,5), daily_brier_sufficient(y,1-p,day,mask,5)]
    draws = bootstrap_draws(2,5,2,31,42)
    result = resampled_brier(annual, draws)
    expected = []
    for sampled_years, sampled_dates in zip(draws["years"],draws["dates"]):
        by_year = []
        for yr, dates in zip(sampled_years,sampled_dates):
            pp = p if yr == 0 else 1-p
            errors = np.concatenate([(pp[(day==d)&mask]-y[(day==d)&mask])**2 for d in dates])
            by_year.append(errors.mean() if len(errors) else np.nan)
        expected.append(np.mean(by_year))
    np.testing.assert_allclose(result,expected,equal_nan=True)
    again = bootstrap_draws(2,5,2,31,42)
    assert draws["sha256"] == again["sha256"]
    for row in draws["dates"].reshape(-1,5):
        assert row[1]-row[0] == row[3]-row[2] == 1
        assert np.all((row >= 0)&(row < 5))


def test_zero_risk_year_invalidates_not_drops_and_bss_recomputes_denominator():
    stats = np.array([[[1,2],[0,0]],[[0,0],[0,0]]],dtype=float)
    draws = {"years":np.array([[0,1],[0,0]]),"dates":np.array([[[0,1],[0,1]],[[0,1],[0,1]]])}
    result = resampled_brier(stats,draws)
    assert np.isnan(result[0]) and result[1] == .5
    ci = bootstrap_comparison(np.array([.1,.3]),np.array([.2,.4]), minimum_valid=1)
    assert ci["brier_skill_lower"] == pytest.approx(np.percentile([.5,.25],2.5))
    zero = bootstrap_comparison([0,.1],[0,.2])
    assert zero["brier_skill_valid_replicates"] == 1
    assert comparison_point([0],[0])["brier_skill"] is None
    assert comparison_point([.2,.4],[.4,.4])["brier_skill"] == pytest.approx(.25)


def test_fixed_pairs_and_equal_year_reporting():
    arms = ["F0","F1","F2","F3","F4","F0P","F1P","F2_minus_NT","F2_minus_NM","F3_minus_RT","F3_minus_RM"]
    pairs = fixed_contrasts(arms)
    assert len({(r["candidate"],r["reference"]) for r in pairs}) == len(pairs)
    assert any(r["candidate"] == "lightgbm_F4" and r["reference"] == "logistic_F4" for r in pairs)
    result = summarize_annual([{"n":1,"events":1,"brier":.1,"ap":None}, {"n":99,"events":0,"brier":.3,"ap":.2}])
    assert result["brier"] == pytest.approx(.2)
    assert result["ap"] is None and result["ap_years_available"] == 1

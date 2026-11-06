"""Scientific boundary cases for frozen five-site temporal case analysis."""

import numpy as np
import pandas as pd
import pytest
import json
from pathlib import Path

from tools.esg_build_research_data import grid_metrics
from tools.esg_case_temporal import analyze_grid, choose_years, describe, extract_episodes, validate_protocol


def test_equal_threshold_single_day_and_peak_tie_earliest():
    rows = extract_episodes(pd.date_range('2020-06-01', periods=6), [40.6, 40, 41, 42, 42, 39])
    assert [r['duration_days'] for r in rows] == [1, 3]
    assert rows[0]['left_season_censored'] is True
    assert rows[1]['peak_date'] == '2020-06-04'
    assert rows[1]['peak_hi_c'] == 42
    assert rows[1]['mean_hi_c'] == pytest.approx(125/3)


def test_runs_stop_at_missing_dates_nonfinite_values_and_seasons():
    rows = extract_episodes(['2020-06-01', '2020-06-03', '2020-06-04', '2020-06-05', '2020-08-31', '2021-06-01'], [41, 41, np.nan, 41, 41, 41])
    assert [r['duration_days'] for r in rows] == [1, 1, 1, 1, 1]
    assert rows[-2]['right_season_censored'] is True
    assert rows[-1]['left_season_censored'] is True


@pytest.mark.parametrize('missing_kind', ['date', 'value'])
def test_incomplete_summer_withholds_all_episodes(missing_kind):
    dates = pd.date_range('2020-06-01', '2020-08-31')
    values = np.full(92, 41.)
    if missing_kind == 'date':
        dates, values = dates.delete(10), np.delete(values, 10)
    else:
        values[10] = np.nan
    grid = dict(grid_id='30.2500,120.5000', latitude=30.25, longitude=120.5,
                **grid_metrics(dates, values, 1, 40.6))
    result = analyze_grid(grid)
    assert result['episodes'] == []
    assert result['annual'][0]['episode_count'] is None
    assert result['annual'][0]['days_ge_threshold'] is None
    assert result['selection'] is None


def test_full_summer_episode_is_censored_at_both_edges():
    dates = pd.date_range('2020-06-01', '2020-08-31')
    rows = extract_episodes(dates, np.full(92, 40.6))
    assert len(rows) == 1 and rows[0]['duration_days'] == 92
    assert rows[0]['left_season_censored'] and rows[0]['right_season_censored']
    assert rows[0]['peak_date'] == '2020-06-01'


def test_year_selection_preserves_all_tie_stages():
    annual = [{'year':year, 'complete_summer':True, 'days_ge_threshold':days, 'max_consecutive_days':run}
              for year,days,run in [(2000,10,4), (2001,20,4), (2002,30,6), (2003,30,7), (2004,30,7), (2005,20,5)]]
    result = choose_years(annual)
    assert result['typical']['median_days_ge_threshold'] == 25
    assert result['typical']['candidate_years'] == [2001, 2002, 2003, 2004, 2005]
    assert result['typical']['year'] == 2001
    assert result['extreme']['candidate_years_max_days'] == [2002, 2003, 2004]
    assert result['extreme']['candidate_years_max_days_and_run'] == [2003, 2004]
    assert result['extreme']['year'] == 2003


def test_distribution_uses_linear_quartiles():
    assert describe([0, 10]) == {'min':0., 'p25':2.5, 'median':5., 'p75':7.5, 'max':10.}


def test_masked_primary_withholds_results_and_rejects_injected_daily_values():
    grid = dict(grid_id='30.7500,121.7500', latitude=30.75, longitude=121.75,
                **grid_metrics(None, None, 0, 40.6))
    assert analyze_grid(grid)['selection'] is None
    grid['daily'] = [{'date':'2020-06-01', 'daily_max_heat_index_c':41}]
    with pytest.raises(ValueError, match='Masked primary'):
        analyze_grid(grid)


def test_duplicate_or_unsorted_dates_are_rejected():
    with pytest.raises(ValueError, match='unique and increasing'):
        extract_episodes(['2020-06-01', '2020-06-01'], [41, 41])


def test_changed_protocol_cannot_silently_run_existing_selection_rules():
    protocol = json.loads((Path(__file__).parents[1]/'data/research_phase1/protocol.json').read_text(encoding='utf-8'))
    validate_protocol(protocol)
    protocol['year_selection']['typical'][-1] = 'Latest year'
    with pytest.raises(ValueError, match='year-selection rule differs'):
        validate_protocol(protocol)

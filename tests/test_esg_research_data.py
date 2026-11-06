"""Boundary and data-withholding cases for the research grid supplement."""

import numpy as np
import pandas as pd
import pytest

from tools.esg_build_research_data import candidate_indices, grid_metrics


def test_bbox_candidates_are_order_invariant_and_include_boundary_contacts():
    axis = [31.0, 30.75, 30.5]
    assert [axis[i] for i in candidate_indices(axis, 30.86, 30.89)] == [30.75, 31.0]
    assert [axis[i] for i in candidate_indices(axis, 30.875, 30.875)] == [30.75, 31.0]
    with pytest.raises(ValueError, match='beyond'):
        candidate_indices(axis, 30.4, 30.6)


def test_non_land_has_no_daily_or_annual_climate_values():
    result = grid_metrics(None, None, 0, 40.6)
    assert result['screening_status'] == 'non_land_grid'
    assert result['daily'] == [] and result['annual'] == []
    assert result['mean_days_ge_threshold_per_complete_summer'] is None


def test_daily_missing_values_do_not_become_zero_or_complete_summers():
    dates = pd.date_range('2020-06-01', '2020-08-31')
    values = np.full(92, 40.6)
    values[10] = np.nan
    result = grid_metrics(dates, values, 1, 40.6)
    assert result['screening_status'] == 'incomplete_climate_data'
    assert result['daily'][10]['daily_max_heat_index_c'] is None
    assert result['annual'][0]['valid_days'] == 91
    assert result['annual'][0]['missing_date_days'] == 0
    assert result['annual'][0]['nonfinite_days'] == 1
    assert result['annual'][0]['days_ge_threshold'] is None
    assert result['complete_summers'] == 0
    assert result['mean_days_ge_threshold_per_complete_summer'] is None


def test_missing_date_is_distinct_from_present_nonfinite_value():
    dates = pd.date_range('2020-06-01', '2020-08-31').delete(10)
    result = grid_metrics(dates, np.full(91, 40.6), 1, 40.6)
    assert result['annual'][0]['missing_date_days'] == 1
    assert result['annual'][0]['nonfinite_days'] == 0
    assert result['annual'][0]['valid_days'] == 91
    assert result['annual'][0]['complete_summer'] is False

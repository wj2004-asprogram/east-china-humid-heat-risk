"""Partition invariance of the approved daily reducer, with literal references.

These do not execute a future Phase 2 downloader or choose an overlap policy.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.time_utils import local_daily


def hours(first, last, offset):
    times = pd.date_range(pd.Timestamp(first) - pd.Timedelta(hours=offset), f'{last} 23:00', freq='h')
    values = ((times.hour.to_numpy() + offset) % 24).astype(float)
    return xr.DataArray(values, dims='time', coords={'time': times}, attrs={'units': 'K'})


def reduce(data, first, last, offset, how):
    return local_daily(data, 'time', offset, how, date.fromisoformat(first), date.fromisoformat(last))


@pytest.mark.parametrize('offset', [0, 8])
@pytest.mark.parametrize('how,reference', [('max', 23.), ('mean', 11.5)])
@pytest.mark.parametrize('periods,total_days', [
    ([('2022-06-01', '2022-06-30'), ('2022-07-01', '2022-07-31'), ('2022-08-01', '2022-08-31')], 92),
    ([('2024-02-01', '2024-02-29'), ('2024-03-01', '2024-03-31')], 60),
    ([('2023-12-01', '2023-12-31'), ('2024-01-01', '2024-01-31')], 62),
])
def test_monthly_and_continuous_aggregation_match_independent_reference(offset, how, reference, periods, total_days):
    first, last = periods[0][0], periods[-1][1]
    source = hours(first, last, offset)
    continuous = reduce(source, first, last, offset, how)
    monthly = []
    for begin, end in periods:
        raw = source.sel(time=slice(pd.Timestamp(begin) - pd.Timedelta(hours=offset), f'{end} 23:00'))
        monthly.append(reduce(raw, begin, end, offset, how))
    joined = xr.concat(monthly, dim='time')
    xr.testing.assert_identical(joined, continuous)
    np.testing.assert_array_equal(joined.values, np.full(total_days, reference))
    assert joined.time.to_index().is_unique
    assert len(joined.time) == total_days


@pytest.mark.parametrize('how', ['max', 'mean'])
def test_bjt_month_boundary_pulses_have_literal_day_ownership(how):
    source = hours('2022-06-01', '2022-07-31', 8) * 0
    source.loc[{'time': '2022-06-30 15:00'}] = 48.
    source.loc[{'time': '2022-06-30 16:00'}] = 96.
    june = reduce(source.sel(time=slice(None, '2022-06-30 23:00')), '2022-06-01', '2022-06-30', 8, how)
    july = reduce(source.sel(time=slice('2022-06-30 16:00', None)), '2022-07-01', '2022-07-31', 8, how)
    assert june.sel(time='2022-06-30').item() == (48 if how == 'max' else 2)
    assert july.sel(time='2022-07-01').item() == (96 if how == 'max' else 4)
    assert june.sel(time='2022-06-29').item() == 0
    assert july.sel(time='2022-07-02').item() == 0


@pytest.mark.parametrize('how', ['max', 'mean'])
def test_missing_hour_value_at_month_join_propagates_only_to_its_day(how):
    source = hours('2022-06-01', '2022-07-31', 8)
    source.loc[{'time': '2022-06-30 16:00'}] = np.nan
    june = reduce(source.sel(time=slice(None, '2022-06-30 23:00')), '2022-06-01', '2022-06-30', 8, how)
    july = reduce(source.sel(time=slice('2022-06-30 16:00', None)), '2022-07-01', '2022-07-31', 8, how)
    joined = xr.concat([june, july], 'time')
    continuous = reduce(source, '2022-06-01', '2022-07-31', 8, how)
    xr.testing.assert_identical(joined, continuous)
    assert np.isnan(joined.sel(time='2022-07-01').item())
    assert joined.sel(time='2022-06-30').item() == (23 if how == 'max' else 11.5)
    assert joined.sel(time='2022-07-02').item() == (23 if how == 'max' else 11.5)


def test_naive_raw_concatenation_of_padded_months_is_rejected():
    june = hours('2022-06-01', '2022-06-30', 8)
    july = hours('2022-07-01', '2022-07-31', 8)
    joined = xr.concat([june, july], 'time')
    assert joined.time.to_index().duplicated().sum() == 8
    with pytest.raises(ValueError, match='Duplicate'):
        reduce(joined, '2022-06-01', '2022-07-31', 8, 'max')


def test_bjt_missing_first_boundary_hour_is_not_borrowed_from_previous_output():
    july = hours('2022-07-01', '2022-07-31', 8).isel(time=slice(1, None))
    with pytest.raises(ValueError, match='2022-07-01'):
        reduce(july, '2022-07-01', '2022-07-31', 8, 'max')

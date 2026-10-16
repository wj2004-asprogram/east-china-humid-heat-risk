"""Independent acceptance of the user-approved T-C1 missing/day contract.

Set EHHR_SCIENCE_SOURCE_ROOT to test exported modules from a pinned older
commit. Expected results below are fixed values or elementary window/area
calculations; production helpers never generate an expected result.
"""

import importlib.util
import os
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr


LABELS = ['No Risk', 'Caution', 'Extreme Caution', 'Danger', 'Extreme Danger']


@pytest.fixture(scope='module')
def api():
    root = Path(os.environ.get('EHHR_SCIENCE_SOURCE_ROOT', Path(__file__).parents[2]))
    modules = {}
    for name in ('risk', 'time_utils'):
        spec = importlib.util.spec_from_file_location(f'acceptance_{name}', root / 'src' / f'{name}.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
    return SimpleNamespace(**modules)


def hourly(day='2024-02-29', offset=8, days=1, coord='time'):
    first = pd.Timestamp(day) - pd.Timedelta(hours=offset)
    return xr.DataArray(
        np.arange(24 * days, dtype=float), dims=coord,
        coords={coord: pd.date_range(first, periods=24 * days, freq='h')},
    )


def category_grid(values):
    return xr.DataArray(
        np.asarray(values, dtype=np.int8), dims=('date', 'latitude', 'longitude'),
        coords={'date': pd.date_range('2024-02-29', periods=len(values)),
                'latitude': [0., 60.], 'longitude': [118., 119.]},
    )


class TestF1Missing:
    def test_nan_is_missing_and_finite_thresholds_keep_their_meaning(self, api):
        hi = xr.DataArray([np.nan, 20., 26.7, 32.2, 40.6, 54.4])
        np.testing.assert_array_equal(api.risk.classify_heat_risk(hi), [-1, 0, 1, 2, 3, 4])

    @pytest.mark.parametrize('how', ['max', 'mean'])
    @pytest.mark.parametrize('offset', [0, 8])
    @pytest.mark.parametrize('coord', ['time', 'valid_time'])
    def test_one_missing_hour_invalidates_only_its_cell_day(self, api, how, offset, coord):
        base = hourly(offset=offset, days=2, coord=coord)
        values = np.stack([base.values, base.values + 100], axis=1)
        values[5, 0] = np.nan
        da = xr.DataArray(values, dims=(coord, 'cell'), coords={coord: base[coord], 'cell': ['a', 'b']})
        got = api.time_utils.local_daily(da, coord, offset, how, date(2024, 2, 29), date(2024, 3, 1))
        assert np.isnan(got.values[0, 0])
        expected = [[np.nan, 123.], [47., 147.]] if how == 'max' else [[np.nan, 111.5], [35.5, 135.5]]
        np.testing.assert_allclose(got, expected, equal_nan=True)

    @pytest.mark.parametrize('how', ['max', 'mean'])
    def test_all_missing_day_remains_missing(self, api, how):
        da = hourly()
        da[:] = np.nan
        got = api.time_utils.local_daily(da, 'time', 8, how, date(2024, 2, 29), date(2024, 2, 29))
        assert np.isnan(got.item())
        assert api.risk.classify_heat_risk(got).item() == -1

    def test_valid_land_denominator_and_coverage_have_distinct_meanings(self, api):
        # Land weights 1 + 1 + 0.5 = 2.5; valid land 1 + 0.5 = 1.5.
        # Last cell is sea, even though its category is Extreme Danger.
        cat = category_grid([[[3, -1], [0, 4]]])
        land = xr.DataArray([[True, True], [True, False]], dims=('latitude', 'longitude'),
                            coords={'latitude': [0., 60.], 'longitude': [118., 119.]})
        row = api.risk.compute_area_fractions(cat, 'latitude', 'longitude', 'date', land).iloc[0]
        assert 'Valid Land Coverage' in row.index, 'coverage is a required output field'
        assert row['Valid Land Coverage'] == pytest.approx(60.)
        assert row['Danger'] == pytest.approx(100 * 2 / 3)
        assert row['No Risk'] == pytest.approx(100 / 3)
        assert row['Extreme Danger'] == pytest.approx(0.)
        assert sum(row[label] for label in LABELS) == pytest.approx(100.)

    def test_coverage_is_computed_separately_for_each_day(self, api):
        cat = category_grid([[[3, -1], [0, 4]], [[-1, -1], [-1, -1]], [[0, 0], [0, 0]]])
        df = api.risk.compute_area_fractions(cat, 'latitude', 'longitude', 'date')
        assert 'Valid Land Coverage' in df.columns
        # All-grid weights 3; first-day valid weights 2, then 0, then 3.
        np.testing.assert_allclose(df['Valid Land Coverage'], [100 * 2 / 3, 0., 100.])
        assert df.loc[1, LABELS].isna().all()
        assert df.loc[2, 'No Risk'] == pytest.approx(100.)

    def test_no_valid_land_is_nan_categories_even_if_sea_is_valid(self, api):
        cat = category_grid([[[-1, 4], [-1, 3]]])
        land = xr.DataArray([[True, False], [True, False]], dims=('latitude', 'longitude'),
                            coords={'latitude': [0., 60.], 'longitude': [118., 119.]})
        row = api.risk.compute_area_fractions(cat, 'latitude', 'longitude', 'date', land).iloc[0]
        assert 'Valid Land Coverage' in row.index
        assert row['Valid Land Coverage'] == pytest.approx(0.)
        assert row[LABELS].isna().all()


class TestF2TimeIntegrity:
    @pytest.mark.parametrize('defect', ['replace_last_with_duplicate', 'duplicate_extra',
                                      'half_hour', 'all_half_hours',
                                      'duplicate_outside_period', 'half_hour_outside_period'])
    def test_invalid_input_is_rejected_including_edges(self, api, defect):
        da = hourly()
        times = list(pd.DatetimeIndex(da.time.values))
        if defect == 'replace_last_with_duplicate':
            times[-1] = times[-2]
        elif defect == 'duplicate_extra':
            times.insert(10, times[10])
        elif defect == 'half_hour':
            times[10] += pd.Timedelta(minutes=30)
        elif defect == 'all_half_hours':
            times = [t + pd.Timedelta(minutes=30) for t in times]
        elif defect == 'duplicate_outside_period':
            before = times[0] - pd.Timedelta(hours=1)
            times = [before, before] + times
        else:
            times = [times[0] - pd.Timedelta(minutes=30)] + times
        da = xr.DataArray(np.zeros(len(times)), dims='time', coords={'time': times})
        with pytest.raises(ValueError):
            api.time_utils.local_daily(da, 'time', 8, 'max', date(2024, 2, 29), date(2024, 2, 29))

    @pytest.mark.parametrize('missing', ['first', 'last', 'middle', 'whole_day'])
    def test_in_period_incomplete_data_is_rejected(self, api, missing):
        da = hourly(days=3)
        keep = np.ones(72, dtype=bool)
        if missing == 'whole_day':
            keep[24:48] = False
        else:
            keep[{'first': 0, 'last': 71, 'middle': 36}[missing]] = False
        with pytest.raises(ValueError):
            api.time_utils.local_daily(da.isel(time=keep), 'time', 8, 'max', date(2024, 2, 29), date(2024, 3, 2))


class TestCalendarWindows:
    @pytest.mark.parametrize('start,days', [('2022-07-31', 2), ('2022-12-31', 2),
                                           ('2024-02-28', 3), ('2024-02-29', 1)])
    @pytest.mark.parametrize('offset', [0, 8])
    @pytest.mark.parametrize('how', ['max', 'mean'])
    def test_calendar_windows_against_fixed_24_hour_blocks(self, api, start, days, offset, how):
        da = hourly(start, offset, days)
        first = date.fromisoformat(start)
        got = api.time_utils.local_daily(da, 'time', offset, how, first, first + timedelta(days=days - 1))
        expected = [24 * d + (23 if how == 'max' else 11.5) for d in range(days)]
        np.testing.assert_allclose(got, expected)
        assert list(pd.DatetimeIndex(got.time).date) == [first + timedelta(days=i) for i in range(days)]

    def test_first_last_and_next_midnight_spikes_do_not_leak(self, api):
        da = hourly('2022-12-31', 8, 2)
        da[:] = 0.
        da.values[[0, 23, 24, 47]] = [4., 8., 16., 32.]
        # Add incomplete outside days containing larger spikes and NaN.
        edge = xr.DataArray([np.nan, 999.], dims='time', coords={'time':
            [pd.Timestamp('2022-12-30 15:00'), pd.Timestamp('2023-01-01 16:00')]})
        da = xr.concat([edge.isel(time=slice(0, 1)), da, edge.isel(time=slice(1, 2))], 'time')
        got = api.time_utils.local_daily(da, 'time', 8, 'max', date(2022, 12, 31), date(2023, 1, 1))
        np.testing.assert_array_equal(got, [8., 32.])

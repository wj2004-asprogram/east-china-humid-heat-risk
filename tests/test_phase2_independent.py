"""Fixed-commit Phase 2 production modules, independent calendar/value references."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src import climatology, download_era5, pipeline, process_temperature
from src.hourly_input import iter_hourly_blocks
from tests.phase2_support import hourly, make_config, rewrite, write_inputs
from tools.validate_era5 import sha256


@pytest.mark.parametrize('offset', [0, 8])
def test_actual_month_reader_uses_literal_utc_windows_and_season_gaps(tmp_path, offset):
    _, cfg = make_config(tmp_path, offset)
    write_inputs(cfg)
    blocks = list(iter_hourly_blocks(cfg))
    assert len(blocks) == 6
    assert [len(b.data.valid_time) for b in blocks] == [720, 744, 744, 720, 744, 744]
    first = '2003-05-31 16:00' if offset == 8 else '2003-06-01 00:00'
    july = '2003-06-30 16:00' if offset == 8 else '2003-07-01 00:00'
    last = '2004-08-31 15:00' if offset == 8 else '2004-08-31 23:00'
    assert pd.Timestamp(blocks[0].data.valid_time.values[0]) == pd.Timestamp(first)
    assert pd.Timestamp(blocks[1].data.valid_time.values[0]) == pd.Timestamp(july)
    assert pd.Timestamp(blocks[-1].data.valid_time.values[-1]) == pd.Timestamp(last)
    assert all(pd.DatetimeIndex(b.data.valid_time.values).is_unique for b in blocks)


@pytest.mark.parametrize('offset', [0, 8])
def test_actual_processing_preserves_boundary_values_city_date_and_run_info(tmp_path, offset):
    _, cfg = make_config(tmp_path, offset)
    write_inputs(cfg)
    before = {p.name: sha256(p) for p in cfg.paths.raw_land_sea_mask.parent.iterdir()}
    for name in ['run_info', 'process_temperature', 'plot_city_tmax', 'process_humid_heat',
                 'classify_heat_risk', 'heat_index_climatology']:
        pipeline.STEPS[name](cfg)
    with xr.open_dataset(cfg.paths.daily_tmax_nc) as ds:
        point = ds.t2m.sel(latitude=30.5, longitude=118.)
        assert len(point.valid_time) == 184
        assert point.sel(valid_time='2003-07-01').item() == pytest.approx((322 if offset == 8 else 302.3) - 273.15, abs=1e-4)
        assert point.sel(valid_time='2003-08-31').item() == pytest.approx((302.3 if offset == 8 else 324) - 273.15, abs=1e-4)
        assert point.sel(valid_time='2003-06-01').item() == pytest.approx((320 if offset == 8 else 302.3) - 273.15, abs=1e-4)
    csv = pd.read_csv(cfg.paths.city_tmax_csv)
    assert 'date' in csv and 'time' not in csv
    assert csv['date'].tolist()[0] == '2003-06-01'
    assert csv['date'].tolist()[-1] == '2004-08-31'
    assert len(csv) == 184 and csv['date'].nunique() == 184
    info = json.loads(cfg.paths.run_info_json.read_text(encoding='utf-8'))
    assert info['period']['years'] == {'first': 2003, 'last': 2004}
    assert info['period']['months'] == [6, 7, 8] and info['period']['days'] == 184
    assert info['utc_offset_hours'] == offset
    with xr.open_dataset(cfg.paths.climatology_nc) as ds:
        assert ds.year.values.tolist() == [2003, 2004]
        assert ds.day_of_year.values.tolist() == list(range(152, 244))
    assert {p.name: sha256(p) for p in cfg.paths.raw_land_sea_mask.parent.iterdir()} == before


def test_actual_reader_propagates_shared_month_nan_to_exactly_one_local_day(tmp_path):
    _, cfg = make_config(tmp_path, first=2003, last=2003)
    write_inputs(cfg)
    path = cfg.paths.raw_land_sea_mask.parent / 'era5_east_china_2003_06.nc'
    def missing(ds):
        ds.t2m.loc[{'valid_time': '2003-06-30 16:00', 'latitude': 30.5, 'longitude': 118.}] = np.nan
        return ds
    rewrite(path, missing)
    process_temperature.run(cfg)
    with xr.open_dataset(cfg.paths.daily_tmax_nc) as ds:
        point = ds.t2m.sel(latitude=30.5, longitude=118.)
        assert point.valid_time.values[np.isnan(point.values)].astype('datetime64[D]').astype(str).tolist() == ['2003-07-01']
        assert np.isfinite(point.sel(valid_time='2003-06-30').item())


def test_climatology_missing_training_date_and_future_values_remain_isolated():
    dates = pd.to_datetime(['2003-06-01', '2003-06-02', '2004-06-01', '2022-06-01', '2022-06-02'])
    daily = xr.DataArray([10., 40., 30., 999., 999.], dims='date', coords={'date': dates}, attrs={'units': 'Celsius'})
    by_year = climatology.by_year_and_day(daily)
    trained = climatology.climatology_for_years(by_year, [2003, 2004])
    assert trained.sel(day_of_year=152).item() == 20.
    assert np.isnan(trained.sel(day_of_year=153).item())
    changed = by_year.copy(deep=True)
    changed.loc[{'year': 2022}] = -999.
    xr.testing.assert_identical(climatology.climatology_for_years(changed, [2003, 2004]), trained)


class IndependentCDS:
    """Independent asynchronous server; its jobs survive client replacement."""

    def __init__(self, monkeypatch, interrupt=False, omit_hour=False, jobs=None):
        self.calls, self.interrupt, self.omit_hour = [], interrupt, omit_hour
        self.jobs = {} if jobs is None else jobs
        self.submissions, self.lookups = [], []
        owner = self
        class Client:
            def __init__(self, capture, options=None):
                self.capture = capture
                self.options = options

            def matching_jobs(self, dataset, request):
                return [ident for ident, entry in owner.jobs.items()
                        if entry == (dataset, request)]

            def submit(self, dataset, request):
                ident = f'independent-job-{len(owner.jobs) + 1}'
                owner.jobs[ident] = (dataset, request)
                owner.submissions.append(ident)
                self.capture(ident)
                return self.remote(ident)

            def remote(self, ident):
                assert ident in owner.jobs, 'Recovery must query an existing job'
                owner.lookups.append(ident)
                return SimpleNamespace(request_id=ident, status='successful')

            def download(self, remote, partial, store, ident):
                _, request = owner.jobs[remote.request_id]
                owner.calls.append(str(partial))
                if owner.interrupt:
                    with open(partial, 'wb') as stream: stream.write(b'synthetic interrupted transfer')
                    raise KeyboardInterrupt()
                times = pd.to_datetime([f'{request["year"][0]}-{request["month"][0]}-{day} {hour}'
                                        for day in request['day'] for hour in request['time']])
                if owner.omit_hour: times = times[1:]
                hourly(times).to_netcdf(partial)
        monkeypatch.setattr(download_era5, 'CDSClient', Client)


@pytest.mark.parametrize('failure', ['interrupt', 'missing_hour'])
@pytest.mark.parametrize('workers', [1, 3])
def test_actual_downloader_recovers_one_missing_file_without_rewriting_accepted_inputs(tmp_path, monkeypatch, failure, workers):
    _, cfg = make_config(tmp_path, first=2003, last=2003)
    name = 'era5_east_china_2003_07.nc'
    write_inputs(cfg, omit=[name])
    folder = cfg.paths.raw_land_sea_mask.parent
    before = {p.name: sha256(p) for p in folder.iterdir()}
    failed = IndependentCDS(monkeypatch, interrupt=failure == 'interrupt', omit_hour=failure == 'missing_hour')
    with pytest.raises(KeyboardInterrupt if failure == 'interrupt' else RuntimeError):
        download_era5.download_raw_files(cfg, workers=workers)
    assert len(failed.calls) == 1
    assert len(failed.submissions) == 1
    assert not (folder / name).exists()
    assert (folder / (name + '.part')).exists()
    recovered = IndependentCDS(monkeypatch, jobs=failed.jobs)
    download_era5.download_raw_files(cfg, workers=workers)
    assert len(recovered.calls) == 1 and recovered.calls[0].endswith(name + '.part')
    assert recovered.submissions == []
    assert recovered.lookups and set(recovered.lookups) == set(failed.submissions)
    assert (folder / name).exists() and not (folder / (name + '.part')).exists()
    assert {name: sha256(folder / name) for name in before} == before
    recovered.calls.clear()
    download_era5.download_raw_files(cfg, workers=workers)
    assert recovered.calls == []

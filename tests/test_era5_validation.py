"""T-C3 acceptance: corrupt synthetic files, do not copy validator logic."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.config import load_config
from tools.validate_era5 import Options, inventory, main, sha256, validate


def config_file(folder, start='2022-07-01', end='2022-07-01', offset=8, name='sample'):
    path = folder / f'{name}.yaml'
    path.write_text(f'''name: {name}
period: {{start: "{start}", end: "{end}"}}
region: {{north: 30.5, west: 118, south: 30, east: 118.5}}
cities:
  - {{name: TestCity, lat: 30.25, lon: 118.25, color: blue}}
utc_offset_hours: {offset}
land_sea_mask_threshold: 0.5
file_tag: {name}
figures_dir: figures/{name}
''', encoding='utf-8')
    return path


def small_era5():
    times = pd.date_range('2022-06-30 16:00', '2022-07-01 23:00', freq='h')
    shape = (32, 3, 3)
    return xr.Dataset({
        't2m': (('valid_time', 'latitude', 'longitude'), np.full(shape, 300.), {'units': 'K'}),
        'd2m': (('valid_time', 'latitude', 'longitude'), np.full(shape, 295.), {'units': 'K'}),
    }, coords={'valid_time': times, 'latitude': [30.5, 30.25, 30.], 'longitude': [118., 118.25, 118.5]})


def small_mask():
    return xr.Dataset({'lsm': (('valid_time', 'latitude', 'longitude'),
                              np.array([[[1., .5, 0.]] * 3]))},
                      coords={'valid_time': [pd.Timestamp('2022-07-01')],
                              'latitude': [30.5, 30.25, 30.], 'longitude': [118., 118.25, 118.5]})


def write_inputs(folder, raw=None, mask=None):
    cfg_path = config_file(folder)
    cfg = load_config(cfg_path)
    raw_path = folder / cfg.paths.raw_era5.name
    mask_path = folder / cfg.paths.raw_land_sea_mask.name
    (small_era5() if raw is None else raw).to_netcdf(raw_path)
    (small_mask() if mask is None else mask).to_netcdf(mask_path)
    return cfg_path, raw_path, mask_path


def status(report, name, role='hourly'):
    matches = [check for item in report['files'] if item['role'] == role
               for check in item['checks'] if check['id'] == name]
    assert len(matches) == 1, (name, matches)
    return matches[0]


def test_good_bjt_file_distinguishes_required_and_producer_hours(tmp_path):
    cfg, raw, mask = write_inputs(tmp_path)
    before = [sha256(raw), sha256(mask)]
    report = validate([cfg], tmp_path)
    assert report['summary']['status'] == 'PASS'
    assert report['files'][0]['time']['required_hours'] == 24
    assert report['files'][0]['time']['producer_hours'] == 32
    assert report['files'][0]['time']['producer_extra_hours'] == 8
    assert report['files'][1]['mask']['land_percent'] == pytest.approx(200 / 3)
    assert [sha256(raw), sha256(mask)] == before
    assert all(f['sha256_before'] == f['sha256_after'] for f in report['files'])


@pytest.mark.parametrize('defect,check_id', [
    ('missing_middle', 'time.required_coverage'), ('missing_bjt_first', 'time.required_coverage'),
    ('duplicate', 'time.unique'), ('missing_and_duplicate', 'time.unique'),
    ('off_hour', 'time.on_hour'), ('unsorted', 'time.sorted'),
    ('wrong_units', 't2m.units'), ('celsius_disguised_as_k', 't2m.range'),
    ('shifted_grid', 'grid.bounds'), ('nan_block', 't2m.missing'),
    ('infinite', 't2m.finite'), ('dewpoint_above_air', 'dewpoint.relationship'),
])
def test_each_anomaly_is_reported_in_its_own_check(tmp_path, defect, check_id):
    raw = small_era5()
    if defect == 'missing_middle':
        raw = raw.isel(valid_time=[i for i in range(32) if i != 12])
    elif defect == 'missing_bjt_first':
        raw = raw.isel(valid_time=slice(1, None))
    elif defect == 'duplicate':
        raw = raw.isel(valid_time=[0, *range(32)])
    elif defect == 'missing_and_duplicate':
        raw = raw.isel(valid_time=[*range(23), 22, *range(24, 32)])
    elif defect == 'off_hour':
        times = raw.valid_time.values.copy()
        times[4] += np.timedelta64(30, 'm')
        raw = raw.assign_coords(valid_time=times)
    elif defect == 'unsorted':
        raw = raw.isel(valid_time=[1, 0, *range(2, 32)])
    elif defect == 'wrong_units':
        raw.t2m.attrs['units'] = 'degC'
    elif defect == 'celsius_disguised_as_k':
        raw.t2m[:] = 27.
    elif defect == 'shifted_grid':
        raw = raw.assign_coords(longitude=raw.longitude + .125)
    elif defect == 'nan_block':
        raw.t2m.values[3:5, 1:, 1:] = np.nan
    elif defect == 'infinite':
        raw.t2m.values[3, 1, 1] = np.inf
    else:
        raw.d2m.values[7, 1, 1] = 302.
    cfg, _, _ = write_inputs(tmp_path, raw=raw)
    report = validate([cfg], tmp_path)
    assert report['summary']['status'] == 'FAIL'
    finding = status(report, check_id)
    assert finding['status'] == 'FAIL'
    if defect == 'nan_block':
        summary = report['files'][0]['variables']['t2m']
        assert summary['nan_count'] == 8
        assert summary['by_time'] == [
            {'utc': '2022-06-30 19:00:00', 'nan_cells': 4},
            {'utc': '2022-06-30 20:00:00', 'nan_cells': 4}]
        assert len(summary['by_grid']) == 4
        assert all(row['nan_hours'] == 2 for row in summary['by_grid'])
    if defect in {'missing_middle', 'missing_bjt_first'}:
        assert finding['details']['count'] == 1
        assert len(finding['details']['examples']) == 1
    if defect == 'dewpoint_above_air':
        assert finding['details']['offending_samples'] == 1
        assert finding['details']['by_grid'] == [{'latitude': 30.25, 'longitude': 118.25, 'hours': 1}]


@pytest.mark.parametrize('defect', ['out_of_range', 'nan', 'different_grid', 'multiple_times'])
def test_land_mask_anomalies(tmp_path, defect):
    mask = small_mask()
    if defect == 'out_of_range':
        mask.lsm.values[0, 1, 1] = 1.1
    elif defect == 'nan':
        mask.lsm.values[0, 1, 1] = np.nan
    elif defect == 'different_grid':
        mask = mask.assign_coords(longitude=mask.longitude + .125)
    else:
        mask = xr.concat([mask, mask.assign_coords(valid_time=[pd.Timestamp('2022-07-02')])], 'valid_time')
    cfg, _, _ = write_inputs(tmp_path, mask=mask)
    report = validate([cfg], tmp_path)
    assert report['summary']['status'] == 'FAIL'
    if defect in {'out_of_range', 'nan'}:
        assert status(report, 'mask.values', 'land_sea_mask')['status'] == 'FAIL'
    elif defect == 'different_grid':
        assert status(report, 'grid.mask_match')['status'] == 'FAIL'
    else:
        assert status(report, 'mask.single_time', 'land_sea_mask')['status'] == 'FAIL'


def test_missing_mask_is_incomplete_instead_of_full_pass(tmp_path):
    cfg, _, mask = write_inputs(tmp_path)
    mask.unlink()  # synthetic test fixture only
    report = validate([cfg], tmp_path)
    assert report['summary']['status'] == 'INCOMPLETE'
    assert report['summary']['counts']['NOT_RUN'] == 2


def test_unreadable_final_file_is_environment_or_format_error(tmp_path):
    cfg, raw, _ = write_inputs(tmp_path)
    raw.write_bytes(b'not a NetCDF file')  # deliberately corrupt this synthetic fixture
    report = validate([cfg], tmp_path)
    assert report['summary']['status'] == 'ERROR'
    assert status(report, 'file.inspect')['status'] == 'ERROR'
    assert status(report, 'file.unchanged')['status'] == 'PASS'


def test_chunk_size_does_not_change_data_findings(tmp_path):
    raw = small_era5()
    raw.d2m.values[[2, 19, 31], 1, 2] = np.nan
    cfg, _, _ = write_inputs(tmp_path, raw=raw)
    a = validate([cfg], tmp_path, Options(hours_per_chunk=1))
    b = validate([cfg], tmp_path, Options(hours_per_chunk=13))
    assert a['summary'] == b['summary']
    assert a['files'] == b['files']


def test_alias_coordinate_and_variable_names(tmp_path):
    raw = small_era5().rename({'valid_time': 'time', 'latitude': 'lat', 'longitude': 'lon',
                               't2m': '2m_temperature', 'd2m': '2m_dewpoint_temperature'})
    mask = small_mask().rename({'valid_time': 'time', 'latitude': 'lat', 'longitude': 'lon', 'lsm': 'land_sea_mask'})
    cfg, _, _ = write_inputs(tmp_path, raw=raw, mask=mask)
    assert validate([cfg], tmp_path)['summary']['status'] == 'PASS'


def test_non_datetime_time_is_not_silently_guessed(tmp_path):
    raw = small_era5().assign_coords(valid_time=np.arange(32))
    cfg, _, _ = write_inputs(tmp_path, raw=raw)
    report = validate([cfg], tmp_path)
    assert report['summary']['status'] == 'ERROR'
    assert 'datetime64' in status(report, 'file.inspect')['message']


def test_inventory_detects_missing_month_without_inventing_winter(tmp_path):
    paths = [config_file(tmp_path, start, end, name=name) for start, end, name in
             [('2020-06-01', '2020-06-30', 'jun20'), ('2020-08-01', '2020-08-31', 'aug20'),
              ('2021-06-01', '2021-08-31', 'jja21')]]
    findings = inventory([load_config(path) for path in paths], Options())
    first, second = findings
    assert first['status'] == 'FAIL'
    assert first['details']['missing']['count'] == 31
    assert second['status'] == 'PASS'
    assert second['details']['local_days'] == 92


def test_duplicate_local_days_are_inventory_failure(tmp_path):
    cfg = load_config(config_file(tmp_path))
    finding = inventory([cfg, cfg], Options())[0]
    assert finding['status'] == 'FAIL'
    assert finding['details']['duplicate']['count'] == 1


def test_cli_writes_both_reports_without_touching_inputs(tmp_path):
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    cfg, raw, mask = write_inputs(inputs)
    before = [sha256(raw), sha256(mask)]
    output = tmp_path / 'reports' / 'report.json'
    assert main(['--config', str(cfg), '--raw-dir', str(inputs), '--output', str(output)]) == 0
    assert output.exists() and output.with_suffix('.md').exists()
    assert [sha256(raw), sha256(mask)] == before
    with pytest.raises(SystemExit) as error:
        main(['--config', str(cfg), '--raw-dir', str(inputs), '--output', str(output)])
    assert error.value.code == 2  # refuses report overwrite


def test_cli_refuses_report_inside_input_directory(tmp_path):
    cfg, raw, mask = write_inputs(tmp_path)
    before = [sha256(raw), sha256(mask)]
    with pytest.raises(SystemExit) as error:
        main(['--config', str(cfg), '--raw-dir', str(tmp_path), '--output', str(tmp_path / 'report.json')])
    assert error.value.code == 2
    assert [sha256(raw), sha256(mask)] == before


def test_nonfinite_grid_is_reportable_json_not_a_serialization_failure(tmp_path):
    import json
    raw = small_era5().assign_coords(latitude=[30.5, np.nan, 30.])
    cfg, _, _ = write_inputs(tmp_path, raw=raw)
    report = validate([cfg], tmp_path)
    assert report['summary']['status'] == 'FAIL'
    assert status(report, 'grid.finite')['status'] == 'FAIL'
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('overrides', [{'hours_per_chunk': 0}, {'grid_step': np.nan},
                                      {'dewpoint_tolerance_k': -1}, {'temperature_min_k': 350}])
def test_invalid_options_are_rejected_before_input_reads(overrides):
    with pytest.raises(ValueError, match='options'):
        validate([], options=replace(Options(), **overrides))

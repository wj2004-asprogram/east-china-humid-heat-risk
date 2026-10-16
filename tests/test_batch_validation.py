"""M1: explicit expectations, deliberate omissions and read-only checkpoint review."""

import calendar
import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.config import load_config
from tools.validate_batch import audit, create_manifest, expected_periods, main, write_json_new
from tools.validate_era5 import sha256


def make_config(root, year=2022, month=7, offset=8):
    folder = root / 'configs'
    folder.mkdir(exist_ok=True)
    period = f'{year}-{month:02d}'
    path = folder / f'{period}.yaml'
    path.write_text(f'''name: {period}
period: {{start: "{period}-01", end: "{period}-{calendar.monthrange(year, month)[1]}"}}
region: {{north: 30.5, west: 118, south: 30, east: 118.5}}
cities: []
utc_offset_hours: {offset}
land_sea_mask_threshold: 0.5
file_tag: {year}_{month:02d}_{offset}
figures_dir: figures/synthetic
''', encoding='utf-8')
    return path


def write_month(root, config, daily=True):
    cfg = load_config(config, project_root=root)
    raw_dir = root / 'raw'
    raw_dir.mkdir(exist_ok=True)
    first = pd.Timestamp(cfg.start_date) - pd.Timedelta(hours=cfg.utc_offset_hours)
    last = pd.Timestamp(cfg.end_date) + pd.Timedelta(hours=23)
    times = pd.date_range(first, last, freq='h')
    shape = (len(times), 3, 3)
    ds = xr.Dataset({'t2m': (('time', 'latitude', 'longitude'), np.full(shape, 300., dtype='float32'), {'units': 'K'}),
                     'd2m': (('time', 'latitude', 'longitude'), np.full(shape, 295., dtype='float32'), {'units': 'K'})},
                    coords={'time': times, 'latitude': [30.5, 30.25, 30.], 'longitude': [118., 118.25, 118.5]})
    raw_path = raw_dir / cfg.paths.raw_era5.name
    ds.to_netcdf(raw_path)
    mask_path = raw_dir / cfg.paths.raw_land_sea_mask.name
    if not mask_path.exists():
        xr.Dataset({'lsm': (('latitude', 'longitude'), np.ones((3, 3), dtype='float32'))},
                   coords={'latitude': ds.latitude, 'longitude': ds.longitude}).to_netcdf(mask_path)
    if daily:
        folder = root / 'daily'
        folder.mkdir(exist_ok=True)
        days = pd.date_range(cfg.start_date, cfg.end_date, freq='D')
        xr.Dataset({'tmax': (('day', 'latitude', 'longitude'), np.full((len(days), 3, 3), 300.), {'units': 'K'})},
                   coords={'day': days, 'latitude': ds.latitude, 'longitude': ds.longitude}).to_netcdf(folder / f'{cfg.name}.nc')
    return raw_path


def make_case(root, years=(2022,), months=(6, 7, 8), daily=True, omitted=()):
    root.mkdir(exist_ok=True)
    bindings = []
    configs = {}
    for year in years:
        for month in months:
            period = f'{year}-{month:02d}'
            config = make_config(root, year, month)
            configs[period] = config
            binding = {'period': period, 'config': f'configs/{period}.yaml', 'raw_dir': 'raw'}
            if daily:
                binding['daily'] = {'path': f'daily/{period}.nc', 'time_coord': 'day', 'variables': {'tmax': 'K'}}
            bindings.append(binding)
            if period not in omitted:
                write_month(root, config, daily)
    plan = create_manifest(next(iter(configs.values())), years, months)
    plan['bindings'] = bindings
    manifest = root / 'manifest.json'
    write_json_new(manifest, plan)
    return manifest, plan, configs


def row(report, period='2022-07'):
    return next(r for r in report['matrix'] if r['period'] == period)


def finding(report, name, period=None):
    checks = row(report, period)['checks'] if period else report['checks']
    return next(c for c in checks if c['id'] == name)


def rewrite(path, change):
    with xr.open_dataset(path) as source:
        data = source.load()
    change(data).to_netcdf(path)


def test_unbound_25_year_plan_reports_all_75_missing_months(tmp_path):
    config = make_config(tmp_path)
    plan = create_manifest(config, range(2000, 2025), [6, 7, 8])
    manifest = tmp_path / 'plan.json'
    write_json_new(manifest, plan)
    report = audit(manifest)
    assert report['scope']['expected_months'] == 75
    assert report['scope']['expected_local_days'] == 2300
    assert report['summary']['status'] == 'INCOMPLETE'
    assert report['summary']['counts']['NOT_RUN'] == 75
    assert report['matrix'][0]['period'] == '2000-06'
    assert report['matrix'][-1]['period'] == '2024-08'


def test_full_summer_passes_with_explicit_daily_contract_and_no_mutation(tmp_path):
    manifest, _, _ = make_case(tmp_path)
    inputs = [p for p in tmp_path.rglob('*') if p.is_file()]
    before = {str(p): sha256(p) for p in inputs}
    report = audit(manifest)
    assert report['summary']['status'] == 'PASS'
    assert report['summary']['ready_months'] == 3
    assert report['scope']['expected_local_days'] == 92
    assert row(report)['raw_validation']['files'][0]['time']['producer_hours'] == 752
    assert row(report)['raw_validation']['files'][0]['time']['required_hours'] == 744
    assert finding(report, 'daily.global_coverage')['status'] == 'PASS'
    assert {str(p): sha256(p) for p in inputs} == before
    json.dumps(report, allow_nan=False)


def test_entire_missing_year_and_month_do_not_disappear(tmp_path):
    manifest, plan, _ = make_case(tmp_path, years=(2020, 2022), months=(6, 7), daily=False)
    plan['bindings'] = [b for b in plan['bindings'] if b['period'] == '2022-06']
    manifest.write_text(json.dumps(plan), encoding='utf-8')
    report = audit(manifest)
    assert report['summary']['status'] == 'INCOMPLETE'
    assert [r['period'] for r in report['matrix'] if r['ready']] == ['2022-06']
    assert report['scope']['expected_local_days'] == 122
    assert report['summary']['counts']['NOT_RUN'] == 3


def test_separate_summers_do_not_require_winter_or_intermediate_year(tmp_path):
    manifest, _, _ = make_case(tmp_path, years=(2020, 2022), months=(6, 7, 8))
    report = audit(manifest)
    assert report['summary']['status'] == 'PASS'
    assert report['scope']['expected_local_days'] == 184
    assert len(report['matrix']) == 6


def test_part_file_does_not_count_as_a_final_file(tmp_path):
    manifest, _, configs = make_case(tmp_path, months=(7,), omitted=('2022-07',))
    cfg = load_config(configs['2022-07'])
    (tmp_path / 'raw').mkdir()
    (tmp_path / 'raw' / (cfg.paths.raw_era5.name + '.part')).write_bytes(b'partial and deliberately unreadable')
    report = audit(manifest)
    assert report['summary']['status'] == 'INCOMPLETE'
    assert report['summary']['counts']['ERROR'] == 0
    assert row(report)['raw_status'] == 'INCOMPLETE'


@pytest.mark.parametrize('defect', ['missing_first', 'duplicate', 'wrong_units', 'wrong_grid'])
def test_daily_month_seam_anomalies_are_located(tmp_path, defect):
    manifest, _, _ = make_case(tmp_path)
    path = tmp_path / 'daily' / '2022-07.nc'
    def change(ds):
        if defect == 'missing_first':
            return ds.isel(day=slice(1, None))
        if defect == 'duplicate':
            return ds.assign_coords(day=[pd.Timestamp('2022-06-30'), *pd.date_range('2022-07-02', '2022-07-31')])
        if defect == 'wrong_units':
            ds.tmax.attrs['units'] = 'degC'
            return ds
        return ds.assign_coords(longitude=ds.longitude + .125)
    rewrite(path, change)
    report = audit(manifest)
    assert report['summary']['status'] == 'FAIL'
    expected = {'missing_first': 'daily.dates', 'duplicate': 'daily.dates',
                'wrong_units': 'daily.tmax.units', 'wrong_grid': 'daily.grid.bounds'}[defect]
    assert finding(report, expected, '2022-07')['status'] == 'FAIL'
    if defect in {'missing_first', 'duplicate'}:
        assert finding(report, 'daily.global_coverage')['status'] == 'FAIL'


def test_daily_duplicate_date_is_not_hidden_by_row_count(tmp_path):
    manifest, _, _ = make_case(tmp_path, months=(7,))
    path = tmp_path / 'daily' / '2022-07.nc'
    dates = list(pd.date_range('2022-07-01', '2022-07-31'))
    dates[-1] = dates[-2]
    rewrite(path, lambda ds: ds.assign_coords(day=dates))
    report = audit(manifest)
    detail = finding(report, 'daily.dates', '2022-07')['details']
    assert detail['observed'] == detail['expected'] == 31
    assert detail['duplicate'] == ['2022-07-30 00:00:00']
    assert detail['missing'] == ['2022-07-31 00:00:00']


def test_daily_nan_is_permitted_without_claiming_value_equivalence(tmp_path):
    manifest, _, _ = make_case(tmp_path, months=(7,))
    def missing(ds):
        ds.tmax.values[0, 1, 1] = np.nan
        return ds
    rewrite(tmp_path / 'daily' / '2022-07.nc', missing)
    assert audit(manifest)['summary']['status'] == 'PASS'


def test_resume_from_missing_to_complete_and_unchanged_inputs(tmp_path):
    manifest, _, configs = make_case(tmp_path, omitted=('2022-07',))
    first = audit(manifest)
    assert first['summary']['status'] == 'INCOMPLETE'
    previous = tmp_path / 'first.json'
    write_json_new(previous, first)
    write_month(tmp_path, configs['2022-07'])
    second = audit(manifest, previous)
    assert second['summary']['status'] == 'PASS'
    assert row(second, '2022-06')['resume_state'] == 'UNCHANGED_VALID'
    assert row(second)['resume_state'] == 'NEW_VALID'
    previous2 = tmp_path / 'second.json'
    write_json_new(previous2, second)
    third = audit(manifest, previous2)
    assert {r['resume_state'] for r in third['matrix']} == {'UNCHANGED_VALID'}


@pytest.mark.parametrize('changed', ['raw', 'mask', 'daily', 'config'])
def test_changed_previously_valid_input_requires_review(tmp_path, changed):
    manifest, _, configs = make_case(tmp_path, months=(7,))
    first = audit(manifest)
    previous = tmp_path / 'previous.json'
    write_json_new(previous, first)
    cfg = load_config(configs['2022-07'])
    if changed == 'config':
        with configs['2022-07'].open('a', encoding='utf-8') as stream:
            stream.write('\n# altered configuration bytes\n')
    else:
        path, variable = {'raw': (tmp_path / 'raw' / cfg.paths.raw_era5.name, 't2m'),
                          'mask': (tmp_path / 'raw' / cfg.paths.raw_land_sea_mask.name, 'lsm'),
                          'daily': (tmp_path / 'daily' / '2022-07.nc', 'tmax')}[changed]
        def modify(ds):
            ds[variable].values.flat[0] -= .01
            return ds
        rewrite(path, modify)
    current = audit(manifest, previous)
    assert current['summary']['status'] == 'FAIL'
    assert current['summary']['resume_review_months'] == 1
    assert row(current)['resume_state'] == 'CHANGED_INPUT'
    assert finding(current, 'resume.inputs_unchanged', '2022-07')['status'] == 'FAIL'


def test_mismatched_config_fails_before_raw_validation(tmp_path, monkeypatch):
    import tools.validate_batch as module
    manifest, _, configs = make_case(tmp_path, months=(7,))
    config = configs['2022-07']
    config.write_text(config.read_text(encoding='utf-8').replace('utc_offset_hours: 8', 'utc_offset_hours: 0'), encoding='utf-8')
    def forbidden(*args, **kwargs):
        pytest.fail('Do not interpret raw data using a mismatched contract')
    monkeypatch.setattr(module, 'validate', forbidden)
    report = audit(manifest)
    assert finding(report, 'config.contract', '2022-07')['status'] == 'FAIL'


@pytest.mark.parametrize('defect', ['duplicate_year', 'boolean_month', 'bad_grid', 'unknown_field', 'duplicate_binding', 'unexpected_month'])
def test_invalid_manifests_are_rejected(tmp_path, defect):
    manifest, plan, _ = make_case(tmp_path, months=(7,))
    if defect == 'duplicate_year': plan['years'] *= 2
    elif defect == 'boolean_month': plan['months'] = [True]
    elif defect == 'bad_grid': plan['contract']['grid_step'] = float('nan')
    elif defect == 'unknown_field': plan['monts'] = [7]
    elif defect == 'duplicate_binding': plan['bindings'] *= 2
    else: plan['bindings'][0]['period'] = '2022-08'
    with pytest.raises(ValueError):
        expected_periods(plan)


def test_previous_report_for_changed_manifest_is_rejected(tmp_path):
    manifest, plan, _ = make_case(tmp_path, months=(7,))
    previous = tmp_path / 'previous.json'
    write_json_new(previous, audit(manifest))
    plan['years'].append(2023)
    manifest.write_text(json.dumps(plan), encoding='utf-8')
    with pytest.raises(ValueError, match='identical manifest'):
        audit(manifest, previous)


def test_cli_does_not_overwrite_reports_or_write_inside_inputs(tmp_path):
    manifest, _, _ = make_case(tmp_path / 'inputs', months=(7,))
    output = tmp_path / 'reports' / 'accepted.json'
    assert main(['audit', '--manifest', str(manifest), '--output', str(output)]) == 0
    original = output.read_bytes()
    for destination in [output, manifest.parent / 'raw' / 'bad.json', manifest.parent / 'daily' / 'bad.json']:
        with pytest.raises(SystemExit) as error:
            main(['audit', '--manifest', str(manifest), '--output', str(destination)])
        assert error.value.code == 2
    assert output.read_bytes() == original
    assert output.with_suffix('.md').exists()


def test_cli_plan_is_unbound_and_does_not_create_production_configs(tmp_path):
    config = make_config(tmp_path)
    before = set(tmp_path.rglob('*'))
    output = tmp_path / 'plan.json'
    assert main(['plan', '--template-config', str(config), '--years', '2022', '2024', '--output', str(output)]) == 0
    plan = json.loads(output.read_text(encoding='utf-8'))
    assert expected_periods(plan) == ['2022-06', '2022-07', '2022-08', '2024-06', '2024-07', '2024-08']
    assert plan['bindings'] == []
    assert set(tmp_path.rglob('*')) - before == {output}


def test_missing_config_does_not_become_an_environment_error(tmp_path):
    manifest, _, configs = make_case(tmp_path, months=(7,))
    configs['2022-07'].unlink()  # This test's own synthetic fixture.
    report = audit(manifest)
    assert report['summary']['status'] == 'INCOMPLETE'
    assert finding(report, 'config.available', '2022-07')['status'] == 'NOT_RUN'


def test_corrupt_daily_file_is_error_and_global_dates_are_not_claimed_checked(tmp_path):
    manifest, _, _ = make_case(tmp_path, months=(7,))
    (tmp_path / 'daily' / '2022-07.nc').write_bytes(b'corrupt synthetic daily file')
    report = audit(manifest)
    assert report['summary']['status'] == 'ERROR'
    assert finding(report, 'daily.inspect', '2022-07')['status'] == 'ERROR'
    assert finding(report, 'daily.global_coverage')['status'] == 'NOT_RUN'


def test_raw_only_audit_does_not_claim_daily_or_batch_execution(tmp_path):
    manifest, _, _ = make_case(tmp_path, months=(7,), daily=False)
    report = audit(manifest)
    assert report['summary']['status'] == 'PASS'
    assert row(report)['daily_status'] == 'NOT_REQUESTED'
    assert report['scope']['daily_requested'] is False
    assert report['scope']['production_batch_executed'] is False


def test_partial_daily_binding_is_not_full_acceptance(tmp_path):
    manifest, plan, _ = make_case(tmp_path)
    del plan['bindings'][1]['daily']
    manifest.write_text(json.dumps(plan), encoding='utf-8')
    report = audit(manifest)
    assert report['summary']['status'] == 'INCOMPLETE'
    assert finding(report, 'daily.global_coverage')['status'] == 'NOT_RUN'


def test_two_month_configs_cannot_alias_one_final_raw_file(tmp_path):
    manifest, _, configs = make_case(tmp_path, months=(6, 7), daily=False)
    path = configs['2022-07']
    path.write_text(path.read_text(encoding='utf-8').replace('file_tag: 2022_07_8', 'file_tag: 2022_06_8'), encoding='utf-8')
    report = audit(manifest)
    assert report['summary']['status'] == 'FAIL'
    assert finding(report, 'binding.unique_raw', '2022-07')['status'] == 'FAIL'


def test_duplicate_json_field_is_rejected_instead_of_silently_changing_scope(tmp_path):
    manifest = tmp_path / 'duplicate.json'
    manifest.write_text('{"years": [2022], "years": [2024]}', encoding='utf-8')
    with pytest.raises(ValueError, match='Duplicate JSON field'):
        audit(manifest)


def test_malformed_previous_report_is_a_clear_configuration_error(tmp_path):
    manifest, _, _ = make_case(tmp_path, months=(7,))
    previous = tmp_path / 'bad-previous.json'
    previous.write_text('[]', encoding='utf-8')
    with pytest.raises(ValueError, match='identical manifest'):
        audit(manifest, previous)


def test_repeated_audit_cannot_silently_accept_changed_inputs(tmp_path):
    manifest, _, configs = make_case(tmp_path, months=(7,), daily=False)
    initial = tmp_path / 'accepted.json'
    accepted = audit(manifest)
    write_json_new(initial, accepted)
    cfg = load_config(configs['2022-07'])
    raw = tmp_path / 'raw' / cfg.paths.raw_era5.name
    def changed(ds):
        ds.t2m.values[0, 0, 0] += .25
        return ds
    rewrite(raw, changed)
    latest = initial
    for index in range(2):
        report = audit(manifest, latest)
        assert report['summary']['status'] == 'FAIL'
        assert row(report)['resume_state'] == 'CHANGED_INPUT'
        assert row(report)['accepted_input_sha256'] == row(accepted)['input_sha256']
        latest = tmp_path / f'pending-{index}.json'
        write_json_new(latest, report)
    raw.unlink()  # This test's own fixture; simulate a temporary missing final file.
    missing = audit(manifest, latest)
    latest = tmp_path / 'missing.json'
    write_json_new(latest, missing)
    write_month(tmp_path, configs['2022-07'], daily=False)
    rewrite(raw, changed)
    returned = audit(manifest, latest)
    assert returned['summary']['status'] == 'FAIL'
    assert row(returned)['resume_state'] == 'CHANGED_INPUT'
    fresh = audit(manifest)
    assert fresh['summary']['status'] == 'PASS'
    assert row(fresh)['resume_state'] == 'NEW_VALID'

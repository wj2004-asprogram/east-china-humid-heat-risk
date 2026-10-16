"""Acceptance of the adapter's actual physical/logical file contracts."""

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from tests.phase2_support import make_config, rewrite, write_inputs
from tools.validate_batch import write_json_new
from tools.validate_era5 import sha256
from tools.validate_multiyear import main, physical_plan, source_catalogue_matches, validate_multiyear


def entry(report, name):
    return next(e for e in report['files'] if e['path'].replace('\\', '/').endswith('/' + name))


def block(report, period):
    return next(b for b in report['logical_months'] if b['period'] == period)


@pytest.mark.parametrize('offset,edges', [(0, 0), (8, 25)])
def test_25_year_catalogue_has_literal_counts_and_hours(tmp_path, offset, edges):
    config, cfg = make_config(tmp_path, offset, first=2000, last=2024)
    specs = physical_plan(cfg, tmp_path)
    assert len(specs) == 75 + edges
    assert [len(p['times']) for p in specs[:3]] == [720, 744, 744]
    assert specs[0]['path'].name == 'era5_east_china_2000_06.nc'
    assert specs[74]['path'].name == 'era5_east_china_2024_08.nc'
    if offset == 8:
        assert specs[75]['path'].name == 'era5_east_china_2000_05_31_h16-23.nc'
        assert specs[-1]['path'].name == 'era5_east_china_2024_05_31_h16-23.nc'
        assert specs[75]['times'].tolist() == list(pd.date_range('2000-05-31 16:00', periods=8, freq='h'))
    report = validate_multiyear(config, tmp_path / 'not_downloaded')
    assert report['scope']['local_days'] == 2300
    assert len(report['files']) == 76 + edges
    assert report['summary']['status'] == 'INCOMPLETE'
    assert report['summary']['quality_ready_files'] == 0
    assert len(report['logical_months']) == 75


def test_catalogue_reference_does_not_trust_production_edge_hours(tmp_path):
    _, cfg = make_config(tmp_path, first=2003, last=2003)
    specs = physical_plan(cfg, tmp_path)
    altered = replace(cfg, raw_files=(*cfg.raw_files[:-1], replace(cfg.raw_files[-1], hours=tuple(range(15, 24)))))
    assert source_catalogue_matches(cfg, specs)
    assert not source_catalogue_matches(altered, specs)


def test_physical_utc_months_and_logical_bjt_dependencies_pass_without_mutation(tmp_path):
    config, cfg = make_config(tmp_path)
    write_inputs(cfg)
    folder = cfg.paths.raw_land_sea_mask.parent
    before = {p.name: sha256(p) for p in folder.iterdir()}
    report = validate_multiyear(config, folder)
    assert report['summary']['status'] == 'PASS'
    assert report['summary']['ready_local_months'] == 6
    assert report['summary']['quality_ready_files'] == 9
    july = entry(report, 'era5_east_china_2003_07.nc')
    assert july['time']['observed_hours'] == july['expected_hours'] == 744
    assert july['time']['utc_offset_hours'] == 0
    assert july['time']['expectation_mode'] == 'explicit_utc'
    june = block(report, '2003-06')
    assert june['first_utc'] == '2003-05-31 16:00:00'
    assert june['last_utc'] == '2003-06-30 15:00:00'
    july = block(report, '2003-07')
    assert july['required_hours'] == 744
    assert [p.split('\\')[-1] for p in july['dependencies']] == ['era5_east_china_2003_06.nc', 'era5_east_china_2003_07.nc']
    assert july['first_utc'] == '2003-06-30 16:00:00'
    assert july['last_utc'] == '2003-07-31 15:00:00'
    assert {p.name: sha256(p) for p in folder.iterdir()} == before
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('missing,affected', [
    ('era5_east_china_2003_05_31_h16-23.nc', ['2003-06']),
    ('era5_east_china_2003_06.nc', ['2003-06', '2003-07']),
])
def test_missing_physical_input_only_blocks_dependent_months(tmp_path, missing, affected):
    config, cfg = make_config(tmp_path, first=2003, last=2003)
    write_inputs(cfg, omit=[missing])
    folder = cfg.paths.raw_land_sea_mask.parent
    (folder / (missing + '.part')).write_bytes(b'partial synthetic file; do not inspect')
    report = validate_multiyear(config, folder)
    assert report['summary']['status'] == 'INCOMPLETE'
    assert [b['period'] for b in report['logical_months'] if b['status'] != 'PASS'] == affected
    assert report['summary']['counts']['ERROR'] == 0


@pytest.mark.parametrize('defect,check_id', [
    ('missing', 'time.required_coverage'), ('duplicate', 'time.unique'),
    ('off_hour', 'time.on_hour'), ('units', 't2m.units'), ('nan', 'd2m.missing'),
    ('nanosecond', 'time.on_hour'), ('nat', 'time.datetime'),
])
def test_eight_hour_edge_file_gets_full_quality_checks(tmp_path, defect, check_id):
    config, cfg = make_config(tmp_path, first=2003, last=2003)
    write_inputs(cfg)
    name = 'era5_east_china_2003_05_31_h16-23.nc'
    def corrupt(ds):
        if defect == 'missing': return ds.isel(valid_time=slice(1, None))
        if defect == 'duplicate': return ds.isel(valid_time=[0, *range(8)])
        if defect == 'off_hour': return ds.assign_coords(valid_time=ds.valid_time + np.timedelta64(30, 'm'))
        if defect in {'nanosecond', 'nat'}:
            times = ds.valid_time.values.astype('datetime64[ns]').copy()
            times[0] = times[0] + np.timedelta64(1, 'ns') if defect == 'nanosecond' else np.datetime64('NaT', 'ns')
            return ds.assign_coords(valid_time=times)
        if defect == 'units': ds.t2m.attrs['units'] = 'degC'
        else: ds.d2m.values[0, 0, 0] = np.nan
        return ds
    rewrite(cfg.paths.raw_land_sea_mask.parent / name, corrupt)
    report = validate_multiyear(config, cfg.paths.raw_land_sea_mask.parent)
    assert report['summary']['status'] == 'FAIL'
    assert next(c for c in entry(report, name)['checks'] if c['id'] == check_id)['status'] == 'FAIL'
    assert block(report, '2003-07')['status'] == 'PASS'


def test_shared_input_changes_propagate_to_both_dependent_months_and_stay_pending(tmp_path):
    config, cfg = make_config(tmp_path, first=2003, last=2003)
    write_inputs(cfg)
    folder = cfg.paths.raw_land_sea_mask.parent
    previous = tmp_path / 'accepted.json'
    write_json_new(previous, validate_multiyear(config, folder))
    name = 'era5_east_china_2003_06.nc'
    def modify(ds):
        ds.t2m.values[0, 0, 0] += .25
        return ds
    rewrite(folder / name, modify)
    for index in range(2):
        report = validate_multiyear(config, folder, previous)
        assert report['summary']['status'] == 'FAIL'
        assert report['summary']['changed_input_files'] == 1
        assert entry(report, name)['resume_state'] == 'CHANGED_INPUT'
        assert [b['period'] for b in report['logical_months'] if b['status'] != 'PASS'] == ['2003-06', '2003-07']
        previous = tmp_path / f'pending_{index}.json'
        write_json_new(previous, report)


def test_missing_then_completed_edge_is_new_valid_and_existing_inputs_unchanged(tmp_path):
    missing = 'era5_east_china_2003_05_31_h16-23.nc'
    config, cfg = make_config(tmp_path, first=2003, last=2003)
    write_inputs(cfg, omit=[missing])
    folder = cfg.paths.raw_land_sea_mask.parent
    previous = tmp_path / 'initial.json'
    write_json_new(previous, validate_multiyear(config, folder))
    # Use an independently literal time window to supply the missing file only.
    from tests.phase2_support import hourly
    hourly(pd.date_range('2003-05-31 16:00', periods=8, freq='h')).to_netcdf(folder / missing)
    report = validate_multiyear(config, folder, previous)
    assert report['summary']['status'] == 'PASS'
    assert entry(report, missing)['resume_state'] == 'NEW_VALID'
    assert entry(report, 'era5_east_china_2003_06.nc')['resume_state'] == 'UNCHANGED_VALID'


def test_changed_policy_rejects_previous_acceptance(tmp_path):
    config, cfg = make_config(tmp_path, first=2003, last=2003)
    write_inputs(cfg)
    folder = cfg.paths.raw_land_sea_mask.parent
    previous = tmp_path / 'accepted.json'
    write_json_new(previous, validate_multiyear(config, folder))
    with config.open('a', encoding='utf-8') as stream: stream.write('\n# changed bytes\n')
    with pytest.raises(ValueError, match='same config'):
        validate_multiyear(config, folder, previous)


def test_cli_report_guards_and_malformed_file_are_explicit(tmp_path):
    config, cfg = make_config(tmp_path, first=2003, last=2003)
    write_inputs(cfg)
    folder = cfg.paths.raw_land_sea_mask.parent
    (folder / 'era5_east_china_2003_06.nc').write_bytes(b'truncated synthetic final')
    report = validate_multiyear(config, folder)
    assert report['summary']['status'] == 'ERROR'
    output = tmp_path / 'report.json'
    assert main(['--config', str(config), '--raw-dir', str(folder), '--output', str(output)]) == 2
    for destination in [output, folder / 'bad.json']:
        with pytest.raises(SystemExit) as error:
            main(['--config', str(config), '--raw-dir', str(folder), '--output', str(destination)])
        assert error.value.code == 2


def test_single_period_config_is_not_misinterpreted_as_years(tmp_path):
    config, _ = make_config(tmp_path)
    text = config.read_text(encoding='utf-8')
    text = text.replace('years: {first: 2003, last: 2004}\n  months: [6, 7, 8]', 'start: "2003-07-01"\n  end: "2003-07-31"')
    config.write_text(text, encoding='utf-8')
    with pytest.raises(ValueError, match='multi-year'):
        validate_multiyear(config, tmp_path)

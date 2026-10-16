"""Explicit monthly inventory and read-only resume review; no batch execution."""

import argparse
import calendar
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.config import load_config
from tools.validate_era5 import Options, check, grid, sha256, validate


def require(condition, message):
    if not condition:
        raise ValueError(message)


def expected_periods(plan):
    require(isinstance(plan, dict) and type(plan.get('schema_version')) is int
            and plan['schema_version'] == 1, 'Unsupported manifest schema_version')
    require(not set(plan) - {'schema_version', 'years', 'months', 'contract', 'bindings', 'source_template'},
            'Unknown manifest fields')
    for key, low, high in [('years', 1900, 2200), ('months', 1, 12)]:
        values = plan.get(key)
        require(isinstance(values, list) and bool(values)
                and all(type(v) is int and low <= v <= high for v in values)
                and len(set(values)) == len(values), f'Invalid {key}: need unique integer values')
    contract = plan.get('contract', {})
    require(isinstance(contract, dict) and set(contract) ==
            {'region', 'utc_offset_hours', 'grid_step', 'land_sea_mask_threshold'}, 'Invalid contract fields')
    offset = contract['utc_offset_hours']
    require(type(offset) is int and -12 <= offset <= 14, 'Invalid utc_offset_hours')
    region = contract['region']
    require(isinstance(region, dict) and set(region) == {'north', 'west', 'south', 'east'}, 'Invalid region')
    numeric = list(region.values()) + [contract['grid_step'], contract['land_sea_mask_threshold']]
    require(all(type(v) in (int, float) and np.isfinite(v) for v in numeric), 'Nonfinite/non-numeric contract')
    require(-90 <= region['south'] < region['north'] <= 90
            and -180 <= region['west'] < region['east'] <= 360
            and contract['grid_step'] > 0 and 0 < contract['land_sea_mask_threshold'] <= 1,
            'Invalid region, grid step or land threshold')
    periods = [f'{year:04d}-{month:02d}' for year in sorted(plan['years']) for month in sorted(plan['months'])]
    require(isinstance(plan.get('bindings'), list), 'bindings must be a list')
    seen = set()
    for binding in plan['bindings']:
        require(isinstance(binding, dict) and set(binding) <= {'period', 'config', 'raw_dir', 'daily'}
                and {'period', 'config', 'raw_dir'} <= set(binding), 'Invalid binding fields')
        period = binding['period']
        require(isinstance(period, str) and period in periods and period not in seen,
                f'Duplicate or out-of-plan binding: {period}')
        seen.add(period)
        require(all(isinstance(binding[k], str) and binding[k].strip() for k in ['config', 'raw_dir']),
                'Bindings require explicit nonempty config/raw_dir paths')
        if 'daily' in binding:
            daily = binding['daily']
            require(isinstance(daily, dict) and set(daily) == {'path', 'time_coord', 'variables'},
                    'daily requires path, time_coord and variables')
            require(all(isinstance(daily[k], str) and daily[k].strip() for k in ['path', 'time_coord'])
                    and isinstance(daily['variables'], dict) and bool(daily['variables'])
                    and all(isinstance(k, str) and k and isinstance(v, str) and v
                            for k, v in daily['variables'].items()), 'Invalid daily variable/unit contract')
    return periods


def create_manifest(template_config, years, months, grid_step=0.25):
    """Copy an existing contract, without inventing future production paths."""
    cfg = load_config(template_config)
    plan = {'schema_version': 1, 'years': list(years), 'months': list(months),
            'contract': {'region': {k: getattr(cfg, k) for k in ['north', 'west', 'south', 'east']},
                         'utc_offset_hours': cfg.utc_offset_hours, 'grid_step': grid_step,
                         'land_sea_mask_threshold': cfg.land_sea_mask_threshold},
            'bindings': [], 'source_template': {'path': str(Path(template_config).resolve()),
                                                'sha256': sha256(template_config)}}
    expected_periods(plan)
    return plan


def read_json(path):
    def invalid(value):
        raise ValueError(f'Nonfinite JSON constant: {value}')
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f'Duplicate JSON field: {key}')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=invalid, object_pairs_hook=pairs)


def resolve_path(base, value):
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def month_dates(period):
    year, month = map(int, period.split('-'))
    return pd.date_range(f'{period}-01', periods=calendar.monthrange(year, month)[1], freq='D')


def inspect_daily(path, spec, cfg, options):
    """Check explicitly labelled local daily fields; missing values are permitted."""
    checks, dates = [], []
    entry = {'path': str(path), 'checks': checks, 'sha256_before': None, 'sha256_after': None}
    if not path.is_file():
        checks.append({'id': 'daily.available', 'status': 'NOT_RUN', 'message': 'Final daily file absent', 'details': {}})
        return entry, dates
    before = path.stat()
    try:
        entry['sha256_before'] = sha256(path)
        with xr.open_dataset(path) as ds:
            name = spec['time_coord']
            require(name in ds.coords and ds[name].ndim == 1
                    and np.issubdtype(ds[name].dtype, np.datetime64), 'Daily coordinate must be 1D datetime64')
            times = pd.DatetimeIndex(ds[name].values)
            wanted = pd.date_range(cfg.start_date, cfg.end_date, freq='D')
            check(checks, 'daily.dates', times.equals(wanted), 'Exactly the configured local dates, in order',
                  missing=[str(t) for t in wanted.difference(times)],
                  unexpected=[str(t) for t in times.difference(wanted)],
                  duplicate=[str(t) for t in times[times.duplicated()]], observed=len(times), expected=len(wanted))
            grid_checks = []
            lat_name, lon_name, _, _ = grid(ds, cfg, grid_checks, options)
            checks += [dict(c, id='daily.' + c['id']) for c in grid_checks]
            for variable, units in spec['variables'].items():
                require(variable in ds.data_vars, f'Missing daily variable: {variable}')
                da = ds[variable]
                check(checks, f'daily.{variable}.dimensions', set(da.dims) == {name, lat_name, lon_name},
                      'Daily grid variable has time/latitude/longitude dimensions')
                check(checks, f'daily.{variable}.units', da.attrs.get('units') == units,
                      'Daily units match the explicit contract', expected=units, actual=str(da.attrs.get('units')))
            dates = [str(t) for t in times]
    except Exception as error:
        checks.append({'id': 'daily.inspect', 'status': 'ERROR', 'message': f'{type(error).__name__}: {error}', 'details': {}})
    finally:
        try:
            entry['sha256_after'] = sha256(path)
            after = path.stat()
            check(checks, 'daily.unchanged', entry['sha256_before'] == entry['sha256_after']
                  and before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns,
                  'Daily input hash, size and modification time unchanged')
        except OSError as error:
            checks.append({'id': 'daily.unchanged', 'status': 'ERROR', 'message': str(error), 'details': {}})
    return entry, dates


def audit(manifest_path, previous_report=None):
    manifest_path = Path(manifest_path).resolve()
    manifest_hash = sha256(manifest_path)
    plan = read_json(manifest_path)
    periods = expected_periods(plan)
    base, contract = manifest_path.parent, plan['contract']
    options = Options(grid_step=contract['grid_step'])
    bindings = {b['period']: b for b in plan['bindings']}
    previous = {}
    if previous_report is not None:
        prior = read_json(previous_report)
        require(isinstance(prior, dict) and prior.get('kind') == 'ehhr_batch_audit' and prior.get('schema_version') == 1
                and prior.get('manifest_sha256') == manifest_hash, 'Previous report must use the identical manifest')
        require(isinstance(prior.get('matrix'), list), 'Invalid previous report matrix')
        for row in prior['matrix']:
            require(isinstance(row, dict) and row.get('period') in periods and row['period'] not in previous
                    and type(row.get('ready')) is bool and isinstance(row.get('input_sha256'), dict)
                    and isinstance(row.get('accepted_input_sha256'), dict),
                    'Invalid previous report row')
            require(not row['ready'] or bool(row['input_sha256']), 'Ready previous row has no input hashes')
            previous[row['period']] = row
        require(set(previous) == set(periods), 'Previous report must contain every planned period')
    report = {'schema_version': 1, 'kind': 'ehhr_batch_audit',
              'created_at_utc': datetime.now(timezone.utc).isoformat(), 'mode': 'read_only',
              'manifest_path': str(manifest_path), 'manifest_sha256': manifest_hash,
              'previous_report': str(Path(previous_report).resolve()) if previous_report else None,
              'contract': contract, 'matrix': [], 'checks': [],
              'scope': {'expected_months': len(periods), 'expected_local_days': sum(len(month_dates(p)) for p in periods),
                        'daily_requested': any('daily' in b for b in bindings.values()),
                        'production_batch_executed': False}}
    raw_paths, daily_dates, daily_contracts = {}, [], set()
    for period in periods:
        row = {'period': period, 'local_days': len(month_dates(period)), 'ready': False,
               'raw_status': 'NOT_RUN', 'daily_status': 'NOT_REQUESTED',
               'resume_state': 'NOT_READY', 'input_sha256': {},
               'accepted_input_sha256': previous.get(period, {}).get('accepted_input_sha256', {}), 'checks': []}
        report['matrix'].append(row)
        binding = bindings.get(period)
        if binding is None:
            row['checks'].append({'id': 'binding.available', 'status': 'NOT_RUN',
                                  'message': 'Expected month has no explicit config/file binding', 'details': {}})
            continue
        config_path = resolve_path(base, binding['config'])
        folder = resolve_path(base, binding['raw_dir'])
        row.update(config_path=str(config_path), raw_dir=str(folder))
        try:
            config_hash = sha256(config_path)
            row['input_sha256'][str(config_path)] = config_hash
            cfg = load_config(config_path)
            wanted = month_dates(period)
            compatible = (str(cfg.start_date) == str(wanted[0].date())
                          and str(cfg.end_date) == str(wanted[-1].date())
                          and cfg.utc_offset_hours == contract['utc_offset_hours']
                          and cfg.land_sea_mask_threshold == contract['land_sea_mask_threshold']
                          and all(np.isclose(getattr(cfg, k), v, rtol=0, atol=options.coordinate_atol)
                                  for k, v in contract['region'].items()))
            check(row['checks'], 'config.contract', compatible, 'Full target month, region, offset and land threshold match plan')
            if not compatible:
                continue
            raw_path = folder / cfg.paths.raw_era5.name
            other = raw_paths.setdefault(str(raw_path), period)
            check(row['checks'], 'binding.unique_raw', other == period, 'Different target months cannot alias one final hourly file', other_period=other)
            row['raw_validation'] = validate([config_path], folder, options)
            row['raw_status'] = row['raw_validation']['summary']['status']
            row['checks'] += row['raw_validation']['checks']
            for entry in row['raw_validation']['files']:
                row['checks'] += entry['checks']
                if entry['sha256_after'] is not None:
                    row['input_sha256'][entry['path']] = entry['sha256_after']
            if 'daily' in binding:
                spec = binding['daily']
                path = resolve_path(base, spec['path'])
                entry, dates = inspect_daily(path, spec, cfg, options)
                row['daily_validation'] = entry
                row['checks'] += entry['checks']
                row['daily_status'] = overall(entry['checks'])
                daily_dates += dates
                daily_contracts.add(tuple(sorted(spec['variables'].items())))
                if entry['sha256_after'] is not None:
                    row['input_sha256'][str(path)] = entry['sha256_after']
            check(row['checks'], 'config.unchanged', config_hash == sha256(config_path), 'Configuration unchanged while validating')
            row['ready'] = overall(row['checks']) == 'PASS'
        except FileNotFoundError as error:
            row['checks'].append({'id': 'config.available', 'status': 'NOT_RUN', 'message': str(error), 'details': {}})
        except Exception as error:
            row['checks'].append({'id': 'binding.inspect', 'status': 'ERROR',
                                  'message': f'{type(error).__name__}: {error}', 'details': {}})
        if row['ready']:
            baseline = row['accepted_input_sha256']
            if baseline:
                same = row['input_sha256'] == baseline
                row['resume_state'] = 'UNCHANGED_VALID' if same else 'CHANGED_INPUT'
                check(row['checks'], 'resume.inputs_unchanged', same,
                      'Inputs must match the last accepted baseline; changed inputs remain pending review across repeated audits')
            else:
                row['resume_state'] = 'NEW_VALID'
                row['accepted_input_sha256'] = dict(row['input_sha256'])
    if report['scope']['daily_requested']:
        all_bound = all(p in bindings and 'daily' in bindings[p] for p in periods)
        all_inspected = all(any(c['id'] == 'daily.dates' for c in r['checks']) for r in report['matrix'])
        if all_bound and all_inspected:
            wanted = [str(t) for p in periods for t in month_dates(p)]
            check(report['checks'], 'daily.global_coverage', daily_dates == wanted,
                  'Monthly local dates concatenate once in calendar order; seasonal gaps remain intentional',
                  observed_days=len(daily_dates), expected_days=len(wanted))
            check(report['checks'], 'daily.variable_contract', len(daily_contracts) == 1,
                  'Daily variable names and units consistent across months')
        else:
            report['checks'].append({'id': 'daily.global_coverage', 'status': 'NOT_RUN',
                                     'message': 'Daily dates not inspected for every planned month; global comparison not run', 'details': {}})
    check(report['checks'], 'manifest.unchanged', manifest_hash == sha256(manifest_path), 'Manifest unchanged while validating')
    findings = report['checks'] + [c for r in report['matrix'] for c in r['checks']]
    counts = Counter(c['status'] for c in findings)
    report['summary'] = {'status': overall(findings), 'counts': {s: counts[s] for s in ['PASS', 'FAIL', 'ERROR', 'NOT_RUN']},
                         'ready_months': sum(r['ready'] for r in report['matrix']),
                         'resume_review_months': sum(r['resume_state'] == 'CHANGED_INPUT' for r in report['matrix'])}
    return report


def overall(checks):
    statuses = {c['status'] for c in checks}
    return next((result for status, result in [('ERROR', 'ERROR'), ('FAIL', 'FAIL'), ('NOT_RUN', 'INCOMPLETE')]
                 if status in statuses), 'PASS')


def markdown(report):
    lines = ['# Monthly inventory and resume review', '', f"Result: **{report['summary']['status']}**", '',
             f"Scope: {report['scope']['expected_months']} months / {report['scope']['expected_local_days']} local days.", '',
             'This is read-only validation; it does not run or resume the production batch.', '',
             '| Month | Local days | Raw checks | Daily checks | Resume review |', '|---|---:|---|---|---|']
    for row in report['matrix']:
        lines.append(f"| {row['period']} | {row['local_days']} | {row['raw_status']} | {row['daily_status']} | {row['resume_state']} |")
    lines += ['', '## Findings requiring review', '']
    for period, checks in [('global', report['checks'])] + [(r['period'], r['checks']) for r in report['matrix']]:
        for finding in checks:
            if finding['status'] != 'PASS':
                message = finding['message'].replace('\n', ' ')
                lines.append(f"- {period}: {finding['status']} {finding['id']}: {message}")
    lines += ['', 'JSON retains per-file checks, hashes, options, and anomaly locations.', '']
    return '\n'.join(lines)


def write_json_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    make = commands.add_parser('plan', help='Create an unbound expectation manifest; no production configs')
    make.add_argument('--template-config', type=Path, required=True)
    make.add_argument('--years', nargs='+', type=int, required=True)
    make.add_argument('--months', nargs='+', type=int, default=[6, 7, 8])
    make.add_argument('--grid-step', type=float, default=0.25)
    run = commands.add_parser('audit')
    run.add_argument('--manifest', type=Path, required=True)
    run.add_argument('--previous-report', type=Path)
    for command in [make, run]:
        command.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    try:
        require(output.suffix.lower() == '.json', '--output must end in .json')
        targets = [output] if args.command == 'plan' else [output, output.with_suffix('.md')]
        require(not any(p.exists() for p in targets), 'Reports/manifests must be new files')
        if args.command == 'plan':
            cfg = load_config(args.template_config)
            require(not output.is_relative_to(cfg.paths.raw_era5.parent.resolve()), 'Manifest must be outside raw input directory')
            value = create_manifest(args.template_config, args.years, args.months, args.grid_step)
        else:
            plan = read_json(args.manifest)
            expected_periods(plan)
            base = args.manifest.resolve().parent
            folders = [resolve_path(base, b['raw_dir']) for b in plan['bindings']]
            folders += [resolve_path(base, b['daily']['path']).parent for b in plan['bindings'] if 'daily' in b]
            require(not any(t.is_relative_to(f) for t in targets for f in folders), 'Reports must be outside all input directories')
            value = audit(args.manifest, args.previous_report)
        write_json_new(output, value)
        if args.command == 'audit':
            with output.with_suffix('.md').open('x', encoding='utf-8') as stream:
                stream.write(markdown(value))
            print(json.dumps(value['summary']))
            return {'PASS': 0, 'FAIL': 1, 'ERROR': 2, 'INCOMPLETE': 2}[value['summary']['status']]
        print(json.dumps({'expected_months': len(expected_periods(value)), 'bindings': 0}))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())

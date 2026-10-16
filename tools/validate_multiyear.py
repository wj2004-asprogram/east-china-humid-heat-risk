"""Read-only Phase 2 acceptance: physical UTC files and logical local months."""

import argparse
import calendar
from collections import Counter
from dataclasses import asdict, replace
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256 as digest
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from tools.validate_batch import overall, read_json, require, write_json_new
from tools.validate_era5 import Options, check, examples, inspect_file, sha256, validate


def physical_plan(cfg, raw_dir):
    """Independent calendar/name expectations from docs/raw_files.md.

    Do not use production build_raw_files or RawFile.utc_times to generate
    expected values. The production catalogue is compared with this plan.
    """
    require(cfg.is_multi_year, 'Use a Phase 2 years/months config; single-period mode remains in validate_era5')
    require(cfg.utc_offset_hours in (0, 8), 'This adapter accepts UTC and Beijing time only')
    raw_dir = Path(raw_dir).resolve()
    result = []
    covered = {(year, month) for year in cfg.years for month in cfg.months}
    for year in cfg.years:
        for month in cfg.months:
            first = pd.Timestamp(year, month, 1)
            last = pd.Timestamp(year, month, calendar.monthrange(year, month)[1], 23)
            result.append({'path': raw_dir / f'era5_east_china_{year:04d}_{month:02d}.nc',
                           'kind': 'monthly', 'times': pd.date_range(first, last, freq='h')})
    if cfg.utc_offset_hours == 8:
        for year in cfg.years:
            previous = date(year, cfg.months[0], 1) - timedelta(days=1)
            if (previous.year, previous.month) not in covered:
                result.append({'path': raw_dir / f'era5_east_china_{previous:%Y_%m_%d}_h16-23.nc',
                               'kind': 'edge', 'times': pd.date_range(f'{previous} 16:00', f'{previous} 23:00', freq='h')})
    return result


def source_catalogue_matches(cfg, plan):
    actual = {raw.path.name: raw.utc_times for raw in cfg.raw_files}
    expected = {item['path'].name: item['times'] for item in plan}
    return (len(actual) == len(cfg.raw_files) and set(actual) == set(expected)
            and all(actual[name].equals(expected[name]) for name in expected))


def validate_multiyear(config_path, raw_dir=None, previous_report=None, options=None):
    config_path = Path(config_path).resolve()
    config_hash = sha256(config_path)
    cfg = load_config(config_path)
    require(cfg.is_multi_year, 'A Phase 2 multi-year config is required')
    options = options or Options()
    validate([], options=options)  # Reuse option validation without reading any files.
    folder = Path(raw_dir).resolve() if raw_dir is not None else cfg.raw_files[0].path.parent
    plan = physical_plan(cfg, folder)
    mask_path = folder / cfg.paths.raw_land_sea_mask.name
    signature_data = {'config_sha256': config_hash, 'raw_dir': str(folder), 'options': asdict(options),
                      'files': [{'name': p['path'].name, 'start': str(p['times'][0]), 'end': str(p['times'][-1])} for p in plan],
                      'mask': mask_path.name}
    signature = digest(json.dumps(signature_data, sort_keys=True, allow_nan=False).encode()).hexdigest()
    previous = {}
    if previous_report is not None:
        old = read_json(previous_report)
        require(isinstance(old, dict) and old.get('schema_version') == 1 and old.get('kind') == 'ehhr_multiyear_validation'
                and old.get('signature') == signature, 'Previous report needs the same config, paths and options')
        require(isinstance(old.get('files'), list), 'Invalid previous file list')
        for entry in old['files']:
            require(isinstance(entry, dict) and isinstance(entry.get('path'), str)
                    and entry['path'] not in previous and 'accepted_sha256' in entry, 'Invalid previous file entry')
            value = entry['accepted_sha256']
            require(value is None or isinstance(value, str) and len(value) == 64
                    and all(c in '0123456789abcdef' for c in value), 'Invalid accepted hash')
            previous[entry['path']] = entry
        require(set(previous) == {str(p['path']) for p in plan} | {str(mask_path)}, 'Previous report has a different file inventory')
    report = {'schema_version': 1, 'kind': 'ehhr_multiyear_validation', 'signature': signature,
              'created_at_utc': datetime.now(timezone.utc).isoformat(), 'mode': 'read_only',
              'config_path': str(config_path), 'config_sha256': config_hash, 'raw_dir': str(folder),
              'options': asdict(options), 'previous_report': str(previous_report) if previous_report else None,
              'scope': {'monthly_files': sum(p['kind'] == 'monthly' for p in plan),
                        'edge_files': sum(p['kind'] == 'edge' for p in plan), 'mask_files': 1,
                        'local_months': len(cfg.years) * len(cfg.months),
                        'local_days': sum(calendar.monthrange(y, m)[1] for y in cfg.years for m in cfg.months),
                        'utc_offset_hours': cfg.utc_offset_hours, 'production_processing_executed': False},
              'checks': [], 'files': [], 'logical_months': []}
    check(report['checks'], 'catalogue.production_matches_reference', source_catalogue_matches(cfg, plan),
          'Production raw-file names/hours match independent calendar expectations')
    mask, mask_coords = inspect_file(mask_path, cfg, 'land_sea_mask', options)
    mask['kind'] = 'mask'
    physical_cfg = replace(cfg, utc_offset_hours=0)
    for spec in plan:
        entry, coords = inspect_file(spec['path'], physical_cfg, 'hourly', options,
                                     expected_times=(spec['times'], spec['times']))
        entry.update(kind=spec['kind'], expected_hours=len(spec['times']),
                     expected_first_utc=str(spec['times'][0]), expected_last_utc=str(spec['times'][-1]))
        if coords is None or mask_coords is None:
            entry['checks'].append({'id': 'grid.mask_match', 'status': 'NOT_RUN',
                                    'message': 'Raw or mask grid unavailable', 'details': {}})
        else:
            same = all(a.shape == b.shape and np.allclose(a, b, rtol=0, atol=options.coordinate_atol)
                       for a, b in zip(coords, mask_coords))
            check(entry['checks'], 'grid.mask_match', same, 'Physical input grid matches the shared mask')
        report['files'].append(entry)
    report['files'].append(mask)
    for entry in report['files']:
        entry['quality_status'] = overall(entry['checks'])
        accepted = previous.get(entry['path'], {}).get('accepted_sha256')
        entry['accepted_sha256'] = accepted
        entry['resume_state'] = 'NOT_READY'
        if entry['quality_status'] == 'PASS':
            if accepted is None:
                entry['accepted_sha256'] = entry['sha256_after']
                entry['resume_state'] = 'NEW_VALID'
            else:
                same = accepted == entry['sha256_after']
                entry['resume_state'] = 'UNCHANGED_VALID' if same else 'CHANGED_INPUT'
                check(entry['checks'], 'resume.inputs_unchanged', same,
                      'Input still matches last accepted baseline; changed inputs stay pending review')
        entry['status'] = overall(entry['checks'])
    by_path = {entry['path']: entry for entry in report['files']}
    for year in cfg.years:
        for month in cfg.months:
            first = date(year, month, 1)
            last = date(year, month, calendar.monthrange(year, month)[1])
            start = pd.Timestamp(first) - pd.Timedelta(hours=cfg.utc_offset_hours)
            end = pd.Timestamp(last) + pd.Timedelta(hours=23 - cfg.utc_offset_hours)
            wanted = pd.date_range(start, end, freq='h')
            needed = sorted([p for p in plan if p['times'][0] <= end and p['times'][-1] >= start],
                            key=lambda p: p['times'][0])
            entries = [by_path[str(p['path'])] for p in needed]
            block = {'period': f'{year:04d}-{month:02d}', 'first_utc': str(start), 'last_utc': str(end),
                     'required_hours': len(wanted), 'dependencies': [e['path'] for e in entries], 'checks': []}
            available = all('time' in e and 'observed_utc' in e['time'] for e in entries)
            if available:
                actual = pd.DatetimeIndex([pd.Timestamp(t) for e in entries for t in e['time']['observed_utc']])
                actual = actual[(actual >= start) & (actual <= end)]
                check(block['checks'], 'logical.utc_hours', actual.equals(wanted),
                      'Required local-month UTC hours present exactly once in order',
                      missing=examples(wanted.difference(actual), options),
                      duplicate=examples(actual[actual.duplicated()], options), observed_hours=len(actual))
            else:
                block['checks'].append({'id': 'logical.utc_hours', 'status': 'NOT_RUN',
                                        'message': 'A required physical time axis is unavailable', 'details': {}})
            dependencies_ready = all(e['status'] == 'PASS' for e in entries) and mask['status'] == 'PASS'
            block['checks'].append({'id': 'logical.inputs_accepted', 'status': 'PASS' if dependencies_ready else 'NOT_RUN',
                                    'message': 'Input quality and prior acceptance checked; no processing executed', 'details': {}})
            block['status'] = overall(block['checks'])
            report['logical_months'].append(block)
    check(report['checks'], 'config.unchanged', config_hash == sha256(config_path), 'Config unchanged while validating')
    findings = report['checks'] + [c for e in report['files'] for c in e['checks']] + [c for b in report['logical_months'] for c in b['checks']]
    counts = Counter(c['status'] for c in findings)
    report['summary'] = {'status': overall(findings), 'counts': {s: counts[s] for s in ['PASS', 'FAIL', 'ERROR', 'NOT_RUN']},
                         'quality_ready_files': sum(e['quality_status'] == 'PASS' for e in report['files']),
                         'ready_local_months': sum(b['status'] == 'PASS' for b in report['logical_months']),
                         'changed_input_files': sum(e['resume_state'] == 'CHANGED_INPUT' for e in report['files'])}
    return report


def markdown(report):
    lines = ['# Phase 2 physical inputs and logical month coverage', '', f"Result: **{report['summary']['status']}**", '',
             f"Scope: `{json.dumps(report['scope'])}`", '', '## Physical files', '',
             '| File | Kind | Quality | Resume review |', '|---|---|---|---|']
    lines += [f"| {Path(e['path']).name} | {e['kind']} | {e['quality_status']} | {e['resume_state']} |" for e in report['files']]
    lines += ['', '## Logical local-month windows', '', '| Month | First UTC | Last UTC | Hours | Status |', '|---|---|---|---:|---|']
    lines += [f"| {b['period']} | {b['first_utc']} | {b['last_utc']} | {b['required_hours']} | {b['status']} |" for b in report['logical_months']]
    lines += ['', '## Findings requiring review', '']
    groups = [('global', report['checks'])] + [(e['path'], e['checks']) for e in report['files']] + [(b['period'], b['checks']) for b in report['logical_months']]
    for label, checks in groups:
        for c in checks:
            if c['status'] != 'PASS':
                lines.append(f"- {label}: {c['status']} {c['id']}: {c['message']}")
    lines += ['', 'No download, production batch, daily numeric comparison, or output regeneration is executed.', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--raw-dir', type=Path)
    parser.add_argument('--previous-report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        require(cfg.is_multi_year, 'A Phase 2 multi-year config is required')
        folder = args.raw_dir.resolve() if args.raw_dir else cfg.raw_files[0].path.parent.resolve()
        output = args.output.resolve()
        targets = [output, output.with_suffix('.md')]
        require(output.suffix.lower() == '.json', '--output must end with .json')
        require(not any(t.exists() or t.is_relative_to(folder) for t in targets), 'Reports must be new files outside raw input directory')
        report = validate_multiyear(args.config, folder, args.previous_report)
        write_json_new(output, report)
        with output.with_suffix('.md').open('x', encoding='utf-8') as stream:
            stream.write(markdown(report))
        print(json.dumps(report['summary']))
        return {'PASS': 0, 'FAIL': 1, 'ERROR': 2, 'INCOMPLETE': 2}[report['summary']['status']]
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())

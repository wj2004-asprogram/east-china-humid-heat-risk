"""Reproduce M1-B with a missing edge, completion, reuse and changed shared input."""

import argparse
from pathlib import Path

import pandas as pd

from tests.phase2_support import hourly, make_config, rewrite, write_inputs
from tools.validate_batch import write_json_new
from tools.validate_multiyear import markdown, validate_multiyear


def run(output_dir):
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    config, cfg = make_config(root / 'inputs')
    name = 'era5_east_china_2004_05_31_h16-23.nc'
    write_inputs(cfg, omit=[name])
    folder = cfg.paths.raw_land_sea_mask.parent
    (folder / (name + '.part')).write_bytes(b'synthetic partial edge transfer')
    previous, cases = None, []
    for index, label in enumerate(['missing-edge', 'completed', 'unchanged', 'changed-shared-month']):
        if index == 1:
            hourly(pd.date_range('2004-05-31 16:00', periods=8, freq='h')).to_netcdf(folder / name)
        if index == 3:
            def modify(ds):
                ds.t2m.values[0, 0, 0] += .25
                return ds
            rewrite(folder / 'era5_east_china_2003_06.nc', modify)
        report = validate_multiyear(config, folder, previous)
        path = root / 'reports' / f'{index + 1}-{label}.json'
        write_json_new(path, report)
        with path.with_suffix('.md').open('x', encoding='utf-8') as stream: stream.write(markdown(report))
        cases.append({'case': label, 'summary': report['summary'],
                      'months_requiring_review': [b['period'] for b in report['logical_months'] if b['status'] != 'PASS']})
        previous = path
    statuses = [c['summary']['status'] for c in cases]
    if statuses != ['INCOMPLETE', 'PASS', 'PASS', 'FAIL']:
        raise AssertionError(statuses)
    write_json_new(root / 'demo-summary.json', {'synthetic_only': True, 'cases': cases})
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True, help='Must not already exist')
    args = parser.parse_args(argv)
    try:
        for case in run(args.output_dir): print(f"{case['case']}: {case['summary']['status']}")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

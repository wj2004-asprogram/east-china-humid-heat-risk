"""Package grid-level research evidence without changing the site-screening result.

Only the small set of primary and bounding-box candidate grids is read.  A
bounding box is deliberately conservative: its candidate grids do not establish
that the mapped factory polygon intersects each grid's nearest-point region.
"""

import argparse
import csv
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from tools.esg_site_screen import nearest_cell, sha256, summer_metrics


def candidate_indices(coordinates, lower, upper):
    """Original-order indices whose nearest-coordinate region meets an interval.

    Boundary contacts are included as conservative sensitivity candidates.  This
    does not change the tie policy used to select a site's primary grid.
    """
    values = np.asarray(coordinates, dtype=float)
    if (not np.isfinite(values).all() or len(np.unique(values)) != len(values)
            or not np.isfinite([lower, upper]).all() or lower > upper):
        raise ValueError('Invalid coordinate axis or bounding interval')
    if lower < values.min() or upper > values.max():
        raise ValueError('Factory bounding box extends beyond climate centre bounds')
    order = np.argsort(values)
    sorted_values = values[order]
    midpoints = (sorted_values[:-1] + sorted_values[1:]) / 2
    result = []
    for position, index in enumerate(order):
        left = midpoints[position - 1] if position else -np.inf
        right = midpoints[position] if position < len(midpoints) else np.inf
        if right >= lower and left <= upper:
            result.append(int(index))
    return result


def grid_metrics(dates, values, land_mask, threshold):
    """Use the accepted annual policy; withhold all climate values on non-land."""
    result = dict(land_mask=float(land_mask), annual=[], daily=[],
                  complete_summers=None, total_summers=None,
                  mean_days_ge_threshold_per_complete_summer=None,
                  max_consecutive_days=None)
    if land_mask != 1:
        result['screening_status'] = 'non_land_grid'
        return result
    dates = pd.DatetimeIndex(dates)
    values = np.asarray(values, dtype=float)
    annual = summer_metrics(dates, values, threshold)
    for row in annual:
        expected = pd.date_range(f'{row["year"]}-06-01', f'{row["year"]}-08-31')
        row['missing_date_days'] = int(len(expected.difference(dates)))
        row['nonfinite_days'] = int((~np.isfinite(values[dates.year == row['year']])).sum())
    complete = [row for row in annual if row['complete_summer']]
    result.update(
        screening_status='ok' if len(complete) == len(annual) else 'incomplete_climate_data',
        annual=annual,
        daily=[{'date': date.strftime('%Y-%m-%d'),
                'daily_max_heat_index_c': float(value) if np.isfinite(value) else None}
               for date, value in zip(dates, values)],
        complete_summers=len(complete), total_summers=len(annual),
        mean_days_ge_threshold_per_complete_summer=(
            round(float(np.mean([row['days_ge_threshold'] for row in complete])), 2)
            if complete else None),
        max_consecutive_days=(max(row['max_consecutive_days'] for row in complete)
                              if complete else None))
    return result


def load_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def json_safe_attributes(attributes):
    """Preserve NetCDF metadata, including NumPy scalar attributes, in JSON."""
    return json.loads(json.dumps(dict(attributes), default=lambda value: value.tolist()))


def write_csv(path, fields, records):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)


def build(catalog_path, boundary_path, screening_dir, heat_path, risk_path, output):
    inputs = {'catalog': catalog_path, 'boundaries': boundary_path,
              'screening_manifest': screening_dir / 'screening_manifest.json',
              'heat_index': heat_path, 'risk': risk_path}
    hashes = {key: sha256(path) for key, path in inputs.items()}
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    boundaries = json.loads(boundary_path.read_text(encoding='utf-8'))
    manifest = json.loads(inputs['screening_manifest'].read_text(encoding='utf-8'))
    for key in ['heat_index', 'risk']:
        if hashes[key] != manifest['input_sha256'][key]:
            raise ValueError(f'Existing climate manifest does not match {key}')
    for name, digest in manifest['outputs_sha256'].items():
        if sha256(screening_dir / name) != digest:
            raise ValueError(f'Screening output changed: {name}')
    summaries = {row['site_id']: row for row in load_csv(screening_dir / 'site_summary.csv')}
    annual_by_site_year = {(row['site_id'], int(row['year'])): row
                          for row in load_csv(screening_dir / 'site_annual.csv')}
    site_ids = {site['site_id'] for site in catalog['sites']}
    if len(site_ids) != len(catalog['sites']) or site_ids != set(summaries):
        raise ValueError('Catalog and existing screening site IDs disagree')
    if {feature['properties']['site_id'] for feature in boundaries['features']} != site_ids:
        raise ValueError('Factory boundary IDs do not match catalog')
    threshold = float(manifest['threshold_c'])
    grids, links, annual_checks = {}, [], 0
    with xr.open_dataset(heat_path) as heat, xr.open_dataset(risk_path) as risk:
        if heat.attrs.get('daily_aggregation') != 'local calendar days, UTC+8':
            raise ValueError('Explicit Beijing-time aggregation metadata is required')
        hi = heat.daily_max_heat_index_c.transpose('date', 'latitude', 'longitude')
        land = risk.land_mask.transpose('latitude', 'longitude')
        for axis in ['latitude', 'longitude']:
            if not np.array_equal(hi[axis].values, land[axis].values):
                raise ValueError('Climate and land-mask grids differ')
        dates = pd.DatetimeIndex(hi.date.values)
        if str(dates.min().date()) != manifest['first_date'] or str(dates.max().date()) != manifest['last_date']:
            raise ValueError('Date bounds differ from accepted screening manifest')
        latitudes, longitudes = hi.latitude.values, hi.longitude.values
        metadata = {
            'heat_index_global_attributes': json_safe_attributes(heat.attrs),
            'heat_index_variable_attributes': json_safe_attributes(hi.attrs),
            'risk_global_attributes': json_safe_attributes(risk.attrs),
            'land_mask_variable_attributes': json_safe_attributes(land.attrs),
            'land_mask_rule': land.attrs.get('description', ''),
            'metadata_interpretation': 'The derived variable units, long_name and description describe heat index. GRIB_* fields are retained original metadata and include raw temperature/K labels; no unit conversion is applied to this package.',
            'note': 'The source daily heat-index variable retains conflicting inherited GRIB temperature/K metadata. This package uses its derived units=Celsius, long_name and description, consistent with src/process_humid_heat.py heat-index processing. GRIB_* attributes are preserved for review; original NetCDF files and numeric values are unchanged, with no 273.15 conversion.'}
        wanted = {}
        for site in catalog['sites']:
            summary = summaries[site['site_id']]
            for key in ['latitude', 'longitude']:
                if float(summary[key]) != site[key]:
                    raise ValueError('Catalog coordinate changed after screening')
            i, j, _ = nearest_cell(latitudes, longitudes, site['latitude'], site['longitude'])
            primary_id = f'{latitudes[i]:.4f},{longitudes[j]:.4f}'
            if primary_id != summary['grid_id']:
                raise ValueError('Primary grid differs from accepted screening')
            west, south, east, north = site['bbox']
            bbox_ids = []
            for bi in candidate_indices(latitudes, south, north):
                for bj in candidate_indices(longitudes, west, east):
                    grid_id = f'{latitudes[bi]:.4f},{longitudes[bj]:.4f}'
                    bbox_ids.append(grid_id)
                    entry = wanted.setdefault(grid_id, {'indices': (bi, bj), 'primary_site_ids': [], 'bbox_site_ids': []})
                    entry['bbox_site_ids'].append(site['site_id'])
            if primary_id not in bbox_ids:
                raise ValueError('Primary grid is absent from conservative bbox candidates')
            wanted[primary_id]['primary_site_ids'].append(site['site_id'])
            links.append({'site_id': site['site_id'], 'primary_grid_id': primary_id,
                          'bbox_grid_ids': sorted(bbox_ids)})
        for grid_id, request in sorted(wanted.items()):
            i, j = request['indices']
            land_value = float(land.isel(latitude=i, longitude=j).values)
            # Do not load or publish climate values for rejected non-land grids.
            values = hi.isel(latitude=i, longitude=j).values if land_value == 1 else None
            record = {'grid_id': grid_id, 'latitude': float(latitudes[i]), 'longitude': float(longitudes[j]),
                      'analysis_role': 'primary' if request['primary_site_ids'] else 'sensitivity_only',
                      'primary_site_ids': sorted(request['primary_site_ids']),
                      'bbox_site_ids': sorted(request['bbox_site_ids']),
                      **grid_metrics(dates, values, land_value, threshold)}
            for site_id in record['primary_site_ids']:
                summary = summaries[site_id]
                if record['screening_status'] != summary['screening_status']:
                    raise ValueError('Primary climate status differs from accepted screening')
                for key in ['mean_days_ge_threshold_per_complete_summer', 'max_consecutive_days']:
                    expected = float(summary[key]) if summary[key] else None
                    if record[key] != expected:
                        raise ValueError(f'Primary summary differs: {site_id} {key}')
                for year in record['annual']:
                    previous = annual_by_site_year[(site_id, year['year'])]
                    if previous['complete_summer'] != str(year['complete_summer']):
                        raise ValueError(f'Primary completeness flag differs: {site_id} {year["year"]}')
                    if float(previous['threshold_c']) != threshold:
                        raise ValueError('Annual threshold differs from accepted screening')
                    for key in ['expected_days', 'valid_days', 'days_ge_threshold', 'max_consecutive_days']:
                        expected = float(previous[key]) if previous[key] else None
                        if year[key] != expected:
                            raise ValueError(f'Primary annual result differs: {site_id} {year["year"]} {key}')
                    annual_checks += 1
            # Independent grouping checks the consecutive-day count of each complete year.
            if land_value == 1:
                for year in record['annual']:
                    if not year['complete_summer']:
                        continue
                    hits = (np.asarray(values, dtype=float)[dates.year == year['year']] >= threshold).tolist()
                    longest = max((sum(1 for _ in group) for hit, group in itertools.groupby(hits) if hit), default=0)
                    if sum(hits) != year['days_ge_threshold'] or longest != year['max_consecutive_days']:
                        raise ValueError('Independent annual grouping verification failed')
            grids[grid_id] = record
    if annual_checks != len(annual_by_site_year):
        raise ValueError('Research records do not exhaust accepted site-year results')
    if any(sha256(path) != hashes[key] for key, path in inputs.items()):
        raise RuntimeError('A research input changed during packaging')
    result = {
        'schema_version': 1, 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'evidence_snapshot_date': catalog['reviewed_on'], 'threshold_c': threshold,
        'calendar': {'first_date': manifest['first_date'], 'last_date': manifest['last_date'],
                     'season': 'JJA', 'expected_days_per_summer': 92, 'timezone': 'UTC+8'},
        'policies': {
            'analysis_unit': 'Unique ERA5 grid; sites sharing a grid are not independent climate samples.',
            'bbox_candidates': 'Conservative bounding-box sensitivity candidates, not confirmed factory-polygon coverage or a position confidence interval.',
            'primary_result': 'The accepted representative-point grid and its result are preserved. No replacement by an alternative land grid.',
            'non_land': 'land_mask != 1: annual and daily climate values withheld, not zero-filled.',
            'incomplete_summer': 'Incomplete summers remain in annual records and are excluded from summary metrics.',
            'daily_precision': 'Decoded NetCDF daily values are serialized without display rounding; non-finite values become JSON null.',
            'interpretation': 'Historical environmental conditions at current mapped locations; no site existence, employee exposure, trend significance or occupational-compliance inference.'},
        'input_sha256': hashes, 'grids': list(grids.values()), 'site_grid_links': links,
        'boundaries': boundaries, 'source_metadata': metadata,
        'validation': {'primary_site_count': len(links), 'primary_annual_records_matched': annual_checks,
                       'unique_primary_grids': len({link['primary_grid_id'] for link in links}),
                       'unique_candidate_grids': len(grids),
                       'unique_primary_land_grids': sum(grid['land_mask'] == 1 and bool(grid['primary_site_ids']) for grid in grids.values()),
                       'unique_primary_grid_year_records': sum(len(grid['annual']) for grid in grids.values() if grid['primary_site_ids']),
                       'sensitivity_only_grids': sum(not grid['primary_site_ids'] for grid in grids.values()),
                       'input_hashes_unchanged': True, 'independent_run_grouping_matches': True}}
    output.mkdir(parents=True, exist_ok=True)
    script = output / 'research-data.js'
    payload = json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    script.write_text('window.FACTORY_RESEARCH = ' + payload + ';\n', encoding='utf-8', newline='\n')
    downloads = output / 'downloads'
    downloads.mkdir(exist_ok=True)
    summary_fields = ['grid_id', 'latitude', 'longitude', 'land_mask', 'screening_status', 'analysis_role',
                      'primary_site_ids', 'bbox_site_ids', 'complete_summers', 'total_summers',
                      'mean_days_ge_threshold_per_complete_summer', 'max_consecutive_days']
    summary_rows = [{key: '|'.join(grid[key]) if isinstance(grid[key], list) else grid[key]
                     for key in summary_fields} for grid in grids.values()]
    write_csv(downloads / 'research_grid_summary.csv', summary_fields, summary_rows)
    write_csv(downloads / 'research_grid_annual.csv',
              ['grid_id', 'analysis_role', 'threshold_c', 'year', 'expected_days', 'valid_days', 'complete_summer', 'days_ge_threshold', 'max_consecutive_days', 'missing_date_days', 'nonfinite_days'],
              [{'grid_id': grid['grid_id'], 'analysis_role': grid['analysis_role'], 'threshold_c': threshold, **year}
               for grid in grids.values() for year in grid['annual']])
    write_csv(downloads / 'research_grid_daily.csv', ['grid_id', 'analysis_role', 'date', 'daily_max_heat_index_c'],
              [{'grid_id': grid['grid_id'], 'analysis_role': grid['analysis_role'], **day}
               for grid in grids.values() for day in grid['daily']])
    write_csv(downloads / 'research_site_grid_links.csv', ['site_id', 'primary_grid_id', 'bbox_grid_ids'],
              [{**link, 'bbox_grid_ids': '|'.join(link['bbox_grid_ids'])} for link in links])
    files = [script] + sorted(downloads.glob('research_*.csv'))
    package = {key: result[key] for key in ['schema_version', 'generated_at_utc', 'evidence_snapshot_date',
                                          'threshold_c', 'calendar', 'policies', 'input_sha256', 'validation', 'source_metadata']}
    package['output_sha256'] = {str(path.relative_to(output)).replace('\\', '/'): sha256(path) for path in files}
    (downloads / 'research_manifest.json').write_text(json.dumps(package, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['catalog', 'boundaries', 'screening', 'heat-index', 'risk', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = build(args.catalog, args.boundaries, args.screening, args.heat_index, args.risk, args.output)
    print(json.dumps(result['validation']))

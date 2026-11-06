"""Reproducible temporal case analysis for the five desk-verified factory sites.

The frozen representative-point result is preserved. Only distinct valid primary
grids enter temporal analysis; co-located site links never duplicate episodes.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools.esg_site_screen import sha256, summer_metrics


POLICIES = {
    'cohort': 'Exactly the five verified_public_sources sites in the 2026-10-29 catalog; their primary grids only.',
    'analysis_unit': 'Distinct primary ERA5 grid; shared site links do not create additional climate observations or independent episodes.',
    'threshold': 'Daily maximum HI >= 40.6 Celsius, using unrounded decoded daily values.',
    'season': 'June 1 through August 31, UTC+8; 92 unique finite daily values required per summer.',
    'episode': 'A maximal run of consecutive threshold-reaching calendar days within one JJA; one-day episodes are retained. This is not an independent-event assumption or an official heatwave definition.',
    'incomplete_season': 'Keep annual completeness diagnostics, but withhold all threshold episodes and annual threshold/run metrics for incomplete summers.',
    'season_censoring': 'An episode touching June 1 or August 31 is flagged at that edge; duration is observed within JJA and may omit continuation outside the season.',
    'episode_peak_tie': 'Choose the earliest date among equal daily HI peaks.',
    'annual_distribution': 'Min, 25th percentile, median, 75th percentile and max of complete-summer annual metrics; quantiles use numpy method=linear. These are empirical distributions, not confidence intervals.',
    'typical_year': 'Minimize absolute distance from the median annual threshold-day count; choose the earliest year if tied. All tied candidates are retained.',
    'extreme_year': 'Maximize annual threshold-day count, then maximize annual longest run among tied years, then choose the earliest year. Both tie stages are retained.',
    'selection_status': 'Analysis rules specified before this module calculated new case results; existing summary values were already known, so this is not a prospective preregistration.',
    'masked_primary': 'Withhold daily, annual and episode results on non-land primary grids; do not substitute a sensitivity grid.',
}


def validate_protocol(protocol):
    """Fail closed when the executable rules and approved phase-1 plan differ."""
    expected = {
        'protocol_id': 'EHHR-FIVE-SITE-01', 'version': '1.0',
        'evidence_snapshot_date': '2026-10-29', 'threshold_c': 40.6,
        'variable': 'daily_max_heat_index_c',
        'primary_site_ids': [f'factory_{i:03d}' for i in range(1, 6)],
        'period': {'first_year': 2000, 'last_year': 2024, 'months': [6, 7, 8],
                   'expected_days': 92, 'timezone': 'UTC+8'},
        'descriptive_statistics': {'quantiles': [0, .25, .5, .75, 1],
                                   'quantile_method': 'linear', 'trend_or_significance_test': False},
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f'Protocol rule differs from this implementation: {key}')
    for key, value in {'minimum_duration_days': 1, 'cross_season_join': False,
                       'cross_missing_date_or_value_join': False,
                       'peak_tie_break': 'Earliest date'}.items():
        if protocol.get('episode', {}).get(key) != value:
            raise ValueError(f'Protocol episode rule differs: {key}')
    selections = {
        'typical': ['Minimum absolute distance between annual threshold-day count and its complete-summer median', 'Earliest year'],
        'extreme': ['Maximum annual threshold-day count', 'Maximum annual longest-run length', 'Earliest year'],
    }
    for key, value in selections.items():
        if protocol.get('year_selection', {}).get(key) != value:
            raise ValueError(f'Protocol year-selection rule differs: {key}')


def load_research(path):
    """Read the fixed JSON assignment without executing JavaScript."""
    text = Path(path).read_text(encoding='utf-8')
    prefix = 'window.FACTORY_RESEARCH = '
    if not text.startswith(prefix) or not text.rstrip().endswith(';'):
        raise ValueError('Unexpected research data wrapper')
    return json.loads(text[len(prefix):].rstrip()[:-1])


def validate_daily(dates, values, threshold_c):
    dates = pd.DatetimeIndex(dates)
    values = np.asarray(values, dtype=float)
    if not len(dates) or len(dates) != len(values):
        raise ValueError('Dates and values must have the same nonzero length')
    if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError('Dates must be unique and increasing')
    if not dates.equals(dates.normalize()) or not dates.month.isin([6, 7, 8]).all():
        raise ValueError('Only midnight-labelled JJA dates are supported')
    if not np.isfinite(threshold_c):
        raise ValueError('Threshold must be finite')
    return dates, values


def extract_episodes(dates, values, threshold_c=40.6):
    """Extract runs without connecting missing dates, NaNs, years or seasons.

    This low-level function accepts a partial series so gaps can be diagnosed.
    Case analysis calls it only after checking a complete 92-day summer.
    """
    dates, values = validate_daily(dates, values, threshold_c)
    episodes, indices = [], []

    def finish():
        if not indices:
            return
        selected = values[indices]
        first, last = dates[indices[0]], dates[indices[-1]]
        peak_position = int(np.argmax(selected))
        episodes.append({
            'year': int(first.year), 'start_date': first.strftime('%Y-%m-%d'),
            'end_date': last.strftime('%Y-%m-%d'), 'duration_days': len(indices),
            'peak_hi_c': float(selected.max()),
            'peak_date': dates[indices[peak_position]].strftime('%Y-%m-%d'),
            'mean_hi_c': float(selected.mean()),
            'left_season_censored': bool(first.month == 6 and first.day == 1),
            'right_season_censored': bool(last.month == 8 and last.day == 31),
        })
        indices.clear()

    for i, (date, value) in enumerate(zip(dates, values)):
        hit = np.isfinite(value) and value >= threshold_c
        if indices and (not hit or date.year != dates[indices[-1]].year
                        or date - dates[indices[-1]] != pd.Timedelta(days=1)):
            finish()
        if hit:
            indices.append(i)
    finish()
    return episodes


def describe(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    if not np.isfinite(values).all():
        raise ValueError('Distribution input must be finite')
    quartiles = np.quantile(values, [0, .25, .5, .75, 1], method='linear')
    return dict(zip(['min', 'p25', 'median', 'p75', 'max'], map(float, quartiles)))


def choose_years(annual):
    complete = [a for a in annual if a['complete_summer']]
    if not complete:
        return None
    median = float(np.median([a['days_ge_threshold'] for a in complete]))
    distance = min(abs(a['days_ge_threshold'] - median) for a in complete)
    typical = sorted(a['year'] for a in complete
                     if abs(a['days_ge_threshold'] - median) == distance)
    extreme_days = max(a['days_ge_threshold'] for a in complete)
    first_stage = [a for a in complete if a['days_ge_threshold'] == extreme_days]
    longest = max(a['max_consecutive_days'] for a in first_stage)
    second_stage = sorted(a['year'] for a in first_stage
                          if a['max_consecutive_days'] == longest)
    return {
        'typical': {'year': typical[0], 'median_days_ge_threshold': median,
                    'absolute_distance_days': float(distance), 'candidate_years': typical,
                    'tie_breaker': 'earliest_year'},
        'extreme': {'year': second_stage[0], 'days_ge_threshold': extreme_days,
                    'candidate_years_max_days': sorted(a['year'] for a in first_stage),
                    'longest_run_among_max_days': longest,
                    'candidate_years_max_days_and_run': second_stage,
                    'tie_breakers': ['maximum_longest_run', 'earliest_year']},
    }


def analyze_grid(grid, threshold_c=40.6):
    """Check annual reference values and return one deduplicated grid case."""
    result = {key: grid[key] for key in ['grid_id', 'latitude', 'longitude', 'land_mask', 'screening_status']}
    result.update(annual=[], episodes=[], distribution=None, selection=None,
                  complete_summers=None, total_summers=None,
                  mean_days_ge_threshold_per_complete_summer=None)
    if grid['land_mask'] != 1:
        if grid['annual'] or grid['daily']:
            raise ValueError('Masked primary must not contain climate results')
        return result
    dates, values = validate_daily(
        [d['date'] for d in grid['daily']],
        [d['daily_max_heat_index_c'] for d in grid['daily']], threshold_c)
    annual = summer_metrics(dates, values, threshold_c)
    reference = {a['year']: a for a in grid['annual']}
    if len(reference) != len(grid['annual']) or set(reference) != {a['year'] for a in annual}:
        raise ValueError('Annual year coverage differs from the research package')
    episodes = []
    for row in annual:
        year = row['year']
        expected = pd.date_range(f'{year}-06-01', f'{year}-08-31')
        present = dates.year == year
        row['missing_date_days'] = int(len(expected.difference(dates)))
        row['nonfinite_days'] = int((~np.isfinite(values[present])).sum())
        for key, value in row.items():
            if reference[year].get(key) != value:
                raise ValueError(f'Annual reference mismatch: {grid["grid_id"]} {year} {key}')
        row.update(episode_count=None, season_edge_episode_count=None,
                   annual_peak_hi_c=None, annual_peak_date=None)
        if row['complete_summer']:
            year_dates, year_values = dates[present], values[present]
            year_episodes = extract_episodes(year_dates, year_values, threshold_c)
            assert sum(e['duration_days'] for e in year_episodes) == row['days_ge_threshold']
            assert max((e['duration_days'] for e in year_episodes), default=0) == row['max_consecutive_days']
            row['episode_count'] = len(year_episodes)
            row['season_edge_episode_count'] = sum(e['left_season_censored'] or e['right_season_censored'] for e in year_episodes)
            peak_index = int(np.argmax(year_values))
            row['annual_peak_hi_c'] = float(year_values[peak_index])
            row['annual_peak_date'] = year_dates[peak_index].strftime('%Y-%m-%d')
            for number, episode in enumerate(year_episodes, 1):
                episodes.append({'episode_id': f'{grid["grid_id"]}:{year}:{number:02d}',
                                 'grid_id': grid['grid_id'], 'threshold_c': threshold_c,
                                 **episode})
    complete = [a for a in annual if a['complete_summer']]
    result.update(
        annual=annual, episodes=episodes,
        complete_summers=len(complete), total_summers=len(annual),
        distribution={key: describe([a[key] for a in complete])
                      for key in ['days_ge_threshold', 'max_consecutive_days', 'episode_count', 'annual_peak_hi_c']},
        mean_days_ge_threshold_per_complete_summer=(float(np.mean([a['days_ge_threshold'] for a in complete])) if complete else None),
        selection=choose_years(annual))
    return result


def write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def build(research_path, catalog_path, protocol_path, output):
    source_manifest_path = research_path.parent/'downloads/research_manifest.json'
    input_paths = {'research_data': research_path, 'catalog': catalog_path,
                   'research_manifest': source_manifest_path, 'protocol': protocol_path}
    input_hashes = {key: sha256(path) for key, path in input_paths.items()}
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    validate_protocol(protocol)
    source_manifest = json.loads(source_manifest_path.read_text(encoding='utf-8'))
    if source_manifest['output_sha256']['research-data.js'] != input_hashes['research_data']:
        raise ValueError('Research data hash differs from its accepted manifest')
    research = load_research(research_path)
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    if research['input_sha256']['catalog'] != input_hashes['catalog']:
        raise ValueError('Research package and catalog hash differ')
    if research['threshold_c'] != 40.6:
        raise ValueError('This frozen case protocol requires threshold 40.6 Celsius')
    if research['calendar'] != {'first_date': '2000-06-01', 'last_date': '2024-08-31',
                                 'season': 'JJA', 'expected_days_per_summer': 92, 'timezone': 'UTC+8'}:
        raise ValueError('Research calendar differs from the frozen protocol')
    if catalog['reviewed_on'] != '2026-10-29' or research['evidence_snapshot_date'] != catalog['reviewed_on']:
        raise ValueError('This phase uses only the frozen 2026-10-29 evidence cohort')
    sites = [s for s in catalog['sites'] if s['site_verification'] == 'verified_public_sources']
    if len(sites) != 5 or {s['site_id'] for s in sites} != {f'factory_{i:03d}' for i in range(1, 6)}:
        raise ValueError('Unexpected five-site verified cohort')
    link_map = {s['site_id']: s for s in research['site_grid_links']}
    source_grids = {g['grid_id']: g for g in research['grids']}
    primary_ids = sorted({link_map[s['site_id']]['primary_grid_id'] for s in sites})
    grids = []
    for grid_id in primary_ids:
        grid = source_grids[grid_id]
        if grid['analysis_role'] != 'primary':
            raise ValueError('Only accepted primary grids may enter case analysis')
        analyzed = analyze_grid(grid, research['threshold_c'])
        analyzed['site_ids'] = sorted(s['site_id'] for s in sites if link_map[s['site_id']]['primary_grid_id'] == grid_id)
        grids.append(analyzed)
    case_map = {g['grid_id']: g for g in grids}
    site_links = []
    for site in sites:
        grid_id = link_map[site['site_id']]['primary_grid_id']
        case = case_map[grid_id]
        site_links.append({
            'site_id': site['site_id'], 'site_name': site['name'],
            'entity': site['entity'], 'production_address': site['production_address'],
            'site_verification': site['site_verification'], 'primary_grid_id': grid_id,
            'screening_status': case['screening_status'],
            'shared_verified_site_count': len(case['site_ids']),
            'case_role': 'verified_primary' if case['land_mask'] == 1 else 'masked_primary_no_temporal_result',
            'typical_year': case['selection']['typical']['year'] if case['selection'] else None,
            'extreme_year': case['selection']['extreme']['year'] if case['selection'] else None,
        })
    result = {
        'schema_version': 1, 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_id': protocol['protocol_id'], 'protocol_version': protocol['version'],
        'rules_fixed_at_utc': protocol['rules_fixed_at_utc'],
        'evidence_snapshot_date': catalog['reviewed_on'], 'threshold_c': research['threshold_c'],
        'calendar': research['calendar'], 'policies': POLICIES, 'input_sha256': input_hashes,
        'source_input_sha256': research['input_sha256'], 'site_links': site_links, 'grids': grids,
        'validation': {
            'verified_site_count': len(sites), 'distinct_primary_grids': len(grids),
            'distinct_valid_primary_grids': sum(g['land_mask'] == 1 for g in grids),
            'unique_primary_grid_years': sum(len(g['annual']) for g in grids),
            'complete_primary_grid_years': sum(a['complete_summer'] for g in grids for a in g['annual']),
            'deduplicated_episode_count': sum(len(g['episodes']) for g in grids),
            'reference_annual_values_match': True,
            'episode_duration_sums_and_maxima_match_annual': True,
            'research_source_manifest_hash_matches': True,
            'protocol_rules_match_implementation': True,
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output/'case_temporal.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')
    annual_fields = ['grid_id', 'threshold_c', 'year', 'expected_days', 'valid_days', 'missing_date_days', 'nonfinite_days', 'complete_summer', 'days_ge_threshold', 'max_consecutive_days', 'episode_count', 'season_edge_episode_count', 'annual_peak_hi_c', 'annual_peak_date']
    write_csv(output/'annual.csv', annual_fields, [{'grid_id':g['grid_id'], 'threshold_c':research['threshold_c'], **a} for g in grids for a in g['annual']])
    episode_fields = ['episode_id', 'grid_id', 'threshold_c', 'year', 'start_date', 'end_date', 'duration_days', 'peak_hi_c', 'peak_date', 'mean_hi_c', 'left_season_censored', 'right_season_censored']
    write_csv(output/'episodes.csv', episode_fields, [e for g in grids for e in g['episodes']])
    summary_rows = []
    for grid in grids:
        row = {key:grid.get(key) for key in ['grid_id', 'land_mask', 'screening_status', 'complete_summers', 'total_summers', 'mean_days_ge_threshold_per_complete_summer']}
        row['verified_site_ids'] = '|'.join(grid['site_ids'])
        row['episode_count'] = len(grid['episodes']) if grid['land_mask'] == 1 else None
        for metric in ['days_ge_threshold', 'max_consecutive_days', 'episode_count', 'annual_peak_hi_c']:
            for statistic in ['min', 'p25', 'median', 'p75', 'max']:
                row[f'{metric}_{statistic}'] = grid['distribution'][metric][statistic] if grid['distribution'] and grid['distribution'][metric] else None
        summary_rows.append(row)
    write_csv(output/'grid_summary.csv', list(summary_rows[0]), summary_rows)
    selection_rows = []
    for grid in grids:
        if not grid['selection']:
            continue
        for role, selection in grid['selection'].items():
            selection_rows.append({'grid_id':grid['grid_id'], 'selection_role':role,
                                   'selected_year':selection['year'],
                                   'selection_detail_json':json.dumps(selection, ensure_ascii=False, separators=(',', ':'))})
    write_csv(output/'year_selection.csv', ['grid_id', 'selection_role', 'selected_year', 'selection_detail_json'], selection_rows)
    write_csv(output/'site_links.csv', list(site_links[0]), site_links)
    if any(sha256(path) != input_hashes[key] for key, path in input_paths.items()):
        raise RuntimeError('Input changed during case analysis')
    manifest = {key:result[key] for key in ['schema_version', 'generated_at_utc', 'protocol_id', 'protocol_version', 'rules_fixed_at_utc', 'evidence_snapshot_date', 'threshold_c', 'calendar', 'policies', 'input_sha256', 'source_input_sha256', 'validation']}
    manifest['validation']['input_hashes_unchanged'] = True
    manifest['output_sha256'] = {p.name:sha256(p) for p in sorted(output.iterdir()) if p.name in ['case_temporal.json', 'annual.csv', 'episodes.csv', 'grid_summary.csv', 'year_selection.csv', 'site_links.csv']}
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research', type=Path, required=True)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.research, args.catalog, args.protocol, args.output)
    print(json.dumps(result['validation']))


if __name__ == '__main__':
    main()

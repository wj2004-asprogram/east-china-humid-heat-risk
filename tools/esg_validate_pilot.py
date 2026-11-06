"""Offline source/hash audit and independent annual climate cross-check for the pilot."""

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from tools.esg_build_catalog import point_in_ring, read_evidence
from tools.esg_site_screen import sha256


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def on_boundary(point, ring):
    return any(abs(cross(a, b, point)) < 1e-12
               and min(a[0], b[0]) - 1e-12 <= point[0] <= max(a[0], b[0]) + 1e-12
               and min(a[1], b[1]) - 1e-12 <= point[1] <= max(a[1], b[1]) + 1e-12
               for a, b in zip(ring, ring[1:]))


def polygons_overlap(a, b):
    """Positive interior overlap for simple rings; shared edges alone are not duplicates."""
    for first, second in [(a, b), (b, a)]:
        points = first[:-1] + [[(p[0]+q[0])/2, (p[1]+q[1])/2] for p, q in zip(first, first[1:])]
        if any(point_in_ring(point, second) and not on_boundary(point, second) for point in points):
            return True
    if all(on_boundary(p, b) for p in a[:-1]) and all(on_boundary(p, a) for p in b[:-1]):
        return True
    return any(cross(p, q, r) * cross(p, q, s) < -1e-24
               and cross(r, s, p) * cross(r, s, q) < -1e-24
               for p, q in zip(a, a[1:]) for r, s in zip(b, b[1:]))


def audit(catalog_path, boundary_path, evidence, screening, heat_index, risk_path, first_five):
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    boundaries = json.loads(boundary_path.read_text(encoding='utf-8'))
    summary = rows(screening / 'site_summary.csv')
    annual = rows(screening / 'site_annual.csv')
    manifest = json.loads((screening / 'screening_manifest.json').read_text(encoding='utf-8'))
    gate = json.loads((first_five / 'verification_gate.json').read_text(encoding='utf-8'))
    assert sha256(first_five / 'factory_sites.json') == gate['catalog_sha256']
    verified = [s for s in catalog['sites'] if s['site_verification'] == 'verified_public_sources']
    assert len(verified) == 5
    assert all(s['production_address'] and len(s['evidence_ids']) >= 3 for s in verified)
    assert all(s['coordinate_crs'] == 'WGS84' for s in catalog['sites'])
    assert all(not s[k] for s in catalog['sites'] for k in ['employee_count', 'shift_pattern', 'heat_protection'])
    for source_id, source in catalog['sources'].items():
        body, meta = read_evidence(evidence, source_id)
        assert meta['sha256'] == source['sha256']
        if source_id in ['minhang_map', 'taicang_map']:
            assert meta['retrieved_at_utc'] > gate['completed_at_utc']
    assert len({s['osm_id'] for s in catalog['sites']}) == len(catalog['sites'])
    rings = {f['properties']['site_id']: f['geometry']['coordinates'][0] for f in boundaries['features']}
    assert set(rings) == {s['site_id'] for s in catalog['sites']}
    for site in catalog['sites']:
        assert point_in_ring([site['longitude'], site['latitude']], rings[site['site_id']])
    # Flag overlapping bounding boxes for manual duplicate review; do not merge by name.
    bbox_pairs, overlap_pairs = [], []
    for a, b in itertools.combinations(catalog['sites'], 2):
        aw, ass, ae, an = a['bbox']; bw, bs, be, bn = b['bbox']
        if min(ae, be) > max(aw, bw) and min(an, bn) > max(ass, bs):
            bbox_pairs.append([a['site_id'], b['site_id']])
            if polygons_overlap(rings[a['site_id']], rings[b['site_id']]):
                overlap_pairs.append([a['site_id'], b['site_id']])
    assert not overlap_pairs, f'Overlapping site boundaries require duplicate review: {overlap_pairs}'
    climate_hashes = {'heat_index': sha256(heat_index), 'risk': sha256(risk_path)}
    assert all(manifest['input_sha256'][k] == v for k, v in climate_hashes.items())
    assert all(sha256(screening / k) == v for k, v in manifest['outputs_sha256'].items())
    checked = 0
    with xr.open_dataset(heat_index) as heat, xr.open_dataset(risk_path) as risk:
        dates = pd.DatetimeIndex(heat.date.values)
        source = heat.daily_max_heat_index_c.load()
        assert dates.min().year == 2000 and dates.max().year == 2024
        assert np.array_equal(heat.latitude, risk.latitude) and np.array_equal(heat.longitude, risk.longitude)
        for site in summary:
            i = int(np.argmin(np.abs(heat.latitude.values - float(site['latitude']))))
            j = int(np.argmin(np.abs(heat.longitude.values - float(site['longitude']))))
            assert float(site['grid_latitude']) == float(heat.latitude[i])
            assert float(site['grid_longitude']) == float(heat.longitude[j])
            years = [a for a in annual if a['site_id'] == site['site_id']]
            if float(risk.land_mask.values[i, j]) != 1:
                assert site['screening_status'] == 'non_land_grid' and not years
                assert site['mean_days_ge_threshold_per_complete_summer'] == ''
                continue
            assert len(years) == 25
            reference_days, reference_runs = [], []
            values = source.values[:, i, j]
            for record in years:
                year = int(record['year'])
                mask = dates.year == year
                expected = pd.date_range(f'{year}-06-01', f'{year}-08-31')
                assert dates[mask].equals(expected) and np.isfinite(values[mask]).all()
                hits = (values[mask] >= manifest['threshold_c']).tolist()
                days = sum(hits)
                longest = max((sum(1 for _ in group) for hit, group in itertools.groupby(hits) if hit), default=0)
                assert int(record['valid_days']) == 92 and record['complete_summer'] == 'True'
                assert float(record['days_ge_threshold']) == days
                assert float(record['max_consecutive_days']) == longest
                reference_days.append(days); reference_runs.append(longest); checked += 1
            assert float(site['mean_days_ge_threshold_per_complete_summer']) == round(sum(reference_days)/25, 2)
            assert float(site['max_consecutive_days']) == max(reference_runs)
    assert sha256(heat_index) == climate_hashes['heat_index'] and sha256(risk_path) == climate_hashes['risk']
    original = {s['site_id']: s for s in rows(first_five / 'site_summary.csv')}
    for site in summary:
        if site['site_id'] in original:
            for key in ['grid_id', 'screening_status', 'mean_days_ge_threshold_per_complete_summer', 'max_consecutive_days']:
                assert site[key] == original[site['site_id']][key]
    return {'passed': True, 'review_type': 'Codex self-review, not independent review',
            'sites': len(catalog['sites']), 'verified_sites': len(verified), 'candidate_sites': len(catalog['sites']) - len(verified),
            'source_hashes_checked': len(catalog['sources']), 'representative_points_inside_boundaries': len(rings),
            'bbox_overlap_pairs_checked_with_polygons': len(bbox_pairs),
            'polygon_interior_overlap_pairs': overlap_pairs, 'annual_records_cross_checked': checked,
            'climate_status_counts': manifest['status_counts'], 'unique_usable_grids': manifest['unique_land_grids'],
            'first_five_unchanged_after_expansion': True, 'five_site_gate_precedes_expansion_fetches': True,
            'climate_inputs_unchanged': True, 'climate_sha256': climate_hashes,
            'catalog_sha256': sha256(catalog_path), 'boundary_sha256': sha256(boundary_path),
            'screening_manifest_sha256': sha256(screening / 'screening_manifest.json')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['catalog', 'boundaries', 'evidence', 'screening', 'heat-index', 'risk', 'first-five', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    result = audit(a.catalog, a.boundaries, a.evidence, a.screening, a.heat_index, a.risk, a.first_five)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))

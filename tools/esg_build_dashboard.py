"""Package audited site/climate joins as a network-free static pilot."""

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import xarray as xr

from tools.esg_site_screen import sha256


def load_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def build(catalog_path, sites_csv, screening, heat_index, output):
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    manifest = json.loads((screening / 'screening_manifest.json').read_text(encoding='utf-8'))
    if sha256(sites_csv) != manifest['input_sha256']['sites']:
        raise ValueError('Site input no longer matches the screening manifest')
    if sha256(heat_index) != manifest['input_sha256']['heat_index']:
        raise ValueError('Climate input no longer matches the screening manifest')
    for name, digest in manifest['outputs_sha256'].items():
        if sha256(screening / name) != digest:
            raise ValueError(f'Screening output changed: {name}')
    summaries = load_rows(screening / 'site_summary.csv')
    annual_rows = load_rows(screening / 'site_annual.csv')
    by_id = {row['site_id']: row for row in summaries}
    if len(by_id) != len(summaries) or set(by_id) != {s['site_id'] for s in catalog['sites']}:
        raise ValueError('Catalog and screening site IDs must agree exactly')
    with xr.open_dataset(heat_index) as heat:
        latitudes = np.sort(heat.latitude.values)
        longitudes = np.sort(heat.longitude.values)
    number_fields = ['grid_latitude', 'grid_longitude', 'grid_distance_km', 'complete_summers',
                     'total_summers', 'mean_days_ge_threshold_per_complete_summer', 'max_consecutive_days']
    for site in catalog['sites']:
        row = by_id[site['site_id']]
        for key in ['name', 'site_verification', 'coordinate_crs', 'coordinate_method']:
            if row[key] != site[key]:
                raise ValueError(f'Catalog/screening mismatch: {site["site_id"]} {key}')
        for key in ['latitude', 'longitude']:
            if float(row[key]) != site[key]:
                raise ValueError('Coordinate changed after screening')
        site.update({key: float(row[key]) if row[key] else None for key in number_fields})
        site.update({key: row[key] for key in ['grid_id', 'screening_status', 'screening_note']})
        site['annual'] = [{key: (float(value) if key == 'threshold_c' else int(float(value))) if value and key not in ['complete_summer', 'site_id'] else value
                           for key, value in a.items()} for a in annual_rows if a['site_id'] == site['site_id']]
        west, south, east, north = site['bbox']
        lat_indices = sorted(int(np.argmin(abs(latitudes - point))) for point in [south, north])
        lon_indices = sorted(int(np.argmin(abs(longitudes - point))) for point in [west, east])
        site['bbox_grid_ids'] = [f'{lat:.4f},{lon:.4f}' for lat in latitudes[lat_indices[0]:lat_indices[1] + 1]
                                for lon in longitudes[lon_indices[0]:lon_indices[1] + 1]]
    catalog['screening_manifest'] = manifest
    catalog['artifact_input_sha256'] = {'catalog': sha256(catalog_path), 'sites_csv': sha256(sites_csv)}
    output.mkdir(parents=True, exist_ok=True)
    # Escaping < also makes this payload safe if embedded in an HTML script later.
    payload = json.dumps(catalog, ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    (output / 'pilot-data.js').write_text('window.FACTORY_PILOT = ' + payload + ';\n', encoding='utf-8')
    downloads = output / 'downloads'
    downloads.mkdir(exist_ok=True)
    for name in ['site_summary.csv', 'site_annual.csv', 'screening_manifest.json']:
        shutil.copyfile(screening / name, downloads / name)
    shutil.copyfile(catalog_path, downloads / 'factory_sites.json')
    files = ['pilot-data.js'] + [f'downloads/{name}' for name in ['site_summary.csv', 'site_annual.csv', 'screening_manifest.json', 'factory_sites.json']]
    package = {'input_sha256': catalog['artifact_input_sha256'],
               'output_sha256': {name: sha256(output / name) for name in files}}
    (output / 'package_manifest.json').write_text(json.dumps(package, indent=2) + '\n', encoding='utf-8')
    return catalog


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--sites', type=Path, required=True)
    parser.add_argument('--screening', type=Path, required=True)
    parser.add_argument('--heat-index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.catalog, args.sites, args.screening, args.heat_index, args.output)
    print(json.dumps({'sites': len(result['sites']), 'status': 'packaged'}))

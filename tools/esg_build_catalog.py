"""Build a provenance-preserving site catalog from reviewed, cached OSM evidence."""

import argparse
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def read_evidence(directory, source_id):
    meta = json.loads((directory / f'{source_id}.meta.json').read_text(encoding='utf-8'))
    raw = directory / f'{source_id}.raw'
    body = raw.read_bytes()
    if meta['http_status'] != 200 or hashlib.sha256(body).hexdigest() != meta['sha256']:
        raise ValueError(f'Evidence failed integrity validation: {source_id}')
    meta['raw_file'] = raw.name
    meta.setdefault('license_note', 'ODbL 1.0; © OpenStreetMap contributors' if 'osm' in source_id or 'nominatim' in source_id else 'Publicly accessible; no open reuse license identified; local audit copy only')
    return body, meta


def osm_ways(body):
    root = ET.fromstring(body)
    nodes = {n.attrib['id']: [float(n.attrib['lon']), float(n.attrib['lat'])] for n in root.findall('node')}
    ways = {}
    for way in root.findall('way'):
        refs = [n.attrib['ref'] for n in way.findall('nd')]
        if len(refs) < 4 or refs[0] != refs[-1] or any(r not in nodes for r in refs):
            continue
        tags = {t.attrib['k']: t.attrib['v'] for t in way.findall('tag')}
        coordinates = [nodes[r] for r in refs]
        ways[way.attrib['id']] = dict(tags=tags, coordinates=coordinates,
                                     version=way.attrib['version'], timestamp=way.attrib['timestamp'])
    return ways


def point_in_ring(point, ring):
    x, y = point
    inside = False
    for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def representative_point(ring):
    """Prefer bbox midpoint; use widest midlatitude interior span for concave rings."""
    west, south = min(p[0] for p in ring), min(p[1] for p in ring)
    east, north = max(p[0] for p in ring), max(p[1] for p in ring)
    x, y = (west + east) / 2, (south + north) / 2
    method = 'osm_closed_way_bbox_midpoint_inside_polygon_not_surveyed'
    if not point_in_ring([x, y], ring):
        crossings = sorted(x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                           for (x1, y1), (x2, y2) in zip(ring, ring[1:]) if (y1 > y) != (y2 > y))
        spans = list(zip(crossings[::2], crossings[1::2]))
        if not spans or len(crossings) % 2:
            raise ValueError('Cannot derive interior point; manual geometry review required')
        left, right = max(spans, key=lambda span: span[1] - span[0])
        x = (left + right) / 2
        method = 'osm_closed_way_widest_midlatitude_interior_span_midpoint_not_surveyed'
    if not point_in_ring([x, y], ring):
        raise ValueError('Representative point must lie inside the mapped boundary')
    return x, y, [west, south, east, north], method


def build(review_path, evidence_dir, output):
    review = json.loads(review_path.read_text(encoding='utf-8'))
    sources, parsed, sites, features = {}, {}, [], []
    seen_objects = set()
    for item in review['sites']:
        for evidence_id in item['evidence_ids']:
            if evidence_id not in sources:
                body, sources[evidence_id] = read_evidence(evidence_dir, evidence_id)
                if evidence_id == item['osm_evidence_id']:
                    parsed[evidence_id] = osm_ways(body)
        source_id = item['osm_evidence_id']
        if source_id not in parsed:
            parsed[source_id] = osm_ways(read_evidence(evidence_dir, source_id)[0])
        osm_id = item['osm_id']
        if osm_id in seen_objects:
            raise ValueError(f'Duplicate OSM site: {osm_id}')
        seen_objects.add(osm_id)
        way = parsed[source_id][osm_id]
        ring = way['coordinates']
        longitude, latitude, bbox, method = representative_point(ring)
        verification = item.get('site_verification', 'verified_public_sources')
        if verification not in ('verified_public_sources', 'candidate_unverified'):
            raise ValueError('Unknown verification status')
        if verification == 'verified_public_sources' and not item.get('production_address'):
            raise ValueError('Verified sites need a production address')
        row = {**item, 'site_verification': verification,
               'latitude': round(latitude, 7), 'longitude': round(longitude, 7),
               'coordinate_crs': 'WGS84', 'coordinate_method': method,
               'source_url': f'https://www.openstreetmap.org/way/{osm_id}',
               'industry_source': item['evidence_ids'][0],
               'osm_name': way['tags'].get('name', ''), 'osm_version': way['version'],
               'osm_timestamp': way['timestamp'], 'bbox': bbox,
               'employee_count': '', 'shift_pattern': '', 'heat_protection': ''}
        sites.append(row)
        features.append({'type': 'Feature', 'properties': {'site_id': item['site_id'], 'osm_id': osm_id, 'osm_tags': way['tags']},
                         'geometry': {'type': 'Polygon', 'coordinates': [ring]}})
    if len({s['site_id'] for s in sites}) != len(sites):
        raise ValueError('Duplicate site IDs')
    if sum(s['site_verification'] == 'verified_public_sources' for s in sites) < 5:
        raise ValueError('Five verified sites are required before pilot publication')
    output.mkdir(parents=True, exist_ok=True)
    catalog = {'reviewed_on': review['reviewed_on'], 'review_type': review['review_type'],
               'verification_definition': 'Public-source cross-check of entity, production address and mapped factory. Not live operational verification or an on-site survey.',
               'scope': 'Convenience sample within 29.5–32.5 N, 118–122 E; not representative of the Yangtze River Delta manufacturing population.',
               'sources': sources, 'sites': sites}
    (output / 'factory_sites.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'factory_boundaries.geojson').write_text(json.dumps({'type': 'FeatureCollection', 'features': features}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (output / 'factory_sites.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sites[0]))
        writer.writeheader()
        for row in sites:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v for k, v in row.items()})
    return catalog


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    catalog = build(args.review, args.evidence, args.output)
    print(json.dumps({'sites': len(catalog['sites']), 'sources': len(catalog['sources'])}))

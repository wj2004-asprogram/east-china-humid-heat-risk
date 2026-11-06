"""Audit actual factory polygons against nearest-centre climate grid regions.

This reads accepted catalogue/research artifacts; it never changes a primary
grid, fills a masked result, repairs invalid geometry, or estimates exposure.
"""

import argparse
import csv
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyproj
import shapely
import xarray as xr
from pyproj import CRS, Geod, Transformer
from shapely.geometry import Point, box, mapping, shape
from shapely.ops import transform

from tools.esg_site_screen import nearest_cell, sha256


SEGMENT_DEGREES = 0.00005
AREA_RELATIVE_TOLERANCE = 1e-7
REFERENCES = [
    'https://shapely.readthedocs.io/en/2.1.2/reference/shapely.intersection.html',
    'https://shapely.readthedocs.io/en/2.1.2/reference/shapely.is_valid.html',
    'https://shapely.readthedocs.io/en/2.1.2/reference/shapely.point_on_surface.html',
    'https://shapely.readthedocs.io/en/2.1.2/reference/shapely.centroid.html',
    'https://proj.org/en/stable/operations/projections/laea.html',
    'https://pyproj4.github.io/pyproj/stable/api/transformer.html',
    'https://pypi.org/project/shapely/2.1.2/',
]


def load_research(path):
    text = Path(path).read_text(encoding='utf-8')
    prefix = 'window.FACTORY_RESEARCH = '
    if not text.startswith(prefix) or not text.rstrip().endswith(';'):
        raise ValueError('Expected accepted FACTORY_RESEARCH JavaScript data file')
    return json.loads(text[len(prefix):].rstrip()[:-1])


def axis_regions(coordinates):
    """Preserve source indices and bound outer regions to accepted centre domain."""
    coordinates = np.asarray(coordinates, dtype=float)
    if (coordinates.ndim != 1 or len(coordinates) < 2
            or not np.isfinite(coordinates).all()
            or len(np.unique(coordinates)) != len(coordinates)):
        raise ValueError('Grid axis must have at least two distinct finite centres')
    order = np.argsort(coordinates)
    values = coordinates[order]
    edges = np.r_[values[0], (values[1:] + values[:-1]) / 2, values[-1]]
    return {int(index): (float(edges[k]), float(edges[k + 1]))
            for k, index in enumerate(order)}


def require_polygon(geometry):
    if (geometry.is_empty or geometry.geom_type not in ('Polygon', 'MultiPolygon')
            or not geometry.is_valid or not np.isfinite(geometry.bounds).all()
            or geometry.area <= 0):
        raise ValueError('Invalid polygon; no automatic geometry repair: '
                         + str(shapely.is_valid_reason(geometry)))
    west, south, east, north = geometry.bounds
    if not (-180 <= west <= east <= 180 and -90 <= south <= north <= 90):
        raise ValueError('Polygon is outside WGS84 longitude/latitude range')


def local_projection(geometry):
    require_polygon(geometry)
    west, south, east, north = geometry.bounds
    proj_string = (f'+proj=laea +lat_0={(south+north)/2:.12f} '
                   f'+lon_0={(west+east)/2:.12f} +datum=WGS84 +units=m +no_defs')
    crs = CRS.from_proj4(proj_string)
    forward = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
    inverse = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
    return proj_string, forward, inverse


def projected_geometry(geometry, transformer, segment_degrees=SEGMENT_DEGREES / 2):
    # Densification follows the existing straight lon/lat segments; it does not
    # reinterpret the OSM edge as a long ellipsoidal geodesic.
    return transform(transformer.transform, shapely.segmentize(geometry, segment_degrees))


def area_record(geometry, transformer):
    if geometry.is_empty or geometry.area == 0:
        return {'area_m2': 0.0, 'coarse_area_m2': 0.0,
                'refinement_absolute_delta_m2': 0.0,
                'geodesic_check_area_m2': 0.0, 'geodesic_check_absolute_delta_m2': 0.0}
    coarse = projected_geometry(geometry, transformer, SEGMENT_DEGREES).area
    refined = projected_geometry(geometry, transformer).area
    if not np.isfinite([coarse, refined]).all() or refined <= 0:
        raise ValueError('Invalid projected polygon area')
    if abs(coarse-refined) > max(1e-4, refined*AREA_RELATIVE_TOLERANCE):
        raise ValueError('Area fails densification refinement tolerance')
    densified = shapely.orient_polygons(shapely.segmentize(geometry, SEGMENT_DEGREES/2))
    geodesic_area, _ = Geod(ellps='WGS84').geometry_area_perimeter(densified)
    area_difference = abs(abs(float(geodesic_area))-refined)
    if area_difference > max(1e-3,refined*1e-6):
        raise ValueError('Projected and geodesic area checks disagree')
    return {'area_m2': float(refined), 'coarse_area_m2': float(coarse),
            'refinement_absolute_delta_m2': float(abs(coarse-refined)),
            'geodesic_check_area_m2': abs(float(geodesic_area)),
            'geodesic_check_absolute_delta_m2': area_difference}


def polygon_grid_intersections(geometry, latitudes, longitudes):
    """Exact planar lon/lat overlay, then locally equal-area measurement."""
    require_polygon(geometry)
    west, south, east, north = geometry.bounds
    lat_regions, lon_regions = axis_regions(latitudes), axis_regions(longitudes)
    if (west < min(longitudes) or east > max(longitudes)
            or south < min(latitudes) or north > max(latitudes)):
        raise ValueError('Polygon extends beyond accepted climate centre domain')
    proj_string, forward, inverse = local_projection(geometry)
    total = area_record(geometry, forward)
    rows = []
    for i, (bottom, top) in lat_regions.items():
        if top < south or bottom > north:
            continue
        for j, (left, right) in lon_regions.items():
            if right < west or left > east:
                continue
            region = box(left, bottom, right, top)
            clipped = shapely.intersection(geometry, region, grid_size=0)
            dimensions = shapely.get_dimensions(clipped)
            relation = ('disjoint' if clipped.is_empty else
                        'positive_area' if dimensions == 2 and clipped.area > 0
                        else 'boundary_touch')
            area = area_record(clipped, forward)
            rows.append({'grid_id': f'{latitudes[i]:.4f},{longitudes[j]:.4f}',
                         'grid_latitude': float(latitudes[i]),
                         'grid_longitude': float(longitudes[j]),
                         'latitude_index': i, 'longitude_index': j,
                         'region_bounds_wsen': [left, bottom, right, top],
                         'relation': relation, **area,
                         'factory_area_fraction': area['area_m2']/total['area_m2'],
                         'geometry': clipped})
    rows.sort(key=lambda row: row['grid_id'])
    sum_area = sum(row['area_m2'] for row in rows)
    residual = sum_area-total['area_m2']
    if abs(residual) > max(1e-4, total['area_m2']*AREA_RELATIVE_TOLERANCE):
        raise ValueError('Polygon partition fails area conservation tolerance')
    summary = {**total, 'projection': proj_string,
               'intersection_area_sum_m2': sum_area,
               'area_conservation_residual_m2': residual,
               'area_fraction_sum': sum(row['factory_area_fraction'] for row in rows)}
    return rows, summary, forward, inverse


def metric_comparison(primary, scenario):
    """Never replace a rejected primary or manufacture its numerical difference."""
    both_valid = (primary['screening_status'] == 'ok'
                  and scenario['screening_status'] == 'ok')
    return {
        'screening_status': scenario['screening_status'],
        'land_mask': scenario['land_mask'],
        'mean_days_ge_threshold_per_complete_summer': scenario['mean_days_ge_threshold_per_complete_summer'],
        'max_consecutive_days': scenario['max_consecutive_days'],
        'mean_days_difference_from_primary': (
            scenario['mean_days_ge_threshold_per_complete_summer']
            - primary['mean_days_ge_threshold_per_complete_summer'] if both_valid else None),
        'difference_status': 'computed' if both_valid else 'unavailable_primary_or_scenario',
    }


def representative_scenarios(geometry, latitude, longitude, latitudes, longitudes,
                             intersections, grids, forward, inverse, primary_id):
    projected = projected_geometry(geometry, forward)
    centroid = transform(inverse.transform, projected.centroid)
    surface = transform(inverse.transform, projected.representative_point())
    locations = [('accepted_primary', Point(longitude, latitude)),
                 ('laea_area_centroid', centroid), ('laea_point_on_surface', surface)]
    rows = []
    for label, point in locations:
        i,j,distance = nearest_cell(latitudes,longitudes,point.y,point.x)
        grid_id = f'{latitudes[i]:.4f},{longitudes[j]:.4f}'
        if grid_id not in grids:
            raise ValueError('Scenario grid absent from accepted research data')
        rows.append({'scenario':label,'scenario_type':'representative_point',
                     'longitude':float(point.x),'latitude':float(point.y),
                     'inside_boundary':bool(geometry.contains(point)),
                     'covered_by_boundary':bool(geometry.covers(point)),
                     'distance_to_selected_grid_km':distance,
                     'grid_id':grid_id,'changes_primary_grid':grid_id!=primary_id,
                     **metric_comparison(grids[primary_id],grids[grid_id])})
    if not rows[0]['covered_by_boundary'] or not rows[2]['covered_by_boundary']:
        raise ValueError('Accepted or point-on-surface representative outside source polygon')
    positive = [row for row in intersections if row['relation']=='positive_area']
    # Equal area ties use ascending latitude, then longitude through grid_id.
    largest = sorted(positive,key=lambda r:(-r['area_m2'],r['grid_id']))[0]
    grid_id=largest['grid_id']
    rows.append({'scenario':'largest_polygon_overlap_grid','scenario_type':'grid_coverage_scenario',
                 'longitude':None,'latitude':None,'inside_boundary':None,
                 'covered_by_boundary':None,'distance_to_selected_grid_km':None,
                 'grid_id':grid_id,'changes_primary_grid':grid_id!=primary_id,
                 **metric_comparison(grids[primary_id],grids[grid_id])})
    return rows


def write_csv(path, rows):
    fields=list(rows[0])
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n')
        writer.writeheader()
        writer.writerows({key:'|'.join(map(str,value)) if isinstance(value,list) else value
                         for key,value in row.items()} for row in rows)


def build(catalog_path,boundary_path,research_path,research_manifest_path,risk_path,protocol_path,output,wheel_path=None):
    inputs={'catalog':catalog_path,'boundaries':boundary_path,'research_data':research_path,
            'research_manifest':research_manifest_path,'risk':risk_path,'protocol':protocol_path}
    hashes={key:sha256(path) for key,path in inputs.items()}
    catalog=json.loads(catalog_path.read_text(encoding='utf-8'))
    boundaries=json.loads(boundary_path.read_text(encoding='utf-8'))
    research=load_research(research_path)
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    accepted=json.loads(research_manifest_path.read_text(encoding='utf-8'))
    for key in ['catalog','boundaries','risk']:
        if hashes[key]!=accepted['input_sha256'][key]:
            raise ValueError('Input hash differs from accepted research package: '+key)
    if hashes['research_data']!=accepted['output_sha256']['research-data.js']:
        raise ValueError('Research JavaScript hash differs from its manifest')
    sites=[s for s in catalog['sites'] if s['site_verification']=='verified_public_sources']
    if (len(sites)!=5 or sorted(s['site_id'] for s in sites)!=sorted(protocol['primary_site_ids'])
            or protocol['threshold_c']!=research['threshold_c']):
        raise ValueError('This protocol is restricted to the fixed five-site verified cohort')
    features={f['properties']['site_id']:f for f in boundaries['features']}
    links={s['site_id']:s for s in research['site_grid_links']}
    grids={g['grid_id']:g for g in research['grids']}
    with xr.open_dataset(risk_path) as risk:
        latitudes=risk.latitude.values.copy()
        longitudes=risk.longitude.values.copy()
        masks=risk.land_mask.transpose('latitude','longitude').values.copy()
    if not all(np.allclose(np.diff(np.sort(axis)),0.25,rtol=0,atol=1e-12)
               for axis in [latitudes,longitudes]):
        raise ValueError('This protocol expects the accepted 0.25 degree grid')
    summaries,overlaps,scenarios,geometries=[],[],[],[]
    for site in sites:
        site_id=site['site_id']
        geometry=shape(features[site_id]['geometry'])
        rows,area,forward,inverse=polygon_grid_intersections(geometry,latitudes,longitudes)
        positive_ids=[r['grid_id'] for r in rows if r['relation']=='positive_area']
        bbox_ids=sorted(links[site_id]['bbox_grid_ids'])
        if sorted(r['grid_id'] for r in rows)!=bbox_ids or not set(positive_ids)<=set(bbox_ids):
            raise ValueError('Exact-polygon candidates disagree with accepted bbox candidate set')
        primary_id=links[site_id]['primary_grid_id']
        point_scenarios=representative_scenarios(
            geometry,site['latitude'],site['longitude'],latitudes,longitudes,
            rows,grids,forward,inverse,primary_id)
        valid_fraction=0.0
        for row in rows:
            i,j=row['latitude_index'],row['longitude_index']
            mask=float(masks[i,j])
            if mask not in (0,1) or mask!=grids[row['grid_id']]['land_mask']:
                raise ValueError('Land mask differs from accepted research package')
            valid_fraction+=row['factory_area_fraction'] if mask==1 else 0
            clipped=row.pop('geometry')
            row.pop('latitude_index');row.pop('longitude_index')
            row.update(land_mask=mask,is_primary_grid=row['grid_id']==primary_id,
                       research_analysis_role=grids[row['grid_id']]['analysis_role'])
            record={'site_id':site_id,'name':site['name'],**row}
            overlaps.append(record)
            if not clipped.is_empty:
                geometries.append({'type':'Feature','properties':record,'geometry':mapping(clipped)})
        masked_fraction=sum(r['factory_area_fraction'] for r in rows if r['land_mask']==0)
        summaries.append({'site_id':site_id,'name':site['name'],'primary_grid_id':primary_id,
                          'osm_id':site['osm_id'],'osm_timestamp':site['osm_timestamp'],
                          'source_url':site['source_url'],'geometry_valid':True,**area,
                          'bbox_candidate_grid_ids':bbox_ids,
                          'positive_area_grid_ids':positive_ids,
                          'boundary_touch_grid_ids':[r['grid_id'] for r in rows if r['relation']=='boundary_touch'],
                          'bbox_without_positive_area_grid_ids':[r['grid_id'] for r in rows if r['relation']!='positive_area'],
                          'land_mask_accepted_area_fraction':valid_fraction,
                          'land_mask_rejected_area_fraction':masked_fraction,
                          'primary_grid_area_fraction':next(r['factory_area_fraction'] for r in rows if r['grid_id']==primary_id),
                          'scenario_count':len(point_scenarios),
                          'changed_grid_scenario_count':sum(r['changes_primary_grid'] for r in point_scenarios),
                          'outside_centroid':not point_scenarios[1]['covered_by_boundary']})
        scenarios.extend({'site_id':site_id,'name':site['name'],'primary_grid_id':primary_id,**r}
                         for r in point_scenarios)
    if any(sha256(path)!=hashes[key] for key,path in inputs.items()):
        raise RuntimeError('An input changed during spatial audit')
    output.mkdir(parents=True,exist_ok=True)
    payload={'schema_version':1,'protocol_id':protocol['protocol_id'],
             'evidence_snapshot_date':catalog['reviewed_on'],
             'analysis_scope':'fixed_five_verified_sites','site_audits':summaries,
             'grid_intersections':overlaps,'representative_scenarios':scenarios}
    (output/'spatial_audit.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    (output/'factory_grid_intersections.geojson').write_text(json.dumps({'type':'FeatureCollection','features':geometries},ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    for name,rows in [('site_spatial_audit.csv',summaries),('polygon_grid_intersections.csv',overlaps),('representative_scenarios.csv',scenarios)]:
        write_csv(output/name,rows)
    files=['spatial_audit.json','factory_grid_intersections.geojson','site_spatial_audit.csv',
           'polygon_grid_intersections.csv','representative_scenarios.csv']
    manifest={'schema_version':1,'generated_at_utc':datetime.now(timezone.utc).isoformat(),
              'evidence_snapshot_date':catalog['reviewed_on'],'source_code_sha256':sha256(Path(__file__)),
              'input_sha256':hashes,'output_sha256':{name:sha256(output/name) for name in files},
              'software':{'python':platform.python_version(),'shapely':shapely.__version__,
                          'GEOS':shapely.geos_version_string,'pyproj':pyproj.__version__,
                          'PROJ':pyproj.proj_version_str,'numpy':np.__version__},
              'methods':{'overlay':'Shapely/GEOS double-precision intersection of original straight lon/lat polygon segments and nearest-centre axis partition rectangles; no snapping or repair.',
                         'grid_ties':'Point matches retain original axis order through numpy.argmin. Closed polygon partitions may share zero-area boundaries.',
                         'area':'Each factory uses a WGS84 local Lambert Azimuthal Equal Area projection centred on its bbox midpoint. Project densified original lon/lat segments; area is in square metres.',
                         'densification_coarse_degrees':SEGMENT_DEGREES,
                         'densification_final_degrees':SEGMENT_DEGREES/2,
                         'area_conservation_relative_tolerance':AREA_RELATIVE_TOLERANCE,
                         'centroid':'Area centroid of locally projected densified polygon; may be outside a concave polygon and is then explicitly marked.',
                         'point_on_surface':'Shapely representative_point of locally projected polygon, inverse-transformed and checked against original polygon.',
                         'largest_overlap':'Largest positive polygon intersection area; exact ties ordered by grid latitude/longitude. Grid scenario, not a factory point.',
                         'mask':'Rejected area fraction is retained, never renormalized onto valid land cells; no area-weighted factory HI or exposure is computed.',
                         'primary_result':'Accepted primary point, grid and missing metrics unchanged; comparisons against a masked primary remain null.',
                         'area_interpretation':'Fractions describe mapped polygon geometry within nearest-centre regions, not worker distribution, land-surface fraction, site exposure, or positional confidence.'},
              'validation':{'verified_sites':len(sites),'intersection_rows':len(overlaps),
                            'representative_scenarios':len(scenarios),'inputs_unchanged':True,
                            'positive_grids_subset_of_bbox_candidates':True,
                            'maximum_absolute_area_conservation_residual_m2':max(abs(s['area_conservation_residual_m2']) for s in summaries),
                            'maximum_absolute_area_fraction_residual':max(abs(s['area_fraction_sum']-1) for s in summaries)},
              'reference_urls':REFERENCES}
    if wheel_path:
        manifest['isolated_dependency_wheel']={'filename':wheel_path.name,'sha256':sha256(wheel_path),
                                               'source':'https://pypi.org/project/shapely/2.1.2/#files',
                                               'installation':'pip --no-deps --target in external task artifact directory; main environment unchanged'}
    (output/'spatial_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    return payload,manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['catalog','boundaries','research','research-manifest','risk','protocol','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--dependency-wheel',type=Path)
    args=parser.parse_args()
    _,manifest=build(args.catalog,args.boundaries,args.research,args.research_manifest,
                     args.risk,args.protocol,args.output,args.dependency_wheel)
    print(json.dumps(manifest['validation']))

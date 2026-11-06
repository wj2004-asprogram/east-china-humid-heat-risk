"""Protect spatial conclusions against false overlap and primary substitution."""

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from tools.esg_case_spatial import (local_projection, metric_comparison,
                                    polygon_grid_intersections, require_polygon,
                                    representative_scenarios)


def grid_record(mask=1, mean=20.0):
    return {'screening_status':'ok' if mask else 'non_land_grid','land_mask':mask,
            'mean_days_ge_threshold_per_complete_summer':mean if mask else None,
            'max_consecutive_days':7 if mask else None}


def test_rectangle_crosses_two_grid_regions_with_area_conservation():
    # Longitude boundary is exactly 120.0; latitude stays inside one region.
    polygon=box(119.99,29.99,120.01,30.01)
    rows,audit,_,_=polygon_grid_intersections(polygon,[29.75,30,30.25],[119.875,120.125])
    assert len(rows)==2
    assert all(r['relation']=='positive_area' for r in rows)
    assert [r['factory_area_fraction'] for r in rows]==pytest.approx([.5,.5],abs=1e-8)
    assert audit['area_fraction_sum']==pytest.approx(1,abs=1e-10)
    assert audit['geodesic_check_absolute_delta_m2'] < audit['area_m2']*1e-6


def test_boundary_contact_is_retained_without_positive_area():
    polygon=box(119.99,29.99,120,30.01)
    rows,audit,_,_=polygon_grid_intersections(polygon,[29.75,30,30.25],[119.875,120.125])
    assert [r['relation'] for r in rows]==['positive_area','boundary_touch']
    assert rows[1]['area_m2']==0
    assert rows[1]['factory_area_fraction']==0
    assert audit['area_fraction_sum']==pytest.approx(1,abs=1e-10)


def test_concave_polygon_bbox_candidate_can_be_disjoint():
    polygon=Polygon([(119.98,29.98),(120.02,29.98),(120.02,29.99),
                     (119.99,29.99),(119.99,30.02),(119.98,30.02)])
    rows,_,_,_=polygon_grid_intersections(polygon,[29.875,30.125],[119.875,120.125])
    northeast=next(r for r in rows if r['grid_id']=='30.1250,120.1250')
    assert northeast['relation']=='disjoint'
    assert northeast['area_m2']==0


def test_concave_centroid_is_flagged_outside_and_surface_stays_inside():
    polygon=Polygon([(120,30),(120.04,30),(120.04,30.01),
                     (120.01,30.01),(120.01,30.04),(120,30.04)])
    latitudes=[29.75,30,30.25];longitudes=[119.75,120,120.25]
    rows,_,forward,inverse=polygon_grid_intersections(polygon,latitudes,longitudes)
    scenarios=representative_scenarios(polygon,30.02,120.005,latitudes,longitudes,
                                        rows,{'30.0000,120.0000':grid_record()},
                                        forward,inverse,'30.0000,120.0000')
    assert scenarios[1]['scenario']=='laea_area_centroid'
    assert scenarios[1]['inside_boundary'] is False
    assert scenarios[1]['covered_by_boundary'] is False
    assert scenarios[2]['inside_boundary'] is True


def test_masked_primary_never_gets_numeric_delta_from_valid_scenario():
    result=metric_comparison(grid_record(0),grid_record(1,10.52))
    assert result['mean_days_ge_threshold_per_complete_summer']==10.52
    assert result['mean_days_difference_from_primary'] is None
    assert result['difference_status']=='unavailable_primary_or_scenario'


def test_two_valid_scenarios_can_have_a_signed_delta():
    assert metric_comparison(grid_record(1,25),grid_record(1,10.52))[
        'mean_days_difference_from_primary']==pytest.approx(-14.48)


def test_invalid_self_intersection_fails_without_repair():
    bowtie=Polygon([(120,30),(120.02,30.02),(120.02,30),(120,30.02)])
    with pytest.raises(ValueError,match='no automatic geometry repair'):
        require_polygon(bowtie)


def test_polygon_outside_climate_domain_is_not_clipped_into_a_valid_site():
    with pytest.raises(ValueError,match='beyond accepted climate centre domain'):
        polygon_grid_intersections(box(119.7,30,119.8,30.01),[29.75,30,30.25],[119.75,120,120.25])


def test_polygon_hole_preserves_net_area_and_fraction_closure():
    polygon=Polygon([(119.99,29.99),(120.01,29.99),(120.01,30.01),(119.99,30.01)],
                    holes=[[(119.995,29.995),(119.995,30.005),(120.005,30.005),(120.005,29.995)]])
    rows,audit,_,_=polygon_grid_intersections(polygon,[29.75,30,30.25],[119.875,120.125])
    _,outer,_,_=polygon_grid_intersections(box(119.99,29.99,120.01,30.01),[29.75,30,30.25],[119.875,120.125])
    assert audit['area_m2']/outer['area_m2']==pytest.approx(.75,abs=1e-7)
    assert sum(r['factory_area_fraction'] for r in rows)==pytest.approx(1,abs=1e-9)


def test_source_axis_order_preserves_nearest_grid_tie_policy():
    polygon=box(119.99,29.99,120.01,30.01)
    latitudes=np.array([30.25,30,29.75]);longitudes=np.array([120.125,119.875])
    rows,_,forward,inverse=polygon_grid_intersections(polygon,latitudes,longitudes)
    grids={r['grid_id']:grid_record() for r in rows}
    scenarios=representative_scenarios(polygon,30,120,latitudes,longitudes,rows,grids,
                                       forward,inverse,'30.0000,120.1250')
    assert scenarios[0]['grid_id']=='30.0000,120.1250'

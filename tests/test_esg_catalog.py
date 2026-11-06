"""Protect against concave-boundary mislocation and loss of evidence integrity."""

import hashlib
import json

import pytest

from tools.esg_build_catalog import osm_ways, point_in_ring, read_evidence, representative_point
from tools.esg_validate_pilot import polygons_overlap


def test_concave_factory_does_not_use_empty_bbox_center():
    # L-shaped factory: (2, 2) is outside; the widest interior span at y=2 is x=0..1.
    ring = [[0, 0], [4, 0], [4, 1], [1, 1], [1, 4], [0, 4], [0, 0]]
    x, y, bbox, method = representative_point(ring)
    assert (x, y) == (0.5, 2)
    assert point_in_ring([x, y], ring)
    assert not point_in_ring([2, 2], ring)
    assert bbox == [0, 0, 4, 4]
    assert 'interior_span' in method


def test_reversed_ring_still_produces_same_point():
    ring = [[120, 30], [121, 30], [121, 31], [120, 31], [120, 30]]
    assert representative_point(ring) == representative_point(ring[::-1])
    assert representative_point(ring)[:2] == (120.5, 30.5)


def test_incomplete_osm_geometry_is_not_silently_truncated():
    body = b'''<osm><node id="1" lat="30" lon="120"/><node id="2" lat="31" lon="120"/>
    <way id="1" version="1" timestamp="2026-01-01T00:00:00Z"><nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="1"/></way></osm>'''
    assert osm_ways(body) == {}


def test_changed_evidence_fails_closed(tmp_path):
    body = b'original evidence'
    (tmp_path / 'test.raw').write_bytes(body)
    meta = {'http_status': 200, 'sha256': hashlib.sha256(body).hexdigest()}
    (tmp_path / 'test.meta.json').write_text(json.dumps(meta), encoding='utf-8')
    assert read_evidence(tmp_path, 'test')[0] == body
    (tmp_path / 'test.raw').write_bytes(b'changed evidence')
    with pytest.raises(ValueError, match='integrity'):
        read_evidence(tmp_path, 'test')


@pytest.mark.parametrize('other, overlaps', [
    ([[1,0],[2,0],[2,1],[1,1],[1,0]], False),  # Shared edge only.
    ([[.5,0],[1.5,0],[1.5,1],[.5,1],[.5,0]], True),  # Aligned partial overlap.
    ([[.2,.2],[.8,.2],[.8,.8],[.2,.8],[.2,.2]], True),  # Contained building.
    ([[0,0],[1,0],[1,1],[0,1],[0,0]], True),  # Duplicate geometry.
])
def test_site_duplicate_geometry_cases(other, overlaps):
    square = [[0,0],[1,0],[1,1],[0,1],[0,0]]
    assert polygons_overlap(square, other) == overlaps

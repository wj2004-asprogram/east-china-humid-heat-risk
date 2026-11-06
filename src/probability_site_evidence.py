"""Translate frozen hindcasts into deduplicated, descriptive site-grid evidence."""
from collections import defaultdict

import numpy as np
import pandas as pd

from src.probability_metrics import reliability_counts
from src.probability_phase4_evaluation import (
    alert_metrics, risk_masks, score_probability, summarize_annual,
)


def map_primary_sites(catalog, links, registry, spatial, scenarios, site_ids):
    """Preserve archived primary assignments, including rejected cells."""
    sites = {s['site_id']: s for s in catalog['sites']}
    output = []
    for site_id in site_ids:
        link = links.loc[links.site_id == site_id]
        audit = spatial.loc[spatial.site_id == site_id]
        point = scenarios.loc[(scenarios.site_id == site_id) &
                              (scenarios.scenario == 'accepted_primary')]
        if len(link) != 1 or len(audit) != 1 or len(point) != 1:
            raise ValueError(f'Nonunique archived primary evidence: {site_id}')
        link, audit, point = link.iloc[0], audit.iloc[0], point.iloc[0]
        lat, lon = map(float, link.primary_grid_id.split(','))
        found = registry.loc[(registry.latitude == lat) & (registry.longitude == lon)]
        accepted = link.screening_status == 'ok'
        if (accepted and len(found) != 1) or (not accepted and len(found)):
            raise ValueError(f'Land registry conflicts with archived status: {site_id}')
        role = str(found.iloc[0].spatial_role) if accepted else None
        s = sites[site_id]
        output.append(dict(site_id=site_id, site_name=s['name'],
                           primary_grid_id=link.primary_grid_id,
                           grid_latitude=lat, grid_longitude=lon,
                           site_latitude=s['latitude'], site_longitude=s['longitude'],
                           grid_index=int(found.index[0]) if accepted else None,
                           spatial_role=role, screening_status=link.screening_status,
                           temporal_available=accepted,
                           strict_H_available=accepted and role == 'holdout',
                           shared_site_count=int(link.shared_verified_site_count),
                           representative_distance_km=float(point.distance_to_selected_grid_km),
                           primary_partition_area_fraction=float(audit.primary_grid_area_fraction),
                           accepted_partition_area_fraction=float(audit.land_mask_accepted_area_fraction)))
    return pd.DataFrame(output)


def grids_for_scope(mapping, scope):
    if scope not in ('temporal', 'strict'):
        raise ValueError('Unknown scope')
    flag = 'temporal_available' if scope == 'temporal' else 'strict_H_available'
    return mapping.loc[mapping[flag]].drop_duplicates('primary_grid_id').copy()


def verify_subset(frame, historical_daily, years, expected_days, threshold):
    """Check labels, common samples, dates and frozen alert decisions."""
    keys = ['scope', 'lag', 'model_id', 'year', 'primary_grid_id', 'target_date_bjt']
    if frame.duplicated(keys).any():
        raise ValueError('Duplicated site-grid prediction observations')
    daily = {(r['grid_id'], r['date']): r['daily_max_heat_index_c']
             for r in historical_daily}
    references = {}
    for (scope, lag, model, grid), part in frame.groupby(['scope', 'lag', 'model_id', 'primary_grid_id']):
        if sorted(part.year.unique().tolist()) != list(years):
            raise ValueError('Incomplete final years')
        if not (part.groupby('year').size() == expected_days).all():
            raise ValueError('Incomplete prediction summer')
        ordered = part.sort_values('target_date_bjt')
        samples = list(zip(ordered.target_date_bjt, ordered.y, ordered.evaluation_previous_y))
        ref_key = (scope, grid)
        if ref_key in references and samples != references[ref_key]:
            raise ValueError('Models or lags use different labels or samples')
        references[ref_key] = samples
    physical = frame[['primary_grid_id', 'target_date_bjt', 'y', 'evaluation_previous_y']].drop_duplicates()
    if physical.duplicated(['primary_grid_id', 'target_date_bjt']).any():
        raise ValueError('Different scopes disagree on observed labels')
    for row in physical.itertuples(index=False):
        yesterday = (pd.Timestamp(row.target_date_bjt)-pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        if row.y != int(daily[(row.primary_grid_id, row.target_date_bjt)] >= threshold):
            raise ValueError('Target label differs from archived unrounded HI')
        if row.evaluation_previous_y != int(daily[(row.primary_grid_id, yesterday)] >= threshold):
            raise ValueError('Previous-day evaluation label differs from archived HI')
    for p in ('p_raw', 'p_calibrated'):
        if not np.isfinite(frame[p]).all() or not frame[p].between(0, 1).all():
            raise ValueError('Invalid probabilities')
    for flag, threshold_column in [('alert_primary', 'primary_threshold'), ('alert_budget', 'budget_threshold')]:
        if not np.array_equal(frame[flag], (frame.p_calibrated >= frame[threshold_column]).astype(int)):
            raise ValueError('Frozen alert decisions changed')
    return dict(prediction_rows=len(frame), distinct_physical_grid_days=len(physical),
                unrounded_legacy_labels_match=True, common_model_lag_samples=True,
                frozen_calibrated_alerts_match=True)


def describe_predictions(frame, risk_sets):
    """Use the phase-four metrics; no refitting or site-level selection."""
    annual, reliability, alerts = [], [], []
    grouped = defaultdict(list)
    columns = ['scope', 'lag', 'model_id', 'primary_grid_id', 'year']
    for key, part in frame.groupby(columns, sort=True):
        base = dict(zip(columns, key))
        masks = risk_masks(part.y.to_numpy(), part.evaluation_previous_y.to_numpy())
        for risk in risk_sets:
            subset = part.loc[masks[risk]]
            y, lat = subset.y.to_numpy(), subset.grid_latitude.to_numpy()
            for label, col in [('raw', 'p_raw'), ('calibrated', 'p_calibrated')]:
                meta = dict(base, risk_set=risk, probability=label)
                p = subset[col].to_numpy()
                row = dict(meta, **score_probability(y, p, lat))
                annual.append(row)
                groupkey = tuple(meta[k] for k in columns[:-1]) + (risk, label)
                grouped[groupkey].append(row)
                reliability.extend(dict(meta, **r) for r in reliability_counts(y, p))
            for rule, col in [('primary', 'primary_threshold'), ('budget', 'budget_threshold')]:
                threshold = float(part[col].iloc[0])
                if part[col].nunique() != 1:
                    raise ValueError('Threshold changes within a frozen model year')
                alerts.append(dict(base, risk_set=risk, rule=rule, probability='calibrated',
                                   **alert_metrics(y, subset.p_calibrated.to_numpy(), threshold)))
    aggregate = [dict(zip(columns[:-1] + ['risk_set', 'probability'], key),
                      **summarize_annual(rows)) for key, rows in grouped.items()]
    return {name: pd.DataFrame(rows) for name, rows in
            [('annual_scores', annual), ('aggregate_scores', aggregate),
             ('reliability', reliability), ('annual_alerts', alerts)]}

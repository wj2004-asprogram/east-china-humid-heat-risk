"""Extract site evidence from immutable phase-four predictions; never train models."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.probability_phase4_data import check_gate
from src.probability_site_evidence import (
    describe_predictions, grids_for_scope, map_primary_sites, verify_subset,
)
from tools.esg_case_temporal import load_research


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def run(config_path):
    config = read_json(config_path)
    root, out = Path(config['source_phase4_root']), Path(config['output_root'])
    out.mkdir(parents=True, exist_ok=True)
    inputs = {}

    def record(path, expected=None):
        path = Path(path)
        value = sha(path)
        if expected is not None and value != expected:
            raise ValueError(f'Input hash mismatch: {path}')
        inputs[path.as_posix()] = value
        return path

    record(config_path)
    gates = check_gate(root, 'final_test')
    state = read_json(record(root/'evaluation/execution_state.json'))
    if state['status'] != 'completed' or state['gates'] != gates:
        raise ValueError('Phase four must be completed with unchanged gates')
    for key, name in [('source_temporal_root', 'manifest.json'), ('source_spatial_root', 'spatial_manifest.json')]:
        source = Path(config[key])
        manifest = read_json(record(source/name))
        for filename, expected in manifest['output_sha256'].items():
            record(source/filename, expected)
        record(config['site_catalog'], manifest['input_sha256']['catalog'])
        record(config['source_research_data'], manifest['input_sha256']['research_data'])
        record(config['source_site_protocol'], manifest['input_sha256']['protocol'])
    catalog = read_json(config['site_catalog'])
    if catalog['reviewed_on'] != config['evidence_snapshot_date']:
        raise ValueError('Evidence snapshot changed')
    registry = pd.read_csv(record(config['grid_registry']))
    temporal, spatial = Path(config['source_temporal_root']), Path(config['source_spatial_root'])
    links = pd.read_csv(temporal/'site_links.csv')
    mapping = map_primary_sites(catalog, links, registry,
                                pd.read_csv(spatial/'site_spatial_audit.csv'),
                                pd.read_csv(spatial/'representative_scenarios.csv'),
                                config['primary_site_ids'])
    mapping.to_csv(out/'site_mapping.csv', index=False, lineterminator='\n')
    accepted_ids = grids_for_scope(mapping, 'temporal').primary_grid_id.tolist()
    research = load_research(Path(config['source_research_data']))
    history = [dict(grid_id=g['grid_id'], **r) for g in research['grids']
               if g['grid_id'] in accepted_ids for r in g['daily']]
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(out/'historical_daily.csv', index=False, lineterminator='\n')
    annual = pd.read_csv(temporal/'annual.csv')
    annual.loc[annual.grid_id.isin(accepted_ids)].to_csv(out/'historical_annual.csv', index=False, lineterminator='\n')
    old_cases = read_json(temporal/'case_temporal.json')
    write_json(out/'historical_grid_summary.json', [g for g in old_cases['grids'] if g['grid_id'] in accepted_ids])
    write_json(out/'source_catalog.json', dict(evidence_snapshot_date=config['evidence_snapshot_date'],
                                            sources=catalog['sources'],
                                            sites=[s for s in catalog['sites'] if s['site_id'] in config['primary_site_ids']]))
    inventory = read_json(record(root/'artifact_manifest.json'))
    expected_files = {Path(r['path']).as_posix(): r['sha256'] for r in inventory['files']}
    calibrators = read_json(record(root/'calibration/calibrators.json', expected_files[(root/'calibration/calibrators.json').as_posix()]))
    cal = {(r['scope'], r['lag'], r['model_id']): r for r in calibrators['models']}
    frames, bundles = [], 0
    for scope in ('temporal', 'strict'):
        grids = grids_for_scope(mapping, scope)
        grid_lookup = grids.set_index('grid_index')
        for lag in config['lags']:
            for model in config['model_ids']:
                checkpoint = read_json(record(root/f'evaluation/checkpoints/{scope}_lag{lag}_{model}.json'))
                hashes = {Path(p).as_posix(): h for p, h in checkpoint['artifacts_sha256'].items()}
                for year in config['final_years']:
                    path = root/f'enriched/final_test/{scope}/lag{lag}/{model}/{year}.npz'
                    record(path, hashes[path.as_posix()])
                    dates_path = root/f'dictionaries/{year}/date_dictionary_lag{lag}.csv'
                    dates = pd.read_csv(record(dates_path, expected_files[dates_path.as_posix()]))
                    grid_path = root/f'dictionaries/{year}/grid_dictionary.csv'
                    dictionary = pd.read_csv(record(grid_path, expected_files[grid_path.as_posix()]), index_col='grid_index')
                    if not dictionary.equals(registry):
                        raise ValueError('Frozen grid dictionary changed')
                    expected_dates = pd.date_range(f"{year}-{config['prediction_period']['start_month_day']}",
                                                   f"{year}-{config['prediction_period']['end_month_day']}").strftime('%Y-%m-%d').tolist()
                    if dates.target_date_bjt.tolist() != expected_dates:
                        raise ValueError('Prediction calendar changed')
                    with np.load(path, allow_pickle=False) as z:
                        if z['model_id'].item() != model or z['lag_days'].item() != lag:
                            raise ValueError('Prediction metadata differs from path')
                        selected = np.isin(z['grid_index'], grids.grid_index.astype(int))
                        if not np.all(z['sample_eligibility'][selected] == 1):
                            raise ValueError('Ineligible sample selected')
                        columns = ['p_raw', 'p_calibrated', 'y', 'evaluation_previous_y',
                                   'day_index', 'grid_index', 'alert_primary', 'alert_budget']
                        frame = pd.DataFrame({k: z[k][selected] for k in columns})
                        for k in ('primary_threshold', 'budget_threshold', 'protocol_version'):
                            frame[k] = z[k].item()
                        if z['primary_threshold'].item() != 0.5 or z['budget_threshold'].item() != cal[(scope, lag, model)]['budget']['threshold']:
                            raise ValueError('Alert threshold differs from calibration freeze')
                    frame = frame.merge(dates, on='day_index', validate='many_to_one')
                    for name in ('primary_grid_id', 'grid_latitude', 'grid_longitude', 'spatial_role'):
                        frame[name] = frame.grid_index.map(grid_lookup[name])
                    frame['scope'], frame['lag'], frame['model_id'], frame['year'] = scope, lag, model, year
                    frames.append(frame)
                    bundles += 1
                print(f'extracted {scope} D-{lag} {model}', flush=True)
    predictions = pd.concat(frames, ignore_index=True).sort_values(
        ['scope', 'lag', 'model_id', 'year', 'primary_grid_id', 'target_date_bjt']).reset_index(drop=True)
    validation = verify_subset(predictions, history, config['final_years'],
                               config['prediction_period']['days_per_summer'], config['threshold_c'])
    predictions.to_csv(out/'predictions.csv', index=False, lineterminator='\n', float_format='%.17g')
    for name, data in describe_predictions(predictions, config['evaluation_policy']['risk_sets']).items():
        data.to_csv(out/f'{name}.csv', index=False, lineterminator='\n', float_format='%.17g')
    validation.update(site_count=len(mapping), accepted_primary_sites=int(mapping.temporal_available.sum()),
                      distinct_accepted_primary_grids=len(accepted_ids), historical_grid_days=len(history),
                      historical_grid_years=int(annual.grid_id.isin(accepted_ids).sum()),
                      source_prediction_bundles=bundles, gates=gates,
                      no_training_or_calibration=True, no_test_based_model_selection=True,
                      source_site_snapshot=config['evidence_snapshot_date'])
    for path, value in inputs.items():
        if sha(path) != value:
            raise ValueError(f'Input changed during extraction: {path}')
    write_json(out/'validation.json', validation)
    output_names = ['site_mapping.csv', 'historical_daily.csv', 'historical_annual.csv',
                    'historical_grid_summary.json', 'source_catalog.json', 'predictions.csv',
                    'annual_scores.csv', 'aggregate_scores.csv', 'reliability.csv',
                    'annual_alerts.csv', 'validation.json']
    write_json(out/'extraction_manifest.json', dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        command='python tools/probability_phase5_sites.py --config '+str(config_path),
        software=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
        source_code_sha256={p: sha(p) for p in ['src/probability_site_evidence.py', 'tools/probability_phase5_sites.py']},
        inputs_sha256=inputs,
        outputs_sha256={(out/name).as_posix(): sha(out/name) for name in output_names},
        validation=validation))
    print(json.dumps(validation, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/probability_phase5_sites.json'))
    args = parser.parse_args()
    run(args.config)

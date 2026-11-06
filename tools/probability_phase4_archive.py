"""Verify completed phase-four deliverables, then hash artifacts without copies."""
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

repo = Path.cwd()
sys.path.insert(0, str(repo))
from src.probability_phase4_data import check_gate, sha256

root = Path('data/probability_phase4')
read = lambda path: json.loads(Path(path).read_text(encoding='utf-8-sig'))
state = read(root / 'evaluation/execution_state.json')
if state['status'] != 'completed':
    raise ValueError('Formal evaluation must already be complete')
gates = check_gate(root, 'final_test')
config = read('configs/probability_phase4_freeze.json')
models = config['models']
assert state['model_instances'] == 104 and state['model_year_bundles'] == 520
assert len(read(root / 'calibration/calibrators.json')['models']) == 104

verified_predictions = 0
for stage, years in [('calibration', [2018,2019]), ('final_test', list(range(2020,2025)))]:
    expected_gates = {k:v for k,v in gates.items() if stage == 'final_test' or k == 'scientific_lock.json'}
    for model in models:
        for year in years:
            path = root / 'predictions' / stage / model['scope'] / f"lag{model['lag']}" / model['model_id'] / f'{year}.npz'
            receipt = read(path.with_suffix('.json'))
            assert receipt['gates'] == expected_gates
            assert sha256(path) == receipt['output_sha256'], path
            assert (root / 'enriched' / stage / model['scope'] / f"lag{model['lag']}" / model['model_id'] / f'{year}.npz').is_file()
            verified_predictions += 1

for filename, digest in read(root / 'evaluation/run_summary.json')['outputs_sha256'].items():
    assert sha256(filename) == digest, filename

raw_inputs = {}
for line in (root / 'access_ledger.jsonl').read_text(encoding='utf-8').splitlines():
    item = json.loads(line)
    if item['status'] == 'read' and item['purpose'] == 'final_integrity_extraction':
        raw_inputs[item['path']] = item['sha256']
assert len(raw_inputs) == 20
for filename, digest in raw_inputs.items():
    assert sha256(filename) == digest, filename

# Independent direct arithmetic, separate from the daily-statistic evaluator.
scores = pd.read_csv(root / 'evaluation/annual_scores.csv')
checks = 0
for scope in ['temporal','strict']:
    for lag in [1,2]:
        for year in [2020,2024]:
            for model in ['logistic_F4','lightgbm_F4']:
                path = root / 'enriched/final_test' / scope / f'lag{lag}' / model / f'{year}.npz'
                with np.load(path, allow_pickle=False) as z:
                    for kind, field in [('raw','p_raw'),('calibrated','p_calibrated')]:
                        masks = {'all':np.ones(len(z['y']),dtype=bool), 'onset':z['evaluation_previous_y']==0,
                                 'persistence_or_clearance':z['evaluation_previous_y']==1}
                        for risk, mask in masks.items():
                            row = scores[(scores.scope==scope)&(scores.lag==lag)&(scores.year==year)&
                                (scores.model_id==model)&(scores.probability==kind)&(scores.risk_set==risk)].iloc[0]
                            assert row.n == mask.sum()
                            assert row.events == z['y'][mask].sum()
                            np.testing.assert_allclose(row.brier, np.mean((z[field][mask]-z['y'][mask])**2), rtol=1e-12, atol=1e-14)
                            checks += 1

report_receipt = read(root / 'report_receipt.json')
for filename,digest in report_receipt['output_sha256'].items():
    assert sha256(filename)==digest,filename
verification = {'completed_utc':datetime.now(timezone.utc).isoformat(), 'gates':gates,
    'raw_prediction_receipts_verified':verified_predictions,
    'enriched_bundles_present':verified_predictions,
    'original_raw_input_files_unchanged_since_access':len(raw_inputs),
    'independent_annual_brier_and_count_checks':checks,
    'evaluation_output_hashes_verified':len(read(root / 'evaluation/run_summary.json')['outputs_sha256']),
    'report_figure_hashes_verified':len(report_receipt['output_sha256']),
    'result':'passed', 'no_new_models_thresholds_or_scientific_choices':True}
(root / 'completion_verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8',newline='\n')

paths = [p for p in root.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json']
for directory, pattern in [('src','probability_phase4_*.py'),('tools','probability_phase4_*.py'),
                           ('tests','test_probability_phase4_*.py'),('docs','PROBABILITY_PHASE4_*.md')]:
    paths.extend(Path(directory).glob(pattern))
paths += [Path('docs/PROBABILITY_MODEL_CARD.md'), Path('configs/probability_phase4_freeze.json')]
paths += [p for p in Path('figures/probability_phase4').glob('*') if p.is_file()]
entries = [{'path':p.as_posix(),'bytes':p.stat().st_size,'sha256':sha256(p),
            'local_array_not_copied_to_git':p.suffix in ('.npz','.npy','.nc')} for p in sorted(set(paths))]
manifest = {'created_at_utc':datetime.now(timezone.utc).isoformat(),
    'purpose':'byte inventory only; no large data copy', 'files':entries,
    'total_files':len(entries), 'total_bytes':sum(x['bytes'] for x in entries),
    'local_array_files':sum(x['local_array_not_copied_to_git'] for x in entries)}
(root / 'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({'verification':verification,'files':len(entries),'bytes':manifest['total_bytes']}))

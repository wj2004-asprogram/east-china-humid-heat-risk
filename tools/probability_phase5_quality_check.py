"""Targeted final delivery review; read saved predictions, never fit or select."""
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.probability_phase5_sites import read_json, sha, write_json


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.targets=[]
    def handle_starttag(self,tag,attrs):
        self.targets.extend(v for k,v in attrs if k in ('href','src'))


def run():
    config=read_json('configs/probability_phase5_sites.json')
    root=Path(config['source_phase4_root']); data=Path(config['output_root'])
    out=data/'quality_review';out.mkdir(exist_ok=True)
    inventory={r['path'].replace('\\','/'):r['sha256'] for r in read_json(root/'artifact_manifest.json')['files']}
    inputs={}; checks=[]

    def record(path):
        path=Path(path); name=path.as_posix(); digest=sha(path)
        if name in inventory: assert digest==inventory[name],f'Frozen artifact changed: {name}'
        inputs[name]=digest
        return path

    def csv(path):return pd.read_csv(record(path),float_precision='round_trip')
    comparisons=csv(root/'evaluation/comparisons.csv')
    intervals=csv(root/'evaluation/bootstrap_intervals.csv')
    annual=csv(root/'evaluation/annual_scores.csv')

    def pick(frame,**keys):
        selected=frame
        for key,value in keys.items():selected=selected[selected[key]==value]
        assert len(selected)==1,(keys,len(selected))
        return selected.iloc[0]

    base=dict(candidate_probability='raw',reference_probability='raw',weighting='equal_grid')
    conclusions=Path('docs/PROBABILITY_PHASE4_INTERPRETATION.md').read_text(encoding='utf-8')
    rows=[line.split('|')[1:-1] for line in conclusions.splitlines() if line.startswith('| 全域 |') or line.startswith('| H |')]
    assert len(rows)==8
    moisture=[]
    for cells in rows:
        cells=[s.strip().replace('−','-') for s in cells]
        scope='temporal' if cells[0]=='全域' else 'strict'
        lag=int(cells[1][-1]);risk='all' if cells[2]=='全体' else 'onset'
        keys=dict(scope=scope,lag=lag,risk_set=risk,candidate='lightgbm_F1',reference='lightgbm_F0',**base)
        row=pick(comparisons,**keys); ci=pick(intervals,block_days=14,**keys)
        bounds=[float(v) for v in re.findall(r'[-+]?\d+\.\d+',cells[4])]
        assert abs(float(cells[3])-row.delta_brier)<=.0000500001
        assert np.allclose(bounds,[ci.delta_brier_lower,ci.delta_brier_upper],rtol=0,atol=.0000500001)
        assert int(cells[5].split('/')[0])==row.years_improving
        moisture.append(dict(scope=scope,lag=lag,risk_set=risk,delta_brier=row.delta_brier,
                             lower=ci.delta_brier_lower,upper=ci.delta_brier_upper,
                             years_improving=int(row.years_improving)))
    checks.append('eight_reported_moisture_contrasts_match_frozen_tables_and_14_day_intervals')

    # Independently compute squared errors from 30 selected archived bundles.
    direct=[];site_direct=[]
    selected_models={'temporal':['lightgbm_F0','lightgbm_F1','logistic_F4','lightgbm_F4'],
                     'strict':['lightgbm_F0','lightgbm_F1']}
    for scope,models in selected_models.items():
        for model in models:
            for year in config['final_years']:
                path=record(root/f'enriched/final_test/{scope}/lag2/{model}/{year}.npz')
                with np.load(path,allow_pickle=False) as z:
                    y,previous=z['y'],z['evaluation_previous_y']
                    for probability,column in [('raw','p_raw'),('calibrated','p_calibrated')]:
                        p=z[column]
                        for risk,mask in [('all',np.ones(y.shape,dtype=bool)),('onset',previous==0)]:
                            value=float(np.mean((p[mask]-y[mask])**2))
                            row=pick(annual,scope=scope,lag=2,model_id=model,year=year,probability=probability,risk_set=risk)
                            assert abs(value-row.brier)<1e-13
                            assert int(mask.sum())==row.n and int(y[mask].sum())==row.events
                            direct.append(dict(scope=scope,lag=2,model_id=model,year=year,
                                               probability=probability,risk_set=risk,brier=value))
                        if scope=='temporal' and model=='logistic_F4':
                            for index in [772,887]:
                                mask=(z['grid_index']==index)&(previous==0)
                                site_direct.append(dict(grid_index=index,year=year,probability=probability,
                                                        n=int(mask.sum()),events=int(y[mask].sum()),
                                                        brier=float(np.mean((p[mask]-y[mask])**2))))
    checks.append('30_frozen_bundles_independently_match_120_annual_brier_scores_and_counts')
    direct_frame=pd.DataFrame(direct)
    for scope in ['temporal','strict']:
        subset=direct_frame[(direct_frame.scope==scope)&(direct_frame.probability=='raw')&
                            (direct_frame.risk_set=='onset')]
        means=subset.groupby('model_id').brier.mean()
        delta=float(means.lightgbm_F1-means.lightgbm_F0)
        row=pick(comparisons,scope=scope,lag=2,risk_set='onset',candidate='lightgbm_F1',reference='lightgbm_F0',**base)
        assert abs(delta-row.delta_brier)<1e-13
        ci=pick(intervals,scope=scope,lag=2,risk_set='onset',candidate='lightgbm_F1',reference='lightgbm_F0',block_days=14,**base)
        assert ci.delta_brier_lower<0<ci.delta_brier_upper
        if scope=='temporal':assert delta>0

    calibration=[]
    for model in ['logistic_F4','lightgbm_F4']:
        sub=direct_frame[(direct_frame.scope=='temporal')&(direct_frame.model_id==model)&(direct_frame.risk_set=='onset')]
        means=sub.groupby('probability').brier.mean();delta=float(means.calibrated-means.raw)
        ci=pick(intervals,scope='temporal',lag=2,candidate=model,reference=model,
                candidate_probability='calibrated',reference_probability='raw',risk_set='onset',weighting='equal_grid',block_days=14)
        assert delta>0 and ci.delta_brier_lower>0 and ci.delta_brier_upper>0
        calibration.append(dict(model_id=model,raw=float(means.raw),calibrated=float(means.calibrated),
                                delta=delta,lower=ci.delta_brier_lower,upper=ci.delta_brier_upper))
    sites=csv(data/'aggregate_scores.csv')
    mapping=csv(data/'site_mapping.csv')
    for index,part in pd.DataFrame(site_direct).groupby('grid_index'):
        grid=mapping[mapping.grid_index==index].primary_grid_id.iloc[0]
        for probability,records in part.groupby('probability'):
            row=pick(sites,scope='temporal',lag=2,model_id='logistic_F4',primary_grid_id=grid,risk_set='onset',probability=probability)
            assert abs(records.brier.mean()-row.brier)<1e-13
            assert records.n.sum()==row.n and records.events.sum()==row.events
    checks.append('regional_and_site_D_minus_2_onset_calibration_counterexamples_preserved')

    gates=read_json(record(root/'evaluation/application_gate.json'))
    assert len(gates)==2 and all(not g['passed'] for g in gates)
    corrections=[]
    for gate in gates:
        for cell in gate['cells']:
            for ci in cell['intervals']:
                if ci['delta_brier_upper'] is None or ci['delta_brier_upper']>=0:
                    corrections.append(dict(scope=gate['scope'],cell=cell['cell'],block_days=ci['block_days'],
                        original_label=f"block_{ci['block_days']}_interval_does_not_exclude_zero",
                        corrected_meaning='interval_not_strictly_below_zero',
                        delta_brier_lower=ci['delta_brier_lower'],delta_brier_upper=ci['delta_brier_upper'],
                        interval_entirely_positive=ci['delta_brier_lower'] is not None and ci['delta_brier_lower']>0,
                        decision_changed=False))
    write_json(out/'application_gate_label_annotations.json',dict(
        source='data/probability_phase4/evaluation/application_gate.json',
        interpretation='The preserved label means upper endpoint is unavailable or >=0, not necessarily that the interval contains zero.',
        gate_decisions_unchanged=True,annotations=corrections))
    checks.append('both_frozen_application_gates_fail_correctly_label_ambiguity_annotated_without_rewriting_seals')
    calibrators=read_json(record(root/'calibration/calibrators.json'))['models']
    statuses=pd.Series([r['mapping']['status'] for r in calibrators]).value_counts().to_dict()
    assert statuses=={'fitted':100,'persistence_raw':4}
    assert all(r['fit_years']==[2018,2019] for r in calibrators)
    diagnostic=[]
    for stage,path in [('calibration',root/'calibration/diagnostics.csv'),('final_test',root/'evaluation/calibration_diagnostics.csv')]:
        frame=csv(path); failed=frame[frame.status!='fitted']
        assert len(failed)==(1 if stage=='calibration' else 3)
        assert failed.intercept.isna().all() and failed.slope.isna().all()
        diagnostic.extend(dict(stage=stage,**row) for row in failed[['scope','lag','model_id','year','probability','status']].to_dict('records'))
    checks.append('100_successful_calibration_maps_four_raw_persistence_and_four_NA_diagnostics_are_distinct')

    registry=csv(config['grid_registry']); cards=read_json(record(data/'site_cards.json'))['cards']
    for site in mapping.itertuples():
        found=registry[(registry.latitude==site.grid_latitude)&(registry.longitude==site.grid_longitude)]
        if site.site_id=='factory_001':
            assert found.empty and not site.temporal_available and not site.strict_H_available
            assert cards[0]['historical'] is None and cards[0]['model_summary']==[]
        else:
            assert len(found)==1 and found.index[0]==site.grid_index
            assert found.iloc[0].spatial_role==site.spatial_role
    assert mapping.iloc[1:4].primary_grid_id.nunique()==1
    assert (mapping.iloc[1:4].spatial_role=='holdout').all()
    assert mapping.iloc[4].spatial_role=='buffer' and not mapping.iloc[4].strict_H_available
    assert cards[1]['model_summary']==cards[2]['model_summary']==cards[3]['model_summary']
    prediction=csv(data/'predictions.csv')
    assert set(prediction[prediction.scope=='strict'].primary_grid_id)=={'30.2500,120.5000'}
    assert len(prediction[['primary_grid_id','target_date_bjt']].drop_duplicates())==840
    assert not prediction.primary_grid_id.isin(['30.7500,121.7500','31.0000,121.7500']).any()
    checks.append('H_buffer_mask_and_shared_grid_match_frozen_registry_and_all_delivered_predictions')

    # Check real entry points and current local prerequisites without regeneration.
    entry=Path(config['report_root'])/'index.html';parser=Links();parser.feed(entry.read_text(encoding='utf-8'))
    for target in parser.targets:
        url=urlparse(target)
        if not url.scheme and url.path: assert (entry.parent/unquote(url.path)).resolve().is_file(),target
    commands=[]
    for script in ['tools/probability_phase5_sites.py','tools/probability_phase5_report.py']:
        result=subprocess.run([sys.executable,script,'--help'],capture_output=True,text=True,encoding='utf-8')
        assert result.returncode==0,(script,result.stderr)
        commands.append(dict(command=[sys.executable,script,'--help'],exit_code=result.returncode))
    source_manifest=read_json(data/'extraction_manifest.json')
    assert all(Path(p).exists() for p in source_manifest['inputs_sha256'])
    record(entry);record('docs/PROBABILITY_PHASE5_REPRODUCE.md')
    checks.append('report_local_links_cli_imports_and_all_declared_existing_extraction_inputs_available')
    receipt=dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),status='passed',
                 source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                 source_code_sha256=sha(__file__),checks=checks,
                 selected_npz_bundles=30,independent_annual_brier_checks=len(direct),
                 moisture_contrasts=moisture,regional_onset_calibration=calibration,
                 diagnostic_failures=diagnostic,
                 gate_label_annotations=len(corrections),
                 entirely_positive_intervals_with_legacy_label=sum(r['interval_entirely_positive'] for r in corrections),
                 command_smoke_checks=commands,input_sha256=inputs,
                 model_training=False,model_selection=False,calibration_refit=False,
                 entire_test_suite_rerun=False,external_data_published=False)
    write_json(out/'review_receipt.json',receipt)
    print(json.dumps({k:receipt[k] for k in ['status','selected_npz_bundles','independent_annual_brier_checks','gate_label_annotations','entirely_positive_intervals_with_legacy_label']},ensure_ascii=False))


if __name__=='__main__':run()

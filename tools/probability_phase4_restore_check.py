"""Training-only restoration parity check for all frozen registrations."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.probability_phase4_data import predict_frozen,read_logged,sha256

def main():
    root=Path('data/probability_phase4/engineering_validation'); root.mkdir(parents=True,exist_ok=True)
    ledger=root/'access_ledger.jsonl'
    with read_logged('configs/probability_phase4_freeze.json',ledger,'restoration_validation') as p: config=json.loads(p.read_text())
    with read_logged('data/probability_phase1/samples/grid_registry.csv',ledger,'geometry') as p: registry=pd.read_csv(p)
    with read_logged('data/probability_phase2/features/2000/feature_registry.json',ledger,'training_schema') as p: names=json.loads(p.read_text())['names']
    results=[]
    for scope in ('temporal','strict'):
        with read_logged(f'data/probability_phase2/features/2000/{scope}.npz',ledger,'training_only_2000') as p:
            with np.load(p,allow_pickle=False) as z: data={k:z[k][:64] for k in z.files}
        for m in config['models']:
            if m['scope']!=scope: continue
            with read_logged(m['metadata_path'],ledger,'frozen_model') as p: state=json.loads(p.read_text())
            booster=None
            if m['learner']=='lightgbm':
                import lightgbm as lgb
                with read_logged(m['model_path'],ledger,'frozen_booster') as p: booster=lgb.Booster(model_file=str(p))
            prob=predict_frozen(state,data,names,registry,2000,m['lag'],scope,m['model_id'],booster)
            base=Path(m['metadata_path']).parents[2]
            prediction=base/'predictions'/scope/f"lag{m['lag']}"/m['model_id']/'2000.npz'
            with read_logged(prediction,ledger,'training_prediction_parity') as p:
                with np.load(p,allow_pickle=False) as z: expected=z['p_raw'][:64]
            np.testing.assert_allclose(prob,expected,rtol=1e-11,atol=1e-12)
            results.append({'scope':scope,'lag':m['lag'],'model':m['model_id'],'max_absolute_error':float(np.max(np.abs(prob-expected)))})
    if len(results)!=104: raise ValueError('Incomplete restoration checks')
    (root/'restoration_2000.json').write_text(json.dumps({'year':2000,'rows_per_instance':64,'instances':results},indent=2))
    print(f'{len(results)} frozen model instances restored with prediction parity')

if __name__=='__main__': main()

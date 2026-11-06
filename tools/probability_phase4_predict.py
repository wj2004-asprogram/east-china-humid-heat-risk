"""Restore frozen models and emit a single annual, auditable prediction bundle."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from src.probability_phase4_data import *


def main(argv=None, _gates=None):
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('data/probability_phase4'))
    p.add_argument('--year',type=int,choices=range(2018,2025),required=True)
    p.add_argument('--scope',choices=['temporal','strict'])
    p.add_argument('--lag',type=int,choices=[1,2])
    p.add_argument('--model')
    p.add_argument('--batch',action='store_true')
    a=p.parse_args(argv)
    stage='calibration' if a.year<2020 else 'final_test'
    gates=_gates if _gates is not None else check_gate(a.root,stage)
    config=json.loads(Path('configs/probability_phase4_freeze.json').read_text(encoding='utf-8-sig'))
    models=config['models']
    keys={(m['scope'],m['lag'],m['model_id']) for m in models}
    if len(models)!=104 or len(keys)!=104: raise ValueError('Expected 104 distinct frozen registrations')
    if a.batch:
        for m in models:
            main(['--root',str(a.root),'--year',str(a.year),'--scope',m['scope'],'--lag',str(m['lag']),'--model',m['model_id']],_gates=gates)
        return
    if (a.scope,a.lag,a.model) not in keys: raise ValueError('Unregistered model configuration')
    registration=next(m for m in models if (m['scope'],m['lag'],m['model_id'])==(a.scope,a.lag,a.model))
    out=a.root/'predictions'/stage/a.scope/f'lag{a.lag}'/a.model
    out.mkdir(parents=True,exist_ok=True)
    target=out/f'{a.year}.npz'
    receipt=out/f'{a.year}.json'
    if target.exists() or receipt.exists():
        if not target.exists() or not receipt.exists(): raise ValueError('Incomplete existing output')
        saved=json.loads(receipt.read_text())
        if saved['gates']!=gates or saved['output_sha256']!=sha256(target): raise ValueError('Corrupt or differently sealed output')
        for field in ('inputs_sha256','source_sha256'):
            if any(sha256(path)!=h for path,h in saved[field].items()): raise ValueError('Resume input/source mismatch')
        return
    ledger=a.root/'access_ledger.jsonl'; inputs={}
    def read(path,kind):
        with read_logged(path,ledger,kind) as path:
            inputs[str(path)]=sha256(path)
            if path.suffix=='.npz':
                with np.load(path,allow_pickle=False) as z: return {k:z[k] for k in z.files}
            if path.suffix=='.csv': return pd.read_csv(path)
            return json.loads(path.read_text(encoding='utf-8-sig'))
    try:
        base=Path('data/probability_phase2/features') if stage=='calibration' else a.root/'features'
        data=read(base/str(a.year)/f'{a.scope}.npz','features')
        names=read(base/str(a.year)/'feature_registry.json','feature_schema')['names']
        registry=read(Path('data/probability_phase1/samples/grid_registry.csv'),'geometry')
        write_dictionaries(a.root/'dictionaries'/str(a.year),registry,a.year)
        if a.scope=='strict':
            expected='train' if stage=='calibration' else 'holdout'
            if not registry.iloc[data['grid_index']].spatial_role.eq(expected).all(): raise ValueError('Forbidden strict spatial role')
        identity=hashlib.sha256(data['day_index'].tobytes()+data['grid_index'].tobytes()).hexdigest()
        if stage=='final_test':
            sealed=json.loads((a.root/'test_open_lock.json').read_text())
            if sealed['final_common_index_sha256'].get(f'{a.scope}/{a.year}')!=identity: raise ValueError('Final index differs from test-open seal')
        modelpath=Path(registration['metadata_path'])
        if sha256(modelpath)!=registration['metadata_sha256'] or sha256(registration['model_path'])!=registration['model_sha256']:
            raise ValueError('Frozen model registration hash mismatch')
        state=read(modelpath,'frozen_model'); booster=None
        if a.model.startswith('lightgbm_'):
            import lightgbm as lgb
            with read_logged(Path(registration['model_path']),ledger,'frozen_booster') as path:
                inputs[str(path)]=sha256(path); booster=lgb.Booster(model_file=str(path))
        probability=predict_frozen(state,data,names,registry,a.year,a.lag,a.scope,a.model,booster)
        if not np.isfinite(probability).all() or np.any((probability<0)|(probability>1)): raise ValueError('Invalid probability')
        np.savez_compressed(target,p_raw=probability,day_index=data['day_index'],grid_index=data['grid_index'],y=data['y'],evaluation_previous_y=data['previous_y'])
        receipt.write_text(json.dumps({'gates':gates,'inputs_sha256':inputs,'output_sha256':sha256(target),'sample_index_sha256':hashlib.sha256(data['day_index'].tobytes()+data['grid_index'].tobytes()).hexdigest(),'source_sha256':{str(Path(__file__)):sha256(__file__),'src/probability_phase4_data.py':sha256('src/probability_phase4_data.py')}} ,indent=2))
    except Exception as exc:
        (out/f'{a.year}.error.json').write_text(json.dumps({'error':repr(exc)})); raise

if __name__=='__main__': main()

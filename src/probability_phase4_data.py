"""Phase-four feature and frozen prediction primitives; no fitting or implicit I/O."""
import numpy as np
import pandas as pd
from src.probability_complete_features import (checked_hours, hourly_fields, DAILY, NIGHT,
    land_arrays, neighbor_statistics, feature_matrix)
import xarray as xr


def complete_final_daily_night(hourly, year):
    if year not in range(2020, 2025):
        raise ValueError('Final extraction accepts 2020-2024 only')
    time_name, actual = checked_hours(hourly)
    dates = pd.date_range(f'{year}-06-01', f'{year}-08-31')
    expected = pd.date_range(dates[0]-pd.Timedelta(hours=8), periods=92*24, freq='h')
    if not expected.isin(actual).all():
        raise ValueError('Missing UTC hour')
    fields = hourly_fields(hourly.sel({time_name: expected}))
    shape = (92, 24, fields.sizes['latitude'], fields.sizes['longitude'])
    arrays = {k: fields[k].transpose(time_name, 'latitude', 'longitude').values.reshape(shape) for k in fields}
    values = {k: getattr(np, op)(arrays[src], axis=1) for k, (src, op) in DAILY.items()}
    for k, (src, op) in NIGHT.items():
        night = np.concatenate([arrays[src][:-1,20:], arrays[src][1:,:8]], axis=1)
        values[k] = np.full((92,*shape[2:]), np.nan)
        values[k][1:] = getattr(np,op)(night,axis=1)
    night = np.concatenate([arrays['temperature_c'][:-1,20:],arrays['temperature_c'][1:,:8]],axis=1)
    values['night_cooling20to07_c'] = np.full((92,*shape[2:]),np.nan)
    values['night_cooling20to07_c'][1:] = np.where(np.isfinite(night).all(axis=1),night[:,0]-night[:,-1],np.nan)
    return xr.Dataset({k:(('date','latitude','longitude'),v) for k,v in values.items()},coords={'date':dates,'latitude':fields.latitude,'longitude':fields.longitude})


def assemble_phase4(daily, registry, sources, *, scope, stage):
    if scope not in ('temporal','strict') or stage not in ('calibration','final_test'):
        raise ValueError('Unknown scope/stage')
    role = 'train' if stage == 'calibration' else 'holdout'
    eligible = np.array([len(s)>=3 for s in sources])
    if scope == 'strict':
        eligible &= registry.spatial_role.eq(role).to_numpy()
        for i, ss in enumerate(sources):
            if eligible[i] and not registry.iloc[ss].spatial_role.eq(role).all():
                raise ValueError('Cross-boundary strict neighborhood')
    arrays = land_arrays(daily,registry)
    regional = {k:neighbor_statistics(arrays[k],sources) for k in ('tmean_c','tdmean_c','rhmean_percent')}
    x1,names,groups = feature_matrix(arrays,daily.date.values,registry,regional,1)
    x2,names2,_ = feature_matrix(arrays,daily.date.values,registry,regional,2)
    if names!=names2 or len(names)!=117: raise ValueError('Feature schema mismatch')
    hi=arrays['himax_c']; y=np.where(np.isfinite(hi),(hi>=40.6).astype(float),np.nan)
    common=eligible[None,:]&np.isfinite(x1).all(2)&np.isfinite(x2).all(2)&np.isfinite(y[8:])&np.isfinite(y[7:-1])
    day,grid=np.where(common)
    payload={'X1':x1[common],'X2':x2[common],'day_index':day.astype(np.int16),'grid_index':grid.astype(np.int16),'y':y[8:][common].astype(np.int8),'previous_y':y[7:-1][common].astype(np.int8),'state1':y[7:-1][common].astype(np.int8),'state2':y[6:-2][common].astype(np.int8)}
    return payload,{'common_samples':int(common.sum()),'geometry_eligible':int(eligible.sum()),'excluded_within_geometry':int(eligible.sum()*84-common.sum())},{'names':names,'groups':groups}


def predict_frozen(state, payload, names, registry, year, lag, scope, model_id, booster=None):
    if booster is not None:
        cols=[names.index(n) for n in state['exact_features']]
        return booster.predict(payload[f'X{lag}'][:,cols],num_threads=1)
    if model_id.startswith('logistic_'):
        cols=[names.index(n) for n in state['features']]
        x=payload[f'X{lag}'][:,cols].astype(np.float64)
        z=((x-np.asarray(state['scaler_mean']))/np.asarray(state['scaler_scale']))@np.asarray(state['coefficient'])+state['intercept']
        from scipy.special import expit
        return expit(z)
    if model_id=='persistence': return payload[f'state{lag}'].astype(float)
    from src.probability_baselines import ClimatologyBaseline, TransitionBaseline
    import copy
    state=copy.deepcopy(state)
    if scope=='strict' and (not state['strict_spatial'] or state['grid_counts']):
        raise ValueError('Strict baselines must use pooled T-only state')
    model=(TransitionBaseline if model_id=='transition' else ClimatologyBaseline)(cutoff_lag_days=lag,strict_spatial=scope=='strict')
    model.state=state
    frame=registry.iloc[payload['grid_index']].reset_index(drop=True).copy()
    frame['date']=pd.Timestamp(year,6,9)+pd.to_timedelta(payload['day_index'],unit='D')
    frame[f'state_lag{lag}']=payload[f'state{lag}']
    frame[f'state_date_lag{lag}']=frame.date-pd.Timedelta(days=lag)
    return model.predict(frame,cutoff_lag_days=lag)

import hashlib
import json
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def check_gate(root, stage):
    root=Path(root)
    required=[('scientific_lock.json','scientific_frozen')]
    if stage in ('extract_final','final_test'): required.append(('calibration_lock.json','calibration_frozen'))
    if stage=='final_test': required.append(('test_open_lock.json','test_open_frozen'))
    if stage not in ('calibration','extract_final','final_test'): raise ValueError('Unknown gate stage')
    hashes={}
    for name,status in required:
        path=root/name
        data=json.loads(path.read_text(encoding='utf-8-sig'))
        if data.get('status')!=status: raise ValueError(f'Gate not sealed: {path}')
        for key,value in data.items():
            if key in ('source_code_sha256','artifacts_sha256','calibration_artifacts_sha256','final_input_artifacts_sha256'):
                for filename,expected in value.items():
                    if sha256(filename)!=expected: raise ValueError(f'Gate hash mismatch: {filename}')
        for previous,expected in hashes.items():
            key=previous.replace('.json','_sha256')
            if data.get(key)!=expected: raise ValueError(f'Gate chain mismatch: {key}')
        if name=='scientific_lock.json':
            for field,filename in [('freeze_config_sha256','configs/probability_phase4_freeze.json'),('protocol_document_sha256','docs/PROBABILITY_PHASE4_PROTOCOL.md')]:
                if data.get(field)!=sha256(filename): raise ValueError(f'Gate configuration mismatch: {filename}')
        hashes[name]=sha256(path)
    return hashes


@contextmanager
def read_logged(path, ledger, purpose):
    path=Path(path); ledger=Path(ledger); ledger.parent.mkdir(parents=True,exist_ok=True)
    record={'path':str(path.resolve()),'purpose':purpose,'time_utc':datetime.now(timezone.utc).isoformat(),'status':'attempt'}
    with ledger.open('a',encoding='utf-8') as f: f.write(json.dumps(record)+'\n')
    try:
        yield path
    except Exception as exc:
        record.update(status='error',error=repr(exc)); raise
    else: record.update(status='read',sha256=sha256(path))
    finally:
        with ledger.open('a',encoding='utf-8') as f: f.write(json.dumps(record)+'\n')


def write_dictionaries(output, registry, year):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    def put_csv(path, content):
        encoded=content.encode('utf-8')
        if path.exists():
            if path.read_bytes()!=encoded: raise ValueError(f'Dictionary differs: {path}')
        else: path.write_bytes(encoded)
    put_csv(output/'grid_dictionary.csv',registry.to_csv(index_label='grid_index',lineterminator='\n'))
    dates=pd.date_range(f'{year}-06-09',f'{year}-08-31',tz='Asia/Shanghai')
    for lag in (1,2):
        fmt=lambda x:x.strftime('%Y-%m-%dT%H:%M:%S%z')
        frame=pd.DataFrame({'day_index':range(84),'target_date_bjt':dates.strftime('%Y-%m-%d'),
            'issue_time_bjt':fmt(dates),'issue_time_utc':fmt(dates.tz_convert('UTC')),
            'target_start_bjt':fmt(dates),'target_start_utc':fmt(dates.tz_convert('UTC')),
            'target_end_bjt_exclusive':fmt(dates+pd.Timedelta(days=1)),
            'target_end_utc_exclusive':fmt((dates+pd.Timedelta(days=1)).tz_convert('UTC')),
            'max_input_valid_time_bjt':fmt(dates-pd.Timedelta(days=lag)+pd.Timedelta(hours=23)),
            'max_input_valid_time_utc':fmt((dates-pd.Timedelta(days=lag)+pd.Timedelta(hours=23)).tz_convert('UTC'))})
        put_csv(output/f'date_dictionary_lag{lag}.csv',frame.to_csv(index=False,lineterminator='\n'))

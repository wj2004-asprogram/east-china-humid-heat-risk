"""Gate-controlled one-year final extraction; never reextract calibration years."""
import argparse,json
from pathlib import Path
from contextlib import ExitStack
import pandas as pd
import numpy as np
import xarray as xr
from src.config import load_config
from src.era5_utils import detect_time_coord,detect_t2m_var,detect_d2m_var
from src.probability_phase4_data import *


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('data/probability_phase4'))
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--source-project-root',type=Path,required=True)
    p.add_argument('--year',type=int,choices=range(2020,2025),required=True)
    p.add_argument('--phase1-dir',type=Path,default=Path('data/probability_phase1'))
    a=p.parse_args(argv); gates=check_gate(a.root,'extract_final')
    out=a.root/'features'/str(a.year); out.mkdir(parents=True,exist_ok=False)
    ledger=a.root/'access_ledger.jsonl'
    try:
        with read_logged(a.config,ledger,'extraction_config') as path: cfg=load_config(path,project_root=a.source_project_root)
        with read_logged(a.phase1_dir/'samples/grid_registry.csv',ledger,'geometry') as path: registry=pd.read_csv(path)
        with read_logged(a.phase1_dir/'geometry/neighbor_sources.json',ledger,'geometry') as path: rows=json.loads(path.read_text())
        with ExitStack() as stack:
            parts=[]
            for raw in cfg.raw_files:
                if raw.year!=a.year: continue
                path=stack.enter_context(read_logged(raw.path,ledger,'final_integrity_extraction'))
                ds=stack.enter_context(xr.open_dataset(path)); parts.append(ds[[detect_t2m_var(ds),detect_d2m_var(ds)]])
            hourly=xr.concat(parts,dim=detect_time_coord(parts[0]),data_vars='all',coords='minimal',compat='equals',join='exact').sortby(detect_time_coord(parts[0])).load()
        daily=complete_final_daily_night(hourly,a.year)
        reports={}
        for scope,key in [('temporal','ordinary_source_row_indices'),('strict','strict_source_row_indices')]:
            data,report,fr=assemble_phase4(daily,registry,[r[key] for r in rows],scope=scope,stage='final_test')
            np.savez_compressed(out/f'{scope}.npz',**data)
            report['sample_index_sha256']=hashlib.sha256(data['day_index'].tobytes()+data['grid_index'].tobytes()).hexdigest()
            reports[scope]=report
        (out/'feature_registry.json').write_text(json.dumps(fr))
        registry.to_csv(out/'grid_dictionary.csv',index_label='grid_index')
        (out/'run.json').write_text(json.dumps({'year':a.year,'gates':gates,'counts':reports,'outputs':{x.name:{'sha256':sha256(x)} for x in out.iterdir() if x.is_file()}},indent=2))
    except Exception as exc:
        (out/'error.json').write_text(json.dumps({'error':repr(exc)})); raise

if __name__=='__main__': main()

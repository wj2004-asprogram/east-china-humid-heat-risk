"""Inventory the small phase-five delivery and record completed verification."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.probability_phase5_sites import read_json, sha, write_json


def main():
    config=read_json('configs/probability_phase5_sites.json')
    out,report=Path(config['output_root']),Path(config['report_root'])
    for filename in ['extraction_manifest.json','report_manifest.json']:
        manifest=read_json(out/filename)
        for field in ['outputs_sha256','source_code_sha256','configuration_sha256']:
            for p,expected in manifest.get(field,{}).items():
                if sha(p)!=expected: raise ValueError(f'Changed artifact: {p}')
    suites=ET.parse(out/'pytest.xml').getroot().findall('testsuite')
    tests=sum(int(s.get('tests','0')) for s in suites)
    failures=sum(int(s.get('failures','0'))+int(s.get('errors','0')) for s in suites)
    if not tests or failures: raise ValueError('Tests have not passed')
    ui=read_json(report/'qa/verification.json')
    if ui['status']!='passed' or ui['errors'] or ui['external_requests']:
        raise ValueError('Local report verification failed')
    rebuild=read_json(out/'rebuild_verification.json')
    if not rebuild['all_eight_numeric_tables_byte_identical']:
        raise ValueError('Rebuild differs')
    for filename,expected in rebuild['deterministic_rebuild_csv_sha256'].items():
        if sha(out/filename)!=expected:raise ValueError(f'Rebuild artifact changed: {filename}')
    diff=subprocess.run(['git','diff','--name-only',config['phase4_commit']],capture_output=True,text=True,check=True).stdout.splitlines()
    allowed={'.gitattributes','README.md'}
    allowed_prefixes=('configs/probability_phase5','data/probability_phase5/',
                      'reports/probability_phase5/','docs/PROBABILITY_PHASE5_',
                      'tools/probability_phase5','src/probability_site_evidence.py',
                      'tests/test_probability_site_evidence.py')
    unexpected=[p for p in diff if p not in allowed and not p.startswith(allowed_prefixes)]
    if unexpected:raise ValueError(f'Unexpected changes to preserved project: {unexpected}')
    receipt=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                 phase4_base_commit=config['phase4_commit'],tests_passed=tests,test_failures=failures,
                 ui=ui,deterministic_numeric_rebuild=True,
                 protected_tracked_files_changed=[],source_inputs_verified_at_extraction=True,
                 model_refit=False,calibration_refit=False,test_based_selection=False,
                 limitations=['No field data or current operating verification.',
                              'No real forecast availability validation.',
                              'Only two valid primary climate sequences; no company-risk interpretation.'])
    write_json(out/'validation_receipt.json',receipt)
    paths={Path('.gitattributes'),Path('README.md'),Path('src/probability_site_evidence.py'),
           Path('tests/test_probability_site_evidence.py')}
    for pattern in ['configs/probability_phase5*.json','tools/probability_phase5*', 'docs/PROBABILITY_PHASE5_*.md']:
        paths.update(Path('.').glob(pattern))
    for folder in [out,report]:
        paths.update(p for p in folder.rglob('*') if p.is_file() and p.name!='artifact_manifest.json')
    if any(p.suffix in ('.nc','.npz','.npy') for p in paths):
        raise ValueError('Large source arrays must not enter this archive')
    files=[dict(path=p.as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(paths)]
    write_json(out/'artifact_manifest.json',dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Small phase-five delivery inventory; no raw weather or phase-four arrays copied',
        phase4_base_commit=config['phase4_commit'],files=files,total_bytes=sum(r['bytes'] for r in files)))
    print(json.dumps(dict(tests_passed=tests,inventory_files=len(files),bytes=sum(r['bytes'] for r in files),
                          protected_files_changed=unexpected),ensure_ascii=False))


if __name__=='__main__':main()

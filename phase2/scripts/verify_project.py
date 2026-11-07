"""Verify both phases with one command; use --rerun to train everything again.

From the repository root: python phase2/scripts/verify_project.py --rerun
Logs are private under .build/verification/. This script does not upload anything.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--rerun', action='store_true', help='Run all ten notebooks in fresh processes first.')
args = parser.parse_args()
started = datetime.now(ZoneInfo('Asia/Kolkata'))
logs = ROOT/'.build/verification'/started.strftime('%Y%m%d-%H%M%S')
logs.mkdir(parents=True, exist_ok=True)
receipt = {'started_at_ist': started.isoformat(), 'rerun_requested': args.rerun,
           'completed': False, 'steps': [], 'logs': str(logs.relative_to(ROOT))}
record = logs/'run.json'
record.write_text(json.dumps(receipt, indent=2)+'\n')

def run(name, script, *extra, cwd=ROOT):
    print(f'{name} ...', flush=True)
    with (logs/f'{name}.log').open('w') as log:
        result = subprocess.run([sys.executable, str(script), *extra], cwd=cwd,
                                stdout=log, stderr=subprocess.STDOUT)
    receipt['steps'].append({'step': name, 'passed': result.returncode == 0})
    record.write_text(json.dumps(receipt, indent=2)+'\n')
    if result.returncode:
        print((logs/f'{name}.log').read_text(), file=sys.stderr)
        raise SystemExit(f'FAIL: {name}. Logs: {logs}')
    print(f'PASS: {name}', flush=True)

if args.rerun:
    run('phase1_notebooks', ROOT/'phase1/scripts/run_all.py')
    run('phase2_notebooks', ROOT/'phase2/scripts/run_all.py')
    run('phase1_exploratory_draft', ROOT/'phase1/scripts/run_all.py',
        str(ROOT/'phase1/notebooks/drafts/06_GMM_Component_Experiment.ipynb'))
    run('shared_index', ROOT/'phase2/scripts/build_index.py')

local_models = all((ROOT/path).exists() for path in [
    'phase1/data/raw/creditcard.csv', 'phase1/data/processed/features.npz',
    'phase1/models/gmm_models.joblib', 'phase2/models/features.npz',
    'phase2/models/denoising_ae.pt'])
if args.rerun or local_models:
    run('model_artifacts', ROOT/'phase2/scripts/check_model_artifacts.py')
else:
    print('Local data/models are absent: checking saved deliverables only. Use --rerun after adding the CSV.')
run('saved_deliverables', ROOT/'phase2/scripts/check_deliverables.py')
run('portable_package', ROOT/'phase2/scripts/package_project.py')

with tempfile.TemporaryDirectory(prefix='portable-', dir=logs) as temp:
    with zipfile.ZipFile(ROOT/'phase2/output/Card_Fraud_Triage_Complete.zip') as archive:
        assert archive.testzip() is None
        archive.extractall(temp)
    copy = Path(temp)/'T011_Card_Fraud_Triage'
    for name, base in [('repository_manifest_sha256.json', copy),
                       ('manifest_sha256.json', copy/'phase2')]:
        manifest = json.loads((copy/'phase2/results'/name).read_text())
        assert all(hashlib.sha256((base/path).read_bytes()).hexdigest() == digest
                   for path, digest in manifest.items()), name
    run('extracted_deliverables', copy/'phase2/scripts/check_deliverables.py', '--no-write', cwd=copy)
receipt.update({'completed': True, 'model_artifacts_checked': bool(args.rerun or local_models),
                'finished_at_ist': datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(),
                'portable_checksums_checked': True})
record.write_text(json.dumps(receipt, indent=2)+'\n')
print(f'PASS: project verification complete. Logs: {logs.relative_to(ROOT)}')

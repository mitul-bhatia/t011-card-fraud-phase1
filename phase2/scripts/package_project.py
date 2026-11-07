"""Package both phases and the single root index, excluding data and model binaries."""
from pathlib import Path
import hashlib
import json
import re
import zipfile

ROOT = Path(__file__).resolve().parents[2]
DESTINATION = ROOT / 'phase2/output/Card_Fraud_Triage_Complete.zip'
PREFIX = Path('T011_Card_Fraud_Triage')
files = {}
for name in ['README.md', 'index.html', 'requirements.txt', '.gitignore', '.nojekyll',
             'CSAI_Project_Instructions_Phase_1_and_Phase_2.pdf']:
    files[name] = (ROOT / name).read_bytes()
for phase in ['phase1', 'phase2']:
    for path in sorted((ROOT / phase).rglob('*')):
        if not path.is_file():
            continue
        local = path.relative_to(ROOT / phase)
        if any(part in ['__pycache__', '.ipynb_checkpoints', 'archive'] for part in local.parts):
            continue
        if path.suffix in ['.zip', '.aux', '.log', '.out', '.toc', '.pyc']:
            continue
        if local.parts[0] == 'models' and path.name != 'README.md':
            continue
        if local.parts[:2] == ('data', 'processed'):
            continue
        if local.parts[:2] == ('data', 'raw') and path.suffix == '.csv':
            continue
        files[path.relative_to(ROOT).as_posix()] = path.read_bytes()
files['phase1/data/raw/README.md'] = (
    'Place the official ULB/Kaggle creditcard.csv here. Both phases reuse this one file.\n'
    'Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n').encode()

def manifests(contents):
    """Hash actual content; manifests exclude themselves to avoid circular hashes."""
    full = {name: hashlib.sha256(content).hexdigest()
            for name, content in sorted(contents.items())
            if not name.endswith('manifest_sha256.json')}
    phase2 = {name.removeprefix('phase2/'): digest
              for name, digest in full.items() if name.startswith('phase2/')}
    return {'phase2/results/manifest_sha256.json': phase2,
            'phase2/results/repository_manifest_sha256.json': full}

# Keep local manifests accurate as well as the manifests inside the ZIP.
# The generated dataset README only exists in the portable copy.
workspace_files = {name: content for name, content in files.items()
                   if (ROOT / name).is_file()}
for name, values in manifests(workspace_files).items():
    (ROOT / name).write_text(json.dumps(values, indent=2) + '\n')

# The extracted package cannot contain its own download archive.
for name, content in list(files.items()):
    if name.endswith('.html'):
        text = content.decode()
        text = re.sub(r'<a href="(?:phase2/)?output/Card_Fraud_Triage_Complete.zip">.*?</a>',
                      '<span>Complete project package: already extracted</span>', text)
        files[name] = text.encode()
    if name.endswith('.md'):
        text = content.decode()
        text = re.sub(r'\[([^\]]+)\]\((?:phase2/)?output/Card_Fraud_Triage_Complete.zip\)',
                      'this extracted project package', text)
        files[name] = text.encode()

for name, values in manifests(files).items():
    files[name] = (json.dumps(values, indent=2) + '\n').encode()
with zipfile.ZipFile(DESTINATION, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for name, content in sorted(files.items()):
        archive.writestr(str(PREFIX/name), content)
with zipfile.ZipFile(DESTINATION) as archive:
    assert archive.testzip() is None
    assert len([name for name in archive.namelist() if name.endswith('/index.html')]) == 1
    assert not any(name.endswith('.pt') or name.endswith('.joblib') or name.endswith('creditcard.csv')
                   for name in archive.namelist())
print(f'Created {DESTINATION.name}: {len(files)} files, {DESTINATION.stat().st_size/1e6:.2f} MB.')

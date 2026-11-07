"""Check saved notebook execution, measured metrics, links, identities and files.

Run from the repository root: python phase2/scripts/check_deliverables.py
This checks the saved artifacts; run_all.py is the separate training rerun.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote

import nbformat
import numpy as np
import pandas as pd
from pypdf import PdfReader
from sklearn.metrics import (auc, average_precision_score, confusion_matrix,
                             precision_recall_curve)

P2 = Path(__file__).resolve().parents[1]
ROOT = P2.parent
P1 = ROOT / 'phase1'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--no-write', action='store_true', help='Check a packaged copy without changing its receipt.')
args = parser.parse_args()
checks = {}
phase1_notebooks = {}
for path in sorted((P1 / 'notebooks').glob('*.ipynb')):
    book = nbformat.read(path, as_version=4)
    nbformat.validate(book)
    cells = [cell for cell in book.cells if cell.cell_type == 'code']
    assert all(cell.execution_count is not None for cell in cells), path.name
    assert not any(output.output_type == 'error' for cell in cells for output in cell.outputs)
    assert book.metadata['execution']['all_code_cells_succeeded']
    assert book.metadata['execution']['fresh_process']
    phase1_notebooks[path.name] = {'code_cells': len(cells), 'errors': 0, 'fresh_process': True}
assert len(phase1_notebooks) == 5
for path in sorted((P2 / 'notebooks').glob('*.ipynb')):
    book = nbformat.read(path, as_version=4)
    nbformat.validate(book)
    cells = [cell for cell in book.cells if cell.cell_type == 'code']
    assert all(cell.execution_count is not None for cell in cells), path.name
    assert not any(output.output_type == 'error' for cell in cells for output in cell.outputs)
    assert book.metadata['execution'].get('all_code_cells_succeeded')
    assert book.metadata['execution'].get('fresh_process')
    checks[path.name] = {'code_cells': len(cells), 'errors': 0,
                         'fresh_process': bool(book.metadata['execution'].get('fresh_process'))}
assert len(checks) == 5
draft_path = P1/'notebooks/drafts/06_GMM_Component_Experiment.ipynb'
draft = nbformat.read(draft_path, as_version=4)
nbformat.validate(draft)
draft_cells = [cell for cell in draft.cells if cell.cell_type == 'code']
assert len(draft_cells) == 6
assert draft.metadata['execution']['all_code_cells_succeeded']
assert draft.metadata['execution']['fresh_process']
assert all(cell.execution_count is not None for cell in draft_cells)
assert not any(output.output_type == 'error' for cell in draft_cells for output in cell.outputs)

scores = pd.read_csv(P2 / 'results/test_scores.csv')
y = scores.Class.to_numpy()
assert len(y) == 56746 and y.sum() == 74
ranking = pd.read_csv(P2 / 'results/model_comparison.csv').set_index('model')
for name, row in ranking.iterrows():
    values = scores[name].to_numpy()
    assert np.isfinite(values).all()
    precision, recall, _ = precision_recall_curve(y, values)
    assert np.isclose(auc(recall, precision), row.pr_auc_trapezoidal)
    assert np.isclose(average_precision_score(y, values), row.average_precision)
    found = int(y[np.argsort(-values, kind='stable')[:100]].sum())
    assert found == row.frauds_at_100
    assert np.isclose(found / 100, row.precision_at_100)

phase1_scores = pd.read_csv(P1 / 'results/test_scores.csv')
phase1_summary = json.loads((P1 / 'results/summary.json').read_text())
assert np.array_equal(y, phase1_scores.Class.to_numpy())
assert np.allclose(scores.GMM, phase1_scores['GMM score'])
for name, key in [('GMM score', 'PR_AUC'), ('One Gaussian score', 'one_gaussian_PR_AUC')]:
    precision, recall, _ = precision_recall_curve(y, phase1_scores[name])
    assert np.isclose(auc(recall, precision), phase1_summary[key])
for _, row in pd.read_csv(P1 / 'results/review_capacities.csv').iterrows():
    order = np.argsort(-phase1_scores['GMM score'].to_numpy(), kind='stable')
    assert y[order[:int(row['Reviews'])]].sum() == row['Fraud found']
for _, row in pd.read_csv(P2 / 'results/review_capacities.csv').iterrows():
    order = np.argsort(-scores[row.model].to_numpy(), kind='stable')
    found = y[order[:int(row.reviews)]].sum()
    assert found == row.frauds_found
    assert np.isclose(row.precision, found/row.reviews)
    assert np.isclose(row.recall, found/y.sum())

# Both models must receive identical rows and input features.
shared_inputs_checked = (P1/'data/processed/features.npz').exists() and (P2/'models/features.npz').exists()
if shared_inputs_checked:
    first = np.load(P1/'data/processed/features.npz')
    second = np.load(P2/'models/features.npz')
    for name in ['X_train', 'X_validation', 'X_test', 'y_validation', 'y_test']:
        assert np.allclose(first[name], second[name]), name

policies = pd.read_csv(P2 / 'results/test_thresholds.csv')
for _, row in policies.iterrows():
    matrix = confusion_matrix(y, scores[row.model] > row.cutoff, labels=[0, 1]).ravel()
    assert np.array_equal(matrix, [row.tn, row.fp, row.fn, row.tp])
    assert row.alerts == row.tp + row.fp
    assert np.isclose(row.test_recall, row.tp / (row.tp + row.fn))
    assert np.isclose(row.test_precision, row.tp / max(row.tp + row.fp, 1))
    assert np.isclose(row.test_fpr, row.fp / (row.fp + row.tn))
validation = pd.read_csv(P2 / 'results/validation_thresholds.csv')
assert (validation.validation_fpr <= validation.budget).all()

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.refs = []
    def handle_starttag(self, tag, attrs):
        self.refs += [value for key, value in attrs if key in ['href', 'src'] and value]

broken = []
assert (ROOT / 'index.html').exists()
assert not (P1 / 'index.html').exists() and not (P2 / 'index.html').exists()
assert not any((ROOT/name).exists() for name in
               ['notebooks', 'results', 'figures', 'models', 'output', 'report', 'docs', 'data'])
for path in [ROOT/'index.html', *(P1/'output').rglob('*.html'), *(P2/'output').rglob('*.html')]:
    parser = Links()
    parser.feed(path.read_text())
    for ref in parser.refs:
        url = urlsplit(ref)
        if url.scheme or url.netloc or not url.path:
            continue
        if not (path.parent / unquote(url.path)).resolve().exists():
            broken.append((str(path.relative_to(ROOT)), ref))
for path in [ROOT/'README.md', *P1.glob('*.md'), *P2.glob('*.md'),
             *(P1/'docs').glob('*.md'), *(P2/'docs').glob('*.md'),
             *(P1/'notebooks').glob('*.md')]:
    for ref in re.findall(r'\[[^\]]*\]\(([^)]+)\)', path.read_text()):
        url = urlsplit(ref.strip('<>'))
        if url.scheme or url.netloc or not url.path:
            continue
        if not (path.parent/unquote(url.path)).resolve().exists():
            broken.append((str(path.relative_to(ROOT)), ref))
assert not broken, broken

report = PdfReader(P2 / 'output/Phase2_Report.pdf')
assert len(report.pages) == 4
report_text = '\n'.join(page.extract_text() for page in report.pages)
assert '??' not in report_text and '[?]' not in report_text
for enrollment in ['2401010174', '2401010486', '2401010267', '2401010222', '2401010279']:
    assert enrollment in report_text
for _, row in ranking.iterrows():
    assert f'{row.pr_auc_trapezoidal:.4f}' in report_text
    assert f'{row.average_precision:.4f}' in report_text
phase1_report = PdfReader(P1/'output/Phase1_Report.pdf')
phase1_text = '\n'.join(page.extract_text() for page in phase1_report.pages)
assert '[?]' not in phase1_text and '??' not in phase1_text
original = json.loads((P1/'results/presentation_integrity.json').read_text())
assert hashlib.sha256((P1/'output/Phase1_Presentation_Final.pptx').read_bytes()).hexdigest() == original['sha256']
slides_pdf = PdfReader(P2 / 'output/Phase2_Presentation.pdf')
assert len(slides_pdf.pages) == 20
with zipfile.ZipFile(P2 / 'output/Phase2_Presentation.pptx') as archive:
    slides = [name for name in archive.namelist()
              if re.fullmatch(r'ppt/slides/slide\d+.xml', name)]
    notes = [name for name in archive.namelist()
             if re.fullmatch(r'ppt/notesSlides/notesSlide\d+.xml', name)]
    assert len(slides) == 20 and len(notes) == 20
    for name in notes:
        text = ' '.join(item.text or '' for item in ET.fromstring(archive.read(name)).iter()
                        if item.tag.endswith('}t'))
        assert len(text.split()) >= 60, name
    chart_count = len([name for name in archive.namelist()
                       if re.fullmatch(r'ppt/(?:slides/)?charts/chart\d+.xml', name)])
    assert chart_count == 10

validation_receipt = json.loads((P2 / 'results/presentation_validation.json').read_text())
deck_hash = hashlib.sha256((P2 / 'output/Phase2_Presentation.pptx').read_bytes()).hexdigest()
assert deck_hash == validation_receipt['sha256']
assert validation_receipt['package_integrity_passed']
assert validation_receipt['layout_checks_passed']
checks.update({'ranking_metrics_recomputed': True, 'confusion_matrices_recomputed': True,
               'exploratory_draft': {'code_cells': len(draft_cells), 'errors': 0, 'fresh_process': True},
               'validation_budgets_checked': True, 'local_html_links_checked': True,
               'team_identities_checked': True, 'report_pages': 4,
               'presentation_slides': 20, 'presentation_pdf_pages': 20,
               'speaker_note_parts': 20, 'editable_charts': chart_count,
               'presentation_matches_validated_hash': True,
               'phase1_notebooks': phase1_notebooks, 'phase1_pr_metrics_recomputed': True,
               'all_review_capacities_recomputed': True, 'single_root_index_checked': True,
               'phase1_presentation_preserved': True, 'shared_inputs_checked': shared_inputs_checked,
               'shared_inputs_note': 'Compared local arrays' if shared_inputs_checked else
                                    'Arrays excluded from portable package; rerun both phases to check'})
if not args.no_write:
    (P2 / 'results/verification.json').write_text(json.dumps(checks, indent=2))
print('PASS: ten main notebooks, exploratory draft, shared index, results, links, reports and presentations.')

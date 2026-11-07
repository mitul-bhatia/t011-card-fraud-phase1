"""Build the single project homepage from the two phases' measured results."""
from pathlib import Path
import html
import json
import mistune
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
PHASE1 = json.loads((ROOT / 'phase1/results/summary.json').read_text())
PHASE2 = pd.read_csv(ROOT / 'phase2/results/model_comparison.csv').set_index('model')
POLICIES = pd.read_csv(ROOT / 'phase2/results/test_thresholds.csv')
DEFAULT = POLICIES[POLICIES.validation_budget == 0.005].set_index('model')

# Keep Phase 1's reading guide alongside its final documents.
guide = mistune.create_markdown(plugins=['table'])((ROOT/'phase1/STUDY_GUIDE.md').read_text())
for attribute in ['href', 'src']:
    for folder in ['docs', 'notebooks', 'figures']:
        guide = guide.replace(f'{attribute}="{folder}/', f'{attribute}="../{folder}/')
(ROOT/'phase1/output/STUDY_GUIDE.html').write_text(
    '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    '<title>Phase 1 study guide</title><style>body{max-width:950px;margin:40px auto;padding:0 24px;'
    'font:17px/1.65 Arial,sans-serif;color:#173447;background:#fafbf9}a{color:#24668d}'
    'td,th{padding:10px;border-bottom:1px solid #d5dfe4}pre{overflow:auto}img{max-width:100%}</style>'
    '<body><p><a href="../../index.html#phase1">Project home</a></p>'+guide+'</body></html>')

NOTEBOOKS = {
    'phase1': [
        ('01_Project_and_Literature', 'Project and literature', 'The review problem, sources and experiment plan.'),
        ('02_Data_Quality_and_EDA', 'Data quality and EDA', 'Check the CSV, remove exact repeats and split by time.'),
        ('03_Feature_Engineering', 'Features and scaling', 'Build the same 29 inputs from legitimate training rows.'),
        ('04_GMM_Training', 'GMM training', 'Understand EM and choose components using AIC and BIC.'),
        ('05_Evaluation_and_Thresholds', 'Evaluation and thresholds', 'Read PR curves, false alerts and review-capacity results.'),
    ],
    'phase2': [
        ('00_Start_Here', 'Start here', 'The DL question, fair-comparison rules and retained team roles.'),
        ('01_Data_and_Split', 'Data and split', 'Verify the shared dataset and the chronological periods.'),
        ('02_Features_and_Scaling', 'Features and scaling', 'Give GMM and AE the same legitimate training inputs.'),
        ('03_Autoencoder_Training', 'Autoencoder training', 'Learn compression, clean targets, noise and the training loop.'),
        ('04_Hybrid_and_Evaluation', 'Hybrid and evaluation', 'Compare scores, thresholds, review workload and ablations.'),
    ]}

def chapters(phase):
    result = []
    for filename, title, description in NOTEBOOKS[phase]:
        number = filename.split('_')[0]
        result.append(f'''<li><span class="chapter-number">{number}</span><div>
          <a class="chapter-title" href="{phase}/output/notebooks/{filename}.html">{title}</a>
          <p>{description}</p></div><a class="source" href="{phase}/notebooks/{filename}.ipynb">Code</a></li>''')
    return '<ol class="chapters">'+''.join(result)+'</ol>'

def figure(path, description):
    width, height = Image.open(ROOT / path).size
    return f'<figure><img src="{path}" width="{width}" height="{height}" loading="lazy" alt="{html.escape(description)}"><figcaption>{description}</figcaption></figure>'

rows = []
for key, title in [('GMM', 'GMM'), ('denoising_ae', 'Denoising AE'), ('Hybrid', 'Hybrid')]:
    metric, policy = PHASE2.loc[key], DEFAULT.loc[key]
    rows.append(f'<tr><th scope="row">{title}</th><td>{metric.pr_auc_trapezoidal:.4f}</td>'
                f'<td>{int(metric.frauds_at_100)}</td><td>{int(policy.tp)}</td><td>{int(policy.fp)}</td></tr>')

page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Team T011 Card Fraud Triage: Phase 1 GMM and Phase 2 autoencoder and hybrid. Read notebooks, measured results, reports and presentations.">
<meta name="theme-color" content="#fafbf9">
<title>Card Fraud Triage · Phase 1 and Phase 2</title>
<style>
:root{{color-scheme:light;--paper:#fafbf9;--ink:#173447;--blue:#24668d;--muted:#4e6472;--line:#d5dfe4;--wash:#edf3f6}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth;scroll-padding-top:24px}}
body{{margin:0;background:var(--paper);color:var(--ink);font:17px/1.65 Arial,sans-serif}}
::selection{{background:#cce3ef;color:var(--ink)}}a{{color:var(--blue);text-underline-offset:.2em}}a:hover{{color:#153e59}}
a:focus-visible,summary:focus-visible,[tabindex]:focus-visible{{outline:3px solid var(--blue);outline-offset:5px}}
.skip{{position:absolute;top:-80px;left:16px;padding:10px 16px;background:white;z-index:2}}.skip:focus{{top:12px}}
.shell{{max-width:1140px;margin:auto;padding:0 36px}}header{{padding:54px 0 30px}}
h1{{margin:0;font-size:clamp(34px,5vw,52px);line-height:1.12;letter-spacing:-.025em}}
.team-line{{color:var(--muted);margin:16px 0}}.intro{{max-width:70ch;margin:0}}
.phase-nav{{display:flex;flex-wrap:wrap;gap:12px 32px;padding:24px 0 28px;border-bottom:1px solid var(--line)}}
.phase-nav a{{padding:7px 0;font-weight:700}}section{{padding:42px 0;border-bottom:1px solid var(--line)}}
h2{{font-size:30px;line-height:1.25;letter-spacing:-.015em;margin:0 0 12px}}
h3{{font-size:22px;line-height:1.35;margin:28px 0 14px}}.phase-intro{{max-width:72ch;color:var(--muted);margin:0 0 28px}}
.phase-layout{{display:grid;grid-template-columns:minmax(0,1.7fr) minmax(240px,1fr);gap:44px}}
.phase-layout h3{{margin-top:0}}.chapters{{list-style:none;margin:0;padding:0}}
.chapters li{{display:grid;grid-template-columns:28px minmax(0,1fr) 48px;gap:14px;padding:17px 0;border-top:1px solid var(--line)}}
.chapter-number{{color:var(--muted);font-variant-numeric:tabular-nums}}.chapter-title{{font-weight:700;font-size:19px}}
.chapters p{{font-size:15px;line-height:1.5;margin:5px 0 0;color:var(--muted)}}.source{{font-size:14px;padding:3px 0}}
.resources{{list-style:none;margin:0;padding:0}}.resources li{{padding:11px 0;border-top:1px solid var(--line)}}
.resources small{{display:block;color:var(--muted);font-size:14px}}.result-note{{max-width:73ch;margin:16px 0}}
.table-wrap{{overflow:auto;max-width:100%}}table{{border-collapse:collapse;width:100%;font-size:16px;font-variant-numeric:tabular-nums}}
td,th{{border-bottom:1px solid var(--line);padding:12px 14px;text-align:left}}thead th{{background:var(--wash);font-size:15px}}
tbody th{{font-weight:400}}.table-wrap table{{min-width:580px}}figure{{margin:26px 0 0;max-width:100%}}
img{{max-width:100%;height:auto;display:block}}figcaption{{font-size:14px;line-height:1.5;color:var(--muted);margin-top:10px;max-width:75ch}}
.team-table{{min-width:600px}}footer{{padding:26px 0 40px;color:var(--muted);font-size:15px}}footer p{{max-width:75ch}}
.footer-links{{display:flex;flex-wrap:wrap;gap:12px 25px}}
@media(max-width:740px){{.shell{{padding:0 20px}}header{{padding-top:32px}}.phase-layout{{grid-template-columns:1fr;gap:30px}}
h2{{font-size:27px}}section{{padding:32px 0}}.phase-nav{{gap:8px 22px}}.chapters li{{gap:10px;grid-template-columns:25px minmax(0,1fr) 40px}}}}
@media(prefers-reduced-motion:reduce){{html{{scroll-behavior:auto}}}}
@media print{{.skip,.phase-nav{{display:none}}.shell{{max-width:none;padding:0}}section{{break-before:auto}}figure,.chapters li{{break-inside:avoid}}}}
</style></head><body><a class="skip" href="#phase1">Skip to the phases</a><main class="shell">
<header><h1>Card Fraud Triage</h1><p class="team-line">Team T011 · Section A · Project 26</p>
<p class="intro">Study the project in two parts: Gaussian mixture density scoring in Phase 1, then autoencoder reconstruction and the hybrid comparison in Phase 2. Read the saved notebooks and results without running Python.</p></header>
<nav class="phase-nav" aria-label="Project navigation"><a href="#phase1">Phase 1 · GMM</a><a href="#phase2">Phase 2 · Deep learning</a><a href="#team">Team roles</a><a href="phase2/output/Card_Fraud_Triage_Complete.zip">Download both phases</a></nav>
<section id="phase1" aria-labelledby="phase1-title"><h2 id="phase1-title">Phase 1 · Machine learning</h2>
<p class="phase-intro">Learn normal transaction density with a Gaussian Mixture Model. Read notebook 01, then run 02–05 in order.</p>
<div class="phase-layout"><div><h3>Notebooks</h3>{chapters('phase1')}</div><aside aria-label="Phase 1 deliverables"><h3>Report, slides and study</h3>
<ul class="resources"><li><a href="phase1/output/Phase1_Report.pdf">IEEE report</a><small>Complete Phase 1 method and results</small></li>
<li><a href="phase1/output/Phase1_Presentation_Final.pptx">Phase 1 presentation</a><small>The original team presentation</small></li>
<li><a href="phase1/output/STUDY_GUIDE.html">Phase 1 study guide</a></li><li><a href="#phase1-results">Measured results</a></li>
<li><a href="phase1/results/threshold_results.csv">Threshold table (CSV)</a></li><li><a href="phase1/results/review_capacities.csv">Review capacities (CSV)</a></li>
<li><a href="phase1/README.md">Phase 1 file guide</a></li></ul></aside></div>
<div id="phase1-results"><h3>Phase 1 results</h3><div class="table-wrap" tabindex="0" role="region" aria-label="Phase 1 measured results">
<table><thead><tr><th scope="col">Model</th><th scope="col">PR area</th><th scope="col">Frauds / 100 reviews</th><th scope="col">Frauds at default</th><th scope="col">False alerts at default</th></tr></thead>
<tbody><tr><th scope="row">GMM, K = 8</th><td>{PHASE1['PR_AUC']:.4f}</td><td>{round(100*PHASE1['precision_at_100'])}</td><td>{PHASE1['TP']} / 74</td><td>{PHASE1['FP']}</td></tr></tbody></table></div>
<p class="result-note">The one-Gaussian reference has PR area {PHASE1['one_gaussian_PR_AUC']:.4f}. The default cutoff uses a 0.5% validation false-positive budget. <a href="phase1/results/summary.json">Saved metric summary</a>.</p>
{figure('phase1/figures/09_precision_recall.png','Phase 1 test precision-recall curves compare the selected GMM with a one-Gaussian reference.')}</div></section>
<section id="phase2" aria-labelledby="phase2-title"><h2 id="phase2-title">Phase 2 · Deep learning and hybrid</h2>
<p class="phase-intro">Train an autoencoder on legitimate transactions, then compare reconstruction error with GMM density scores and their normalized combination. Read 00, then run 01–04.</p>
<div class="phase-layout"><div><h3>Notebooks</h3>{chapters('phase2')}</div><aside aria-label="Phase 2 deliverables"><h3>Report, slides and study</h3>
<ul class="resources"><li><a href="phase2/output/Phase2_Report.pdf">IEEE report</a><small>DL method, hybrid logic, ablations and limitations</small></li>
<li><a href="phase2/output/Phase2_Presentation.pptx">Phase 2 presentation</a><small>20 slides with editable charts and speaking notes</small></li>
<li><a href="phase2/output/Phase2_Presentation.pdf">Read the slides as PDF</a></li><li><a href="phase2/output/LAST_DAY_STUDY_GUIDE.html">Last-day study and viva guide</a></li>
<li><a href="phase2/output/PRESENTATION_SCRIPT.html">Simple speaking script</a></li><li><a href="phase2/output/RESULTS_EXPLAINED.html">Results explained simply</a></li>
<li><a href="#phase2-results">Measured results</a></li><li><a href="phase2/results/test_thresholds.csv">Threshold table (CSV)</a></li>
<li><a href="phase2/results/ablation_results.csv">Ablations (CSV)</a></li><li><a href="phase2/START_HERE.md">Phase 2 file guide</a></li></ul></aside></div>
<div id="phase2-results"><h3>Phase 2 results</h3><div class="table-wrap" tabindex="0" role="region" aria-label="Phase 2 measured results">
<table><thead><tr><th scope="col">Model</th><th scope="col">PR area</th><th scope="col">Frauds / 100 reviews</th><th scope="col">Frauds at default</th><th scope="col">False alerts at default</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="result-note">GMM ranks best here. At the default validation policy, hybrid makes 40 fewer false alerts than GMM while catching two fewer frauds. The test benchmark was already used in Phase 1. <a href="phase2/results/summary.json">Saved metric summary</a>.</p>
{figure('phase2/figures/06_main_results.png','All three methods use the same later test period. The hybrid improves AE ranking and changes the false-alert trade-off, while GMM retains the strongest overall ranking.')}</div></section>
<section id="team" aria-labelledby="team-title"><h2 id="team-title">The same team and roles</h2><div class="table-wrap" tabindex="0" role="region" aria-label="Team responsibilities">
<table class="team-table"><thead><tr><th scope="col">Member</th><th scope="col">Enrollment</th><th scope="col">Responsibility</th></tr></thead><tbody>
<tr><th scope="row">H Maheshwari (Hardik)</th><td>2401010174</td><td>Evaluation and thresholds</td></tr>
<tr><th scope="row">Vaibhav</th><td>2401010486</td><td>Features and scaling</td></tr>
<tr><th scope="row">Mayank</th><td>2401010267</td><td>Model training; README and slides with Keshav</td></tr>
<tr><th scope="row">Keshav</th><td>2401010222</td><td>Data quality and EDA; README and slides with Mayank</td></tr>
<tr><th scope="row">Mitul Bhatia</th><td>2401010279</td><td>Overall project lead and coordination</td></tr></tbody></table></div>
<p class="result-note">All five jointly contribute to the report and should understand the complete pipeline. <a href="phase2/docs/TEAM_ROLES.md">Review the contribution record</a>.</p></section>
<footer><div class="footer-links"><a href="README.md">Run and folder instructions</a><a href="CSAI_Project_Instructions_Phase_1_and_Phase_2.pdf">Faculty instructions</a><a href="phase2/docs/PHASE2_CHECKLIST.md">Phase 2 checklist</a></div>
<p>The official CSV is stored once in <code>phase1/data/raw/</code> and reused by both phases. An anomaly score prioritizes review; it is not proof of fraud.</p></footer>
</main></body></html>'''
(ROOT / 'index.html').write_text(page)
print('Built the single root index for both phases, notebooks, results and deliverables.')

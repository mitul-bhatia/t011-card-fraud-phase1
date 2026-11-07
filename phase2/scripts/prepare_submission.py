"""Build readable summaries and the IEEE source from measured result files."""
from pathlib import Path
import json, html, os
import pandas as pd
import mistune

P2 = Path(__file__).resolve().parents[1]
ROOT = P2.parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/phase2/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
ranking = pd.read_csv(P2 / 'results/model_comparison.csv').set_index('model')
policies = pd.read_csv(P2 / 'results/test_thresholds.csv')
default = policies[policies.validation_budget == .005].set_index('model')
settings = json.loads((P2 / 'results/training_settings.json').read_text())
summary = json.loads((P2 / 'results/summary.json').read_text())
complement = json.loads((P2 / 'results/complementarity.json').read_text())
names = {'GMM':'GMM', 'PCA8':'PCA (8 components)', 'plain_ae':'Plain AE',
         'denoising_ae':'Denoising AE', 'wide_ae':'Denoising AE, bottleneck 12',
         'seed43_ae':'Denoising AE, seed 43', 'Hybrid':'Hybrid'}

def metric_table(models):
    lines = ['| Model | PR AUC (trapezoidal) | AP | Frauds in 100 reviews |',
             '|---|---:|---:|---:|']
    for n in models:
        r = ranking.loc[n]
        lines.append(f'| {names[n]} | {r.pr_auc_trapezoidal:.4f} | {r.average_precision:.4f} | {int(r.frauds_at_100)} |')
    return '\n'.join(lines)

main = ['GMM', 'denoising_ae', 'Hybrid']
results = f'''# Results explained simply

These are measured results from the real dataset. The GMM is the frozen Phase 1 baseline. The same later test period has **56,746 transactions and 74 frauds**, a fraud base rate of **{summary['test_base_rate']:.4%}**. The test benchmark was already exposed in Phase 1; Phase 2 settings were fixed without using its labels for tuning.

## Which method gives the best ranking?

{metric_table(main)}

The GMM gives the strongest ranking here. The hybrid improves the AE result but does not beat the GMM on PR AUC, AP or precision at 100. All three rankings find substantially more fraud in the first 100 reviews than the approximately **0.13 frauds expected from 100 random reviews**. This is a dataset comparison, not a deployment guarantee.

## What happens at the default 0.5% validation FPR policy?

| Model | Fraud found / 74 | False alerts | Total reviews | Recall | Precision | Actual test FPR |
|---|---:|---:|---:|---:|---:|---:|'''
for n in main:
    r = default.loc[n]
    results += f'\n| {names[n]} | {int(r.tp)} | {int(r.fp)} | {int(r.alerts)} | {r.test_recall:.2%} | {r.test_precision:.2%} | {r.test_fpr:.4%} |'
results += f'''

Compared with the GMM at this predeclared validation policy, the hybrid makes **40 fewer false alerts (23.3% fewer)** and catches **2 fewer frauds**. Its higher precision comes with lower recall. These are the same *validation target*, not identical realized test FPRs or identical review counts. The top-100 comparison above is the equal-review-count comparison.

Compared with the AE, the hybrid catches **2 more frauds** and creates **5 fewer false alerts** at this policy. That is evidence of a local benefit over the DL branch, while the GMM remains stronger on the overall ranking measures.

At the relaxed 1% validation policy, all three find 52 frauds. GMM makes 381 false alerts, AE 314, hybrid 304. At the strict 0.1% policy, GMM finds 6 frauds while AE and hybrid find 0. These differences show why one headline metric is insufficient.

## What did the ablations teach us?

{metric_table(list(ranking.index))}

- **Noise:** Denoising modestly improves PR AUC from 0.1281 to 0.1308 and finds 24 rather than 22 frauds in the top 100. At the default policy both catch 33, and the plain AE makes fewer false alerts (127 vs 137). Noise does not improve every measure.
- **Compression:** The 12-unit bottleneck lowers normal-validation MSE to 0.6599 from 0.7130, yet reduces test PR AUC to 0.1044. Better copying of normal transactions does not guarantee better fraud ranking.
- **Randomness:** Seed 43 reduces the denoising AE's PR AUC to 0.0915 and top-100 frauds to 12. The model is sensitive to initialization and batch order. We retain the declared seed 42 main result and disclose the weaker run.
- **Linear reconstruction:** PCA8 reaches PR AUC 0.0712 and finds 6 frauds in 100 reviews. The tested nonlinear AE is better here, although the fitting methods and parameterizations differ.
- **Hybrid branches:** Removing the AE leaves the GMM, which is the best overall ranking model. Removing the GMM leaves the weaker AE. This combination has a measured trade-off, not universal superiority.

## Why does the hybrid fail to beat the GMM at 100 reviews?

The GMM and AE score ranks have Spearman correlation **0.746**. All **24** frauds found by the AE in its top 100 are already among the GMM's **29** detected frauds at that capacity. The AE contributes **0 unique frauds** there, while the GMM contributes **5 unique frauds**. Equal averaging changes the order and loses some of the GMM's advantage. This overlap is specific to the top-100 policy; it does not prove there are no complementary cases at other thresholds.

## How certain is the comparison?

A 200-replicate paired, stratified row bootstrap gives a 95% percentile interval of **[{summary['hybrid_minus_ae_ap_paired_bootstrap_95pct'][0]:.4f}, {summary['hybrid_minus_ae_ap_paired_bootstrap_95pct'][1]:.4f}]** for hybrid AP minus main-AE AP. This describes sampling variation conditional on these fitted models and the observed class counts. It ignores temporal dependence, training variation and benchmark reuse. The large seed effect is a reason to avoid a broad superiority claim.

## Training actually observed

Both the plain and main denoising AE reach the 30-epoch cap and restore epoch 30. The wide model stops at epoch 15 and restores epoch 10. Seed 43 reaches epoch 30 and restores epoch 28. We do not claim early stopping fired on every run.

## A defensible viva conclusion

“We implemented and evaluated both density and nonlinear reconstruction scoring. The GMM ranks fraud best on this reused benchmark. The hybrid improves the AE and can reduce false alerts relative to the GMM at declared validation policies, with a recall cost at the default policy. Our ablations show that better reconstruction and a larger bottleneck do not automatically produce better anomaly detection.”
'''
(P2 / 'docs/RESULTS_EXPLAINED.md').write_text(results)

readme = f'''# Phase 2: Card Fraud Triage

**Team T011 · Section A · Project 26**

This folder contains the deep learning extension and an implemented GMM–autoencoder hybrid. Phase 1 lives in the neighbouring `phase1/` folder. Both phases use the single root `index.html`.

For the shortest file guide, open [START_HERE.md](START_HERE.md).

## Start here

- **Study without running Python:** open [the project index](../index.html#phase2). It links all five executed notebooks, the study guide, results and deliverables.
- **Revise for the viva:** [last-day study guide](docs/LAST_DAY_STUDY_GUIDE.md), then [results explained](docs/RESULTS_EXPLAINED.md) and [simple presentation script](docs/PRESENTATION_SCRIPT.md).
- **Present and submit after team review:** [IEEE report](output/Phase2_Report.pdf), [20-slide PowerPoint](output/Phase2_Presentation.pptx), [presentation PDF](output/Phase2_Presentation.pdf), [report LaTeX](report/Phase2_Report.tex), and the notebooks.
- **Download everything together:** [complete portable package](output/Card_Fraud_Triage_Complete.zip). It includes both phases and the shared root index. Download the official CSV separately on a new machine.
- **Check requirements and roles:** [faculty checklist](docs/PHASE2_CHECKLIST.md), [retained roles](docs/TEAM_ROLES.md), [literature](docs/LITERATURE_REVIEW.md).

## Measured result

{metric_table(main)}

At the default validation FPR policy, GMM finds 37 frauds with 172 false alerts; AE finds 33 with 137; hybrid finds 35 with 132. The hybrid improves the AE, but does not beat the GMM on ranking or top-100 reviews. Its 40 fewer false alerts than the GMM come with 2 fewer detected frauds. The test period is reused from Phase 1 and is not newly blind.

## Notebooks, in order

| Notebook | Lead | One question |
|---|---|---|
| 00 Start Here | Shared | What are we building and why? |
| 01 Data and Split | Keshav | Which rows teach and which rows evaluate? |
| 02 Features and Scaling | Vaibhav | What 29 numbers enter both models? |
| 03 Autoencoder Training | Mayank | How does the AE learn, and what do ablations change? |
| 04 Hybrid and Evaluation | Hardik | Does the combination help at a fixed review policy? |

Mitul retains project leadership, integration and coordination. All five jointly contribute to the report. Assignments are not a record of completed individual work; fill the actual-work table before claiming contributions.

## Run from the repository root

Use Python 3.12. Keep the official CSV at `phase1/data/raw/creditcard.csv`. It already exists in this workspace; on a fresh checkout, download it from the [dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). No Kaggle credentials are embedded.

```bash
python3.12 -m venv .venv-phase2
source .venv-phase2/bin/activate
python -m pip install -r phase2/requirements.txt
python phase2/scripts/run_all.py
```

Alternatively, open the notebooks in VS Code or Jupyter using this environment. Read 00, then restart each kernel and Run All for 01 → 04. The terminal runner uses a fresh Python process for each notebook, captures its outputs, and writes self-contained HTML reading copies. It stops on a cell error. It does not need a notebook server.

Training used CPU, seed 42 for the main model and 43 for the sensitivity run. Four small AE runs took about 33 seconds of training in this workspace; full execution and GMM rebuilding take additional time and depend on hardware. The tested environment is saved in `results/run_environment.json`; `requirements-lock.txt` records installed versions. The ipython<9 bound keeps the pinned matplotlib version compatible.

## Files and reproducibility

- `notebooks/`: full explanations, visible training code and saved execution outputs.
- `results/`: measured tables, policies, model settings, histories, uncertainty and audit checks.
- `figures/`: ten notebook figures plus two plots sized for the report.
- `models/`: locally generated split CSVs, scaled arrays, scaler and model weights, excluded from Git.
- `report/`: IEEE LaTeX source and a copy of the existing IEEE class.
- `output/`: final report, presentation and notebook/study reading copies.
- `scripts/run_all.py`: execute all notebooks. `prepare_submission.py`: regenerate summaries and report source from saved measurements.
- `scripts/check_deliverables.py`: verify the saved metrics, notebook outputs, links, report, slides and team identifiers. `package_project.py`: rebuild the portable archive.

Notebook 04 uses the frozen local Phase 1 GMM where available. A fresh checkout without the excluded binary rebuilds the same K grid using the exact declared recipe. It checks that the GMM PR AUC matches Phase 1. All models fit the same legitimate training rows, and all preprocessing is fitted on those rows only. Validation legitimate rows set stopping, normalization and FPR cutoffs. Test labels are used for final reporting and diagnostics, not Phase 2 tuning.

The report and figures describe this historical two-day dataset. A wider bottleneck and a second seed perform worse. These runs remain visible. No model score is a calibrated fraud probability.

A fresh-copy check also rebuilt the GMM from the original K grid, rather than loading the local binary, and reproduced every model's saved ranking metrics. Details are in `results/reproducibility_check.json`.

## Final team review

Open every final artifact, complete `docs/TEAM_ROLES.md`, rehearse the individual viva and check current faculty Phase 2 submission instructions. The provided PDF gives no Phase 2 date or form. Publishing this repository does not submit coursework through a faculty form.
'''
(P2 / 'README.md').write_text(readme)

latex_rows = []
for n in ranking.index:
    r = ranking.loc[n]
    short = {'PCA8':'PCA8','wide_ae':'AE bottleneck 12','seed43_ae':'AE seed 43'}.get(n, names[n])
    latex_rows.append(f'{short} & {r.pr_auc_trapezoidal:.4f} & {r.average_precision:.4f} & {int(r.frauds_at_100)} \\\\')
policy_rows = []
for budget in [.001,.005,.01]:
    for n in main:
        r = policies[(policies.model == n) & (policies.validation_budget == budget)].iloc[0]
        policy_rows.append(f'{100*budget:.1f} & {names[n]} & {int(r.tp)} & {int(r.fp)} & {100*r.test_recall:.1f} & {100*r.test_precision:.1f} \\\\')

tex = r'''\documentclass[conference]{IEEEtran}
\usepackage{amsmath,graphicx,url,array,booktabs}
\usepackage[hidelinks]{hyperref}
\hypersetup{pdftitle={Card Fraud Triage - Phase 2},pdfauthor={Team T011}}
\graphicspath{{../figures/}}
\title{Card Fraud Triage via Gaussian Mixture Density Scoring and Autoencoder Anomaly Detection\\Phase 2: Deep Learning and Score Fusion}
\author{\IEEEauthorblockN{H Maheshwari, Vaibhav, Mayank, Keshav, Mitul Bhatia}
\IEEEauthorblockA{Team T011, Section A, Project 26\\Newton School of Technology, Rishihood University\\
2401010174, 2401010486, 2401010267, 2401010222, 2401010279}}
\begin{document}
\maketitle
\begin{abstract}
We extend our Gaussian Mixture Model (GMM) fraud-triage baseline with a small denoising autoencoder (AE). Both models learn from the same 169,893 earlier legitimate transactions and score the same later mixed test period. We also implement fixed equal-weight fusion of normalized density and reconstruction scores. On 56,746 test transactions containing 74 frauds, trapezoidal precision-recall area is 0.1496 for GMM, 0.1308 for AE and 0.1412 for the hybrid. The top 100 reviews find 29, 24 and 25 frauds respectively. At the default 0.5\% validation false-positive policy, the hybrid finds 35 frauds with 132 false alerts, compared with the GMM's 37 with 172. Thus fusion improves the AE and changes the false-alert trade-off, but does not beat the GMM's overall ranking. Noise, bottleneck size, a second random seed and linear reconstruction ablations expose important limitations. Phase 1 already exposed the test benchmark; this is a consistent reused-benchmark comparison, not a new blind test.
\end{abstract}
\begin{IEEEkeywords}fraud triage, anomaly detection, autoencoder, Gaussian mixture, score fusion\end{IEEEkeywords}

\section{Problem and Phase 2 Objective}
Fraud triage orders transactions for human review. An anomaly is a reason to investigate rather than proof of fraud. Rare positives make accuracy an inadequate headline: always predicting legitimate would achieve about 99.87\% test accuracy while finding no fraud. We instead measure precision-recall (PR) ranking, recall at validation false-positive rate (FPR) budgets, and precision at a fixed review capacity.

Phase 1 modeled normal transaction density with a GMM. Phase 2 asks whether nonlinear reconstruction captures useful information missed by that density model. Our hybrid is an implemented score-fusion rule, not merely a side-by-side comparison. Its value must be demonstrated by ablation and measured trade-offs, as required by the supplied course instructions. We do not claim a new algorithm or guaranteed additional marks.

\section{Related Work and Motivation}
Dempster, Laird and Rubin~\cite{em} provide the EM framework underlying mixture estimation. Our frozen baseline retains Phase 1's diagonal-covariance GMM and AIC/BIC component study. Better legitimate density fit does not itself establish better fraud ranking.

The assignment recommends Sakurada and Yairi~\cite{sakurada} for AE anomaly detection using nonlinear dimensionality reduction. We use this as motivation for the reconstruction comparison; we do not reproduce its experiments or import its performance claims. Publisher full text was unavailable during preparation, so no detailed experimental assertion is based on that paper.

Vincent et al.~\cite{vincent} study learning representations by recovering clean inputs from corrupted versions. We adapt this clean-target denoising idea to a small jointly trained tabular AE. We do not reproduce the full stacked pretraining procedure or assume its classification findings transfer to fraud anomaly ranking. Our plain-versus-denoising experiment checks the local intervention.

Saito and Rehmsmeier~\cite{pr} motivate examining precision and recall when positives are rare. We also retain operational counts and FPR budgets, since a small FPR can still create many legitimate alerts. Software documentation~\cite{sk,torch} supports the metric and loss implementation. In particular, average precision (AP) and trapezoidal PR area use different integration conventions.

\section{Data and Fair Comparison}
\subsection{Quality, Splits and Base Rates}
The ULB/Worldline dataset~\cite{data} contains 284,807 transactions and 492 frauds, spanning two historical days. We verify the exact CSV SHA-256 against Phase 1 and find no missing or nonfinite values. Following Phase 1, stable time sorting and removal of 1,081 exact repeated rows leave 283,726 rows and 473 frauds. Without transaction IDs we cannot prove every repeated row is an accidental duplicate; removal is a declared preprocessing assumption.

The chronological 60/20/20 split contains 170,235 training rows (342 frauds), 56,745 validation rows (57 frauds), and 56,746 test rows (74 frauds). Corresponding fraud rates are 0.2009\%, 0.1004\%, and 0.1304\%. Only the 169,893 legitimate training rows fit the GMM, AE and linear reference. The label is used to identify known legitimate rows, so this is commonly termed one-class or semi-supervised anomaly detection; the reconstruction objective itself uses no fraud targets.

\subsection{Shared Inputs and Leakage Controls}
Both main models receive V1--V28 and $\log(1+\mathrm{Amount})$, totaling 29 features. Time orders the split but is excluded from the inputs. Class is never an input. StandardScaler fits means and standard deviations on legitimate training rows only; validation and test reuse them unchanged. We check the scaled arrays and row order against the existing Phase 1 arrays. We apply no clipping. Provider anonymization and PCA preparation cannot be independently audited from this CSV.

Legitimate validation rows choose AE checkpoints, score normalization and FPR cutoffs. They never update model weights. No Phase 2 model or hybrid-weight choice uses test labels. Nevertheless, Phase 1 results on this exact test period were already visible, so it cannot serve as a newly blind holdout. All test conclusions are descriptive results on that reused chronological benchmark.

\section{Methods}
\subsection{Frozen GMM Baseline}
For $K$ components, the normal density is
\begin{equation}
p(x)=\sum_{k=1}^{K}\pi_k\mathcal{N}(x\mid\mu_k,\Sigma_k),\qquad s_G(x)=-\log p(x).
\end{equation}
EM alternates membership estimates and parameter updates. Phase 1 fits $K\in\{1,2,4,8\}$ using three starts, seed 42, at most 200 iterations, and covariance regularization $10^{-6}$. All candidates converge. K=8 minimizes training BIC (10,144,915) in this grid. Both AIC and BIC continue falling at its edge; eight is not established as globally optimal. Diagonal covariance reduces parameters but ignores within-component cross-feature covariance. This frozen baseline reproduces Phase 1 PR area to numerical tolerance.

\subsection{Undercomplete Denoising Autoencoder}
Our fully connected network has widths $29,16,8,16,29$ and 1,253 trainable parameters. The encoder compresses to eight units and the decoder reconstructs 29 features. ReLU follows each hidden Linear layer, including the bottleneck. A linear output permits negative standardized targets. The narrow bottleneck discourages direct copying while hidden nonlinearities allow relationships beyond linear reconstruction.

For a clean legitimate input $x$ and corrupted input $\widetilde{x}=x+\epsilon$, with independent $\epsilon_j\sim\mathcal{N}(0,0.1^2)$, the reconstruction objective is
\begin{equation}
\mathcal{L}(\theta)=\frac{1}{29N}\sum_{i=1}^{N}\sum_{j=1}^{29}\left[f_\theta(\widetilde{x}_i)_j-x_{ij}\right]^2.
\end{equation}
At evaluation, inputs are clean and the anomaly score is
\begin{equation}
s_A(x)=\frac{1}{29}\sum_{j=1}^{29}(f_\theta(x)_j-x_j)^2.
\end{equation}
For example, reconstructing $[1,2]$ as $[1.1,1.8]$ yields MSE 0.025. High error suggests reconstruction difficulty, not a fraud probability. Fraud can reconstruct well and rare legitimate activity can reconstruct poorly.

\subsection{Training and Regularization}
Adam uses learning rate 0.001, weight decay $10^{-5}$ and batches of 1,024. Each update corrupts the input, performs a forward pass, computes clean-target MSE, backpropagates gradients and updates weights after clearing old gradients. An epoch is one pass over training rows. Training uses CPU with two threads, deterministic operations and seed 42.

We evaluate clean training and legitimate-validation MSE after each epoch. The best accepted improvement exceeds $10^{-4}$. We restore that checkpoint and stop after five epochs without improvement or at 30 epochs. Compression, noise, weight decay and stopping constrain training. The main architecture and settings are fixed in advance rather than selected for test fraud ranking. Both plain and main denoising AE reach epoch 30 and restore epoch 30. The wider model stops after 15 epochs and restores epoch 10; seed 43 restores epoch 28 after running 30.

\begin{figure}[!t]\centering
\includegraphics[width=\columnwidth]{report_learning.png}
\caption{Main denoising AE on clean inputs. It reaches the 30-epoch cap. Later validation loss can reflect distribution change as well as fit quality.}\label{fig:learning}
\end{figure}

\subsection{Technically Justified Score Fusion}
Density rarity and reconstruction difficulty are distinct anomaly questions. Their raw score units differ. Define $t_G=\operatorname{asinh}(s_G)$ and $t_A=\log(1+s_A)$. For each branch, legitimate validation scores supply median $m$ and interquartile range $q$, with a numerical floor of $10^{-8}$:
\begin{equation}
z_G=(t_G-m_G)/q_G,\quad z_A=(t_A-m_A)/q_A,
\end{equation}
\begin{equation}
s_H=0.5z_G+0.5z_A.
\end{equation}
The transformations compress extreme magnitudes; median/IQR reduces outlier sensitivity. Asinh also accepts signed GMM scores. Individual monotone transformations preserve branch rankings, while fusion can change the ordering. Equal weights are fixed without a fraud-label search. The rationale is complementary evidence, not added complexity alone. Its limitations include correlated errors and loss of a strong branch's ranking advantage. This is score-level fusion; we do not fit a GMM on the learned bottleneck.

\section{Evaluation Protocol}
For each model and $b\in\{0.001,0.005,0.01\}$, the cutoff is the normal-validation $(1-b)$ quantile using the higher observed value. The alert rule is strictly $s>\tau$. Assertions verify validation FPR at or below each budget. The default policy is $b=0.005$. Future FPR is not guaranteed by the observed validation rate. Model-specific normalization and thresholding use the same validation period, so their empirical rate is a calibration-set property, not a finite-sample future guarantee.

We report PR area using trapezoidal integration for continuity with Phase 1, and AP using recall-increment weighting. Precision is TP/(TP+FP), recall is TP/(TP+FN), and FPR is FP/(FP+TN). Precision at 100 reviews sorts test scores decreasingly, with stable ties, and counts fraud among those 100. Review capacities 50,100,200,500 refer to the whole test batch, not reviews per day. Test labels report outcomes and diagnostics only.

\section{Results and Trade-offs}
\begin{table}[!t]\centering\caption{Test ranking and ablations; top-100 column is fraud count.}\label{tab:ranking}
\begin{tabular}{lrrr}\toprule
Model & PR area & AP & Top 100\\\midrule
@@RANKING@@
\bottomrule\end{tabular}\end{table}

Table~\ref{tab:ranking} shows the GMM's stronger overall ranking. Denoising AE reaches PR area 0.1308 versus GMM 0.1496. Fusion raises the AE to 0.1412 but remains below GMM. The same ordering holds for AP and precision at 100: 29\%, 24\% and 25\% for GMM, AE and hybrid. For context, 100 random reviews would contain about 0.13 frauds in expectation at this test base rate.

\begin{figure}[!t]\centering
\includegraphics[width=\columnwidth]{report_pr.png}
\caption{Measured test PR curves. Equal-capacity fraud counts are in Table~\ref{tab:ranking}.}\label{fig:results}
\end{figure}

\begin{table}[!t]\centering\caption{Cutoffs set by validation FPR target, applied unchanged to test. Recall and precision are percentages.}\label{tab:policy}
\begin{tabular}{rlrrrr}\toprule
Budget\% & Model & TP & FP & Recall & Prec.\\\midrule
@@POLICIES@@
\bottomrule\end{tabular}\end{table}

At the default policy, GMM produces 209 reviews: TP=37, FP=172, FN=37, TN=56,500. AE produces 170: TP=33, FP=137, FN=41, TN=56,535. Hybrid produces 167: TP=35, FP=132, FN=39, TN=56,540. Corresponding recall is 50.00\%, 44.59\%, 47.30\%; precision is 17.70\%, 19.41\%, 20.96\%; and test FPR is 0.3035\%, 0.2417\%, 0.2329\%.

Thus hybrid has 40 fewer legitimate alerts (23.3\% fewer) than GMM but detects two fewer frauds. It detects two more than AE with five fewer false alerts. These share a validation target, not equal realized test FPR or review volume. At the 1\% validation policy, all three find 52 frauds with 381,314,304 false alerts respectively. At 0.1\%, GMM finds six frauds while AE and hybrid find none. Policy choice therefore matters. We do not retrospectively select a test-winning cutoff as a new default.

\subsection{Complementarity and Uncertainty}
GMM and AE score ranks have Spearman correlation 0.746. Among the top 100, both find the same 24 frauds, and GMM finds five additional frauds. AE contributes no unique fraud at that capacity. Equal averaging therefore loses some of the GMM ranking advantage. This diagnostic is capacity-specific and does not rule out complementary cases at other thresholds.

A paired stratified bootstrap resamples legitimate and fraud test rows separately, preserving class counts, for 200 replicates. The percentile 95\% interval for hybrid AP minus AE AP is [0.0060,0.0177]. This conditional interval ignores temporal dependence, refitting variation and benchmark reuse. It does not establish future-data or cross-seed superiority. Each of only 74 frauds changes recall by about 1.35 percentage points.

\section{Ablation Studies}
\textbf{Input noise:} Plain and denoising AE differ only in corruption SD (0 versus 0.10); training rows, seed, architecture, decay and stopping rule match. Noise raises PR area from 0.1281 to 0.1308 and top-100 frauds from 22 to 24. At the default policy both find 33, but plain AE creates fewer false alerts (127 versus 137). The benefit is metric-specific.

\textbf{Bottleneck size:} Widening 8 to 12 also increases parameters from 1,253 to 1,385. Normal-validation MSE improves from 0.7130 to 0.6599, yet PR area falls to 0.1044 and top-100 frauds to 13. Reconstruction fit and fraud ranking are different goals.

\textbf{Seed sensitivity:} Changing only the seed from 42 to 43 gives PR area 0.0915 and 12 frauds in 100 reviews. This exposes substantial sensitivity. Two seeds are a diagnostic rather than a distributional uncertainty study; the declared main seed remains fixed.

\textbf{Linear reference:} An eight-component PCA fitted on the same legitimate rows scores reconstruction MSE. It reaches PR area 0.0712 with six frauds in 100 reviews. The nonlinear AE is stronger here, while fitting methods and parameter counts differ.

\textbf{Hybrid branch removal:} GMM-only and AE-only results quantify the effect of removing each branch. GMM-only ranks best. The hybrid's operational gains at some policies do not establish a universal improvement. All weaker runs, checkpoints and histories are retained; no test-ranked ablation replaces the declared main model.

\section{Error Analysis and Limitations}
Notebook 04 shows a caught fraud, a false alert and a missed fraud at the default AE threshold, including the five largest squared feature errors. These are qualitative examples chosen after evaluation. Anonymous features permit numerical attribution but not a business explanation of stolen-card behavior.

The two-day historical dataset cannot demonstrate long-term drift resistance. The provider's PCA fitting scope is unavailable. Exact-duplicate removal is an assumption. Normal-only filtering relies on known labels. Unusual legitimate transactions and normal-looking fraud cause overlapping scores. Diagonal covariance and small model grids constrain GMM; initialization, reconstruction generalization and an unoptimized architecture constrain AE. Fusion is simple and fixed, and its gain over AE does not beat the GMM ranking. Costs, latency and analyst outcomes are not measured. This is a research comparison, not a production blocking system.

\section{Reproducibility and Team Responsibilities}
Five ordered notebooks provide shared motivation, data checks, feature scaling, AE training and final evaluation. Each starts from a fresh namespace and saves its outputs. Assertions check data identity, feature equality, finite scores, counts, validation FPR budgets and historical GMM parity. Tested versions include Python 3.12.14, NumPy 1.26.4, pandas 2.2.2, scikit-learn 1.4.2, PyTorch 2.6.0 and matplotlib 3.8.4. CPU settings and dependency versions are saved. Four AE runs require about 33 seconds of training in this workspace; total execution and fresh GMM fitting add time. Hardware can change timing and low-level numerics.

The retained responsibilities are Keshav (2401010222), data/EDA and notebook 01; Vaibhav (2401010486), features and notebook 02; Mayank (2401010267), training/ablations and notebook 03; H Maheshwari, called Hardik (2401010174), evaluation and notebook 04; and Mitul Bhatia (2401010279), leadership, integration and coordination. Mayank and Keshav retain README and presentation responsibility. All five jointly contribute to the report. These are responsibility assignments, not fabricated evidence of completed individual work. The actual-work record must be completed by members. AI assistance supported preparation; each member must verify and understand the work and follow faculty disclosure rules.

\section{Conclusion}
The DL extension successfully implements normal-only denoising reconstruction and an auditable score-fusion hybrid. GMM remains the best overall ranking model on the reused benchmark. Hybrid improves AE and can reduce false alerts compared with GMM at declared validation policies, with a recall cost at the default. Ablations show that wider bottlenecks and lower normal reconstruction error need not improve fraud detection, and that random-seed sensitivity is material. Future work should start with truly later data, more training seeds and explicit analyst costs before choosing additional complexity.

\begin{thebibliography}{00}
\bibitem{em} A. P. Dempster, N. M. Laird, and D. B. Rubin, ``Maximum likelihood from incomplete data via the EM algorithm,'' JRSS-B, 1977, doi:10.1111/j.2517-6161.1977.tb01600.x.
\bibitem{sakurada} M. Sakurada and T. Yairi, ``Anomaly detection using autoencoders with nonlinear dimensionality reduction,'' MLSDA, pp. 4--11, 2014, doi:10.1145/2689746.2689747.
\bibitem{vincent} P. Vincent et al., ``Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion,'' JMLR, vol. 11, pp. 3371--3408, 2010. \url{https://www.jmlr.org/papers/v11/vincent10a.html}
\bibitem{pr} T. Saito and M. Rehmsmeier, ``The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets,'' PLOS ONE, 2015, doi:10.1371/journal.pone.0118432.
\bibitem{sk} scikit-learn, ``Average precision score,'' version 1.4.2 documentation. \url{https://scikit-learn.org/1.4/modules/generated/sklearn.metrics.average_precision_score.html}
\bibitem{torch} PyTorch, ``MSELoss'' and ``Adam,'' documentation. \url{https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html}.
\bibitem{data} ULB Machine Learning Group/Worldline, ``Credit Card Fraud Detection,'' Kaggle. \url{https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud}
\end{thebibliography}
\end{document}
'''
tex = tex.replace('@@RANKING@@', '\n'.join(latex_rows)).replace('@@POLICIES@@', '\n'.join(policy_rows))
(P2 / 'report/Phase2_Report.tex').write_text(tex)
(P2 / 'report/IEEEtran.cls').write_bytes((ROOT / 'phase1/report/IEEEtran.cls').read_bytes())

# Dedicated plots remain legible at the report's actual IEEE column width.
with plt.rc_context({'font.size':8, 'axes.titlesize':9, 'axes.spines.top':False,
                     'axes.spines.right':False, 'legend.fontsize':7}):
    h = pd.read_csv(P2 / 'results/denoising_ae_history.csv')
    fig, ax = plt.subplots(figsize=(3.5,2.35))
    ax.plot(h.epoch, h.train_mse, label='Clean training', color='#24668d')
    ax.plot(h.epoch, h.normal_validation_mse, label='Legitimate validation', color='#338575')
    ax.set(xlabel='Epoch', ylabel='Reconstruction MSE', title='Main denoising autoencoder')
    ax.legend(); fig.tight_layout()
    fig.savefig(P2 / 'figures/report_learning.png', dpi=220); plt.close(fig)
    scores = pd.read_csv(P2 / 'results/test_scores.csv')
    fig, ax = plt.subplots(figsize=(3.5,2.4))
    for n, color in zip(main, ['#24668d','#338575','#a9761a']):
        p, r, _ = precision_recall_curve(scores.Class, scores[n])
        ax.plot(r,p,label=f'{names[n]} ({ranking.loc[n,"pr_auc_trapezoidal"]:.4f})', color=color, linewidth=1.2)
    ax.axhline(scores.Class.mean(), color='gray', linestyle='--', linewidth=.8)
    ax.set(xlabel='Recall', ylabel='Precision', xlim=(0,1), ylim=(0,.4), title='Test PR curves (legend: trapezoidal area)')
    ax.legend(); fig.tight_layout()
    fig.savefig(P2 / 'figures/report_pr.png', dpi=220); plt.close(fig)

css = '''body{max-width:1000px;margin:40px auto;padding:0 24px;font:17px/1.65 system-ui,sans-serif;color:#183243;background:#fafbf9}h1{font-size:36px;line-height:1.2}h2{margin-top:2em;font-size:25px}h3{margin-top:1.7em}a{color:#155c8a}table{border-collapse:collapse;width:100%;font-size:15px}td,th{padding:10px;border-bottom:1px solid #d4dde2;text-align:left}th{background:#e7eef1}code,pre{font-family:ui-monospace,monospace;background:#edf1f3}pre{padding:18px;overflow:auto}img{max-width:100%}li{margin:7px 0}@media print{body{margin:0;font-size:11pt}a{color:inherit}h2,h3{break-after:avoid}tr{break-inside:avoid}}'''
markdown = mistune.create_markdown(plugins=['table'])
for file in ['LAST_DAY_STUDY_GUIDE', 'RESULTS_EXPLAINED', 'PRESENTATION_SCRIPT']:
    text = (P2 / 'docs' / f'{file}.md').read_text()
    body = markdown(text)
    body = body.replace('href="RESULTS_EXPLAINED.md"', 'href="RESULTS_EXPLAINED.html"')
    body = body.replace('href="TEAM_ROLES.md"', 'href="../docs/TEAM_ROLES.md"')
    (P2 / 'output' / f'{file}.html').write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{file.replace("_"," ")}</title><style>{css}</style><body><p><a href="../../index.html#phase2">Project home</a></p>{body}</body></html>')

# The shared root index is generated by build_index.py.

deck_data = {'ranking': ranking.reset_index().to_dict('records'), 'policies': policies.to_dict('records'),
             'settings': settings, 'summary': summary, 'complementarity': complement,
             'learning': pd.read_csv(P2 / 'results/denoising_ae_history.csv').to_dict('records')}
(ROOT / '.build/phase2').mkdir(parents=True, exist_ok=True)
(ROOT / '.build/phase2/deck_data.json').write_text(json.dumps(deck_data, indent=2))
print('Prepared measured summaries, IEEE LaTeX source and reading page.')

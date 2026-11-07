# Phase 2: Card Fraud Triage

**Team T011 · Section A · Project 26**

This folder contains the deep learning extension and an implemented GMM–autoencoder hybrid. Phase 1 lives in the neighbouring `phase1/` folder. Both phases use the single root `index.html`.

For the shortest file guide, open [START_HERE.md](START_HERE.md).

## Start here

- **Study without running Python:** open [the project index](../index.html#phase2). It links all five executed notebooks, the study guide, results and deliverables.
- **Understand the DL part first:** [deep learning in 10 minutes](output/DEEP_LEARNING_IN_10_MINUTES.html), then notebook 03. The autoencoder is the neural model; score fusion is the hybrid extension.
- **Revise for the viva:** [last-day study guide](docs/LAST_DAY_STUDY_GUIDE.md), then [results explained](docs/RESULTS_EXPLAINED.md) and [simple presentation script](docs/PRESENTATION_SCRIPT.md).
- **Present and submit after team review:** [IEEE report](output/Phase2_Report.pdf), [20-slide PowerPoint](output/Phase2_Presentation.pptx), [presentation PDF](output/Phase2_Presentation.pdf), [report LaTeX](report/Phase2_Report.tex), and the notebooks.
- **Download everything together:** [complete portable package](output/Card_Fraud_Triage_Complete.zip). It includes both phases and the shared root index. Download the official CSV separately on a new machine.
- **Check requirements and roles:** [faculty checklist](docs/PHASE2_CHECKLIST.md), [retained roles](docs/TEAM_ROLES.md), [literature](docs/LITERATURE_REVIEW.md).

## Measured result

| Model | PR AUC (trapezoidal) | AP | Frauds in 100 reviews |
|---|---:|---:|---:|
| GMM | 0.1496 | 0.1532 | 29 |
| Denoising AE | 0.1308 | 0.1338 | 24 |
| Hybrid | 0.1412 | 0.1445 | 25 |

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

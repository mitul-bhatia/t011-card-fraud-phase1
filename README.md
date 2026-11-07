# Card Fraud Triage

**Team T011 · Section A · Project 26**

Open the **[project website](https://mitul-bhatia.github.io/t011-card-fraud-phase1/)**. It is the single navigation page for both phases, their notebooks, results, reports, presentations and study guides. In a downloaded copy, open **[index.html](index.html)** locally.

| Folder | Work | Read or run |
|---|---|---|
| [phase1](phase1/README.md) | Machine learning: GMM density scoring | Read 01, then run 02–05 |
| [phase2](phase2/START_HERE.md) | Deep learning: autoencoder, hybrid and ablations | Read 00, then run 01–04 |

Each phase keeps its own `notebooks/`, `results/`, `figures/`, `report/`, `output/` and documentation. Executed HTML reading copies are in each phase's `output/notebooks/` folder. Final reports and presentations are in that phase's `output/` folder.

The dataset lives once at **`phase1/data/raw/creditcard.csv`**. Phase 2 reuses it and the Phase 1 baseline for a consistent comparison. Data, generated arrays and model files are excluded from Git and the portable package. Download the official CSV from the [ULB/Kaggle dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) on a new machine.

## Run the project

Use Python 3.12. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python phase1/scripts/run_all.py
python phase2/scripts/run_all.py
python phase2/scripts/prepare_submission.py
python phase2/scripts/build_index.py
python phase2/scripts/check_deliverables.py
python phase2/scripts/package_project.py
```

To install only the ML dependencies, use `phase1/requirements.txt`. Each runner uses a fresh Python process for every notebook and refreshes the linked reading copies. You can also open the notebooks in VS Code or Jupyter and Run All in the same order.

For future full checks, use one command after installing dependencies and adding the CSV:

```bash
python phase2/scripts/verify_project.py --rerun
```

This reruns both phases and the separate exploratory GMM notebook, reloads model weights to check scores and cutoffs, checks the deliverables, rebuilds the ZIP and checks its extracted contents. Omit `--rerun` to check the current saved work. Logs stay in `.build/verification/`. See the [requirements and verification review](phase2/docs/VERIFICATION_REVIEW.md) for the remaining team actions.

## Measured result

The GMM has test PR area **0.1496**, the denoising AE **0.1308**, and the hybrid **0.1412**. Their first 100 reviews find **29, 24 and 25 frauds**. At the default validation policy, the hybrid makes **40 fewer false alerts** than the GMM while catching **two fewer frauds**. The test benchmark was already used in Phase 1; Phase 2 does not use its labels for tuning.

## Team

| Member | Enrollment | Retained responsibility |
|---|---|---|
| H Maheshwari (Hardik) | 2401010174 | Evaluation and thresholds |
| Vaibhav | 2401010486 | Features and scaling |
| Mayank | 2401010267 | Model training; README and presentation with Keshav |
| Keshav | 2401010222 | Data quality and EDA; README and presentation with Mayank |
| Mitul Bhatia | 2401010279 | Overall project lead, integration and coordination |

All five jointly contribute to the report. Each member should record and explain their actual work in the [team record](phase2/docs/TEAM_ROLES.md).

The [complete portable package](phase2/output/Card_Fraud_Triage_Complete.zip) includes both phases and the shared index. The supplied [course instructions](CSAI_Project_Instructions_Phase_1_and_Phase_2.pdf) remain at the root because they apply to both phases.

# Phase 1: Gaussian Mixture Model

[Project home](../index.html#phase1) · [Phase 2](../index.html#phase2)

Team T011, Section A, Project 26. This phase learns legitimate transaction density with a GMM and evaluates later mixed transactions.

Read notebook 01, then run 02–05. Use the root [index.html](../index.html#phase1) to read all five executed notebooks, see the results and open the deliverables.

| Folder | Contents |
|---|---|
| `notebooks/` | Five main notebooks; exploratory work is labelled in `drafts/` |
| `results/` | Measured metrics, thresholds, capacities and audit records |
| `figures/` | Data, training and evaluation charts |
| `data/raw/` | The official CSV, shared with Phase 2 and excluded from Git |
| `data/processed/` | Generated time splits and scaled arrays, excluded from Git |
| `models/` | Local fitted models and scaler, excluded from Git |
| `report/` | IEEE LaTeX source |
| `output/` | Final report and original presentation; reading copies in `notebooks/` |
| `docs/` | Literature, code explanation, team contributions and viva material |

## Run from the repository root

```bash
python -m pip install -r phase1/requirements.txt
python phase1/scripts/run_all.py
```

The runner starts a fresh process per notebook and regenerates the linked HTML reading copies. You can also Run All in each notebook using VS Code or Jupyter. Its setup finds the `phase1` folder automatically.

The official dataset belongs at `phase1/data/raw/creditcard.csv`. The first data notebook downloads the documented public mirror if this file is missing. Phase 2 reuses the same CSV and baseline.

## Measured result

The selected K=8 GMM has test PR area **0.1496**. Its default validation policy catches **37 of 74 frauds**, with **172 false alerts**. The top 100 reviews contain **29 frauds**. The one-Gaussian reference has PR area **0.0527**.

## Deliverables and study

- [IEEE report](output/Phase1_Report.pdf) and [LaTeX source](report/Phase1_Report.tex).
- [Original team presentation](output/Phase1_Presentation_Final.pptx), preserved unchanged.
- [Earlier detailed Phase 1 overview](docs/PHASE1_ORIGINAL_OVERVIEW.md).
- [Study guide](output/STUDY_GUIDE.html), [code guide](docs/CODE_GUIDE.md) and [file map](FILE_MAP.md).
- [Contribution record](docs/TEAM_CONTRIBUTIONS.md): Keshav handles data, Vaibhav features, Mayank training, Hardik evaluation, and Mitul overall leadership. Mayank and Keshav retain README and presentation responsibility; all five jointly contribute to the report.

The original saved Phase 1 measurements are retained in `results/original_run/`; direct result files describe the checked rerun.

Earlier ZIP exports remain in `output/archive/` as historical local copies. The current combined package and single navigation page are linked from the root index. Publishing the repository is separate from submitting coursework to faculty.

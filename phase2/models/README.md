# Generated local artifacts

Notebooks 01-04 write split CSVs, scaled arrays, a scaler and PyTorch state dictionaries here. They are excluded from Git. Rerun the notebooks in order to regenerate them. Training history, settings, summaries and figures live in the adjacent tracked folders.

Notebook 04 reuses the frozen Phase 1 GMM if its local binary exists. On a fresh checkout it rebuilds the original K grid and selects minimum training BIC. The dataset must be downloaded separately into the `phase1/data/raw/creditcard.csv` path.

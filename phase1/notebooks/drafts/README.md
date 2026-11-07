# Draft experiment: GMM component count

Open `06_GMM_Component_Experiment.ipynb` and read/run its cells in order. It uses the same prepared inputs as notebooks 03–05 and compares K=4, 6, 8, 12, 16, 24 and 32. It loads the original K=8 submission model as the exact baseline.

All experiment calculations are visible inside this notebook. No separate analysis script is required. Run the original notebooks 02–04 first if the saved inputs/models are missing.

Saved draft tables, charts, and an HTML reading copy are under `output/drafts/gmm_k_sweep/`. The original notebooks, results, report, and presentation are preserved.

From the repository root, rerun this optional experiment in a fresh process with:

```bash
python phase1/scripts/run_all.py phase1/notebooks/drafts/06_GMM_Component_Experiment.ipynb
```

The full `python phase2/scripts/verify_project.py --rerun` command also checks this draft. Its results remain exploratory and do not replace the declared Phase 1 baseline.

These are exploratory results on a previously examined test period. Better numbers here do not establish performance on fresh unseen transactions. The density-model selection rule is minimum training BIC; test-based comparisons are descriptive.

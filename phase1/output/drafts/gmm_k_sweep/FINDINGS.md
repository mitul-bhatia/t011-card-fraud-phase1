# Draft GMM component experiment

## Measured findings: increasing K helped on this split

| K | Test PR AUC | Fraud found / 74 | Fraud missed | False alerts | Fraud in top 100 |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.1249 | 34 | 40 | 172 | 25 |
| 6 | 0.1248 | 34 | 40 | 171 | 25 |
| 8 | 0.1496 | 37 | 37 | 172 | 29 |
| 12 | 0.1431 | 36 | 38 | 174 | 27 |
| 16 | 0.1448 | 38 | 36 | 179 | 30 |
| 24 | 0.1782 | 44 | 30 | 186 | 31 |
| 32 | 0.2481 | 45 | 29 | 195 | 34 |

**Same protocol:** identical 169,893 legitimate training rows, 29 already-scaled features, diagonal covariance, three starts, maximum 200 iterations, seed 42. Every candidate converged. Each model uses its own legitimate-validation threshold at the same 0.5% FPR budget; all produced 283 legitimate validation alerts (0.4992%).

**K=32 versus the original K=8:**
- Finds **45 instead of 37** of the 74 test frauds: eight additional frauds found, eight fewer missed.
- Recall rises from **50.00% to 60.81%**.
- False alerts rise from **172 to 195**: 23 additional legitimate reviews.
- Total alerts rise from **209 to 240**: 31 additional reviews in total.
- Test FPR rises from **0.3035% to 0.3441%**. The validation budgets match, while the later observed test FPRs differ.
- Precision at the default policy rises from **17.70% to 18.75%**.
- In exactly 100 reviews, fraud found rises from **29 to 34**, so precision@100 rises from **29% to 34%**.
- Test trapezoidal PR AUC rises from **0.1496 to 0.2481**.
- Validation PR AUC also rises, from **0.0718 to 0.1535**; validation fraud found rises from 29 to 41 of 57.

**What this establishes:** changing K can help this experiment. K=32 has the smallest training BIC among the expanded candidates and improves the reported test ranking and default recall over K=8. The improvement is not free: it creates more false alerts and still misses 29 test frauds.

**What it does not establish:** 32 is not proven optimal. BIC is still falling at the right boundary. Increasing K is not uniformly helpful: K=12 has lower BIC than K=8 but slightly worse test PR AUC and one fewer fraud found. K=24 has higher default test precision than K=32, while K=32 has higher PR AUC, recall and precision@100. A model is not best at every measure merely because it has more components.

**Selection and uncertainty:** the prespecified expanded-grid density selection is minimum training BIC, which chooses K=32. We also show test comparisons descriptively. Because the test period was already inspected, this is exploratory evidence rather than a new unbiased final evaluation. There are only 74 test frauds and this draft uses one seed with three initializations. No statistical significance or future performance guarantee is claimed.

**Submission status:** this exploratory experiment does not replace the official K=8 baseline, five main notebooks, report or presentation. Publishing these draft files with the repository does not make them the declared final comparison.

**Files to read:**
- `training_comparison.csv`: fit times, convergence, AIC and BIC.
- `comparison.csv`: complete validation and test measurements.
- `01_training_bic.png`: legitimate-density selection.
- `02_pr_auc.png`: validation and test ranking comparison.
- `03_default_policy.png`: fraud found and false alerts together.
- `04_top_100.png`: equal 100-review capacity.
- `05_pr_curve.png`: original K=8 versus training-BIC-selected K=32.

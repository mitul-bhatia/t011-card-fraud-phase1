# Results explained simply

These are measured results from the real dataset. The GMM is the frozen Phase 1 baseline. The same later test period has **56,746 transactions and 74 frauds**, a fraud base rate of **0.1304%**. The test benchmark was already exposed in Phase 1; Phase 2 settings were fixed without using its labels for tuning.

## Which method gives the best ranking?

| Model | PR AUC (trapezoidal) | AP | Frauds in 100 reviews |
|---|---:|---:|---:|
| GMM | 0.1496 | 0.1532 | 29 |
| Denoising AE | 0.1308 | 0.1338 | 24 |
| Hybrid | 0.1412 | 0.1445 | 25 |

The GMM gives the strongest ranking here. The hybrid improves the AE result but does not beat the GMM on PR AUC, AP or precision at 100. All three rankings find substantially more fraud in the first 100 reviews than the approximately **0.13 frauds expected from 100 random reviews**. This is a dataset comparison, not a deployment guarantee.

## What happens at the default 0.5% validation FPR policy?

| Model | Fraud found / 74 | False alerts | Total reviews | Recall | Precision | Actual test FPR |
|---|---:|---:|---:|---:|---:|---:|
| GMM | 37 | 172 | 209 | 50.00% | 17.70% | 0.3035% |
| Denoising AE | 33 | 137 | 170 | 44.59% | 19.41% | 0.2417% |
| Hybrid | 35 | 132 | 167 | 47.30% | 20.96% | 0.2329% |

Compared with the GMM at this predeclared validation policy, the hybrid makes **40 fewer false alerts (23.3% fewer)** and catches **2 fewer frauds**. Its higher precision comes with lower recall. These are the same *validation target*, not identical realized test FPRs or identical review counts. The top-100 comparison above is the equal-review-count comparison.

Compared with the AE, the hybrid catches **2 more frauds** and creates **5 fewer false alerts** at this policy. That is evidence of a local benefit over the DL branch, while the GMM remains stronger on the overall ranking measures.

At the relaxed 1% validation policy, all three find 52 frauds. GMM makes 381 false alerts, AE 314, hybrid 304. At the strict 0.1% policy, GMM finds 6 frauds while AE and hybrid find 0. These differences show why one headline metric is insufficient.

## What did the ablations teach us?

| Model | PR AUC (trapezoidal) | AP | Frauds in 100 reviews |
|---|---:|---:|---:|
| GMM | 0.1496 | 0.1532 | 29 |
| PCA (8 components) | 0.0712 | 0.0729 | 6 |
| Plain AE | 0.1281 | 0.1312 | 22 |
| Denoising AE | 0.1308 | 0.1338 | 24 |
| Denoising AE, bottleneck 12 | 0.1044 | 0.1068 | 13 |
| Denoising AE, seed 43 | 0.0915 | 0.0938 | 12 |
| Hybrid | 0.1412 | 0.1445 | 25 |

- **Noise:** Denoising modestly improves PR AUC from 0.1281 to 0.1308 and finds 24 rather than 22 frauds in the top 100. At the default policy both catch 33, and the plain AE makes fewer false alerts (127 vs 137). Noise does not improve every measure.
- **Compression:** The 12-unit bottleneck lowers normal-validation MSE to 0.6599 from 0.7130, yet reduces test PR AUC to 0.1044. Better copying of normal transactions does not guarantee better fraud ranking.
- **Randomness:** Seed 43 reduces the denoising AE's PR AUC to 0.0915 and top-100 frauds to 12. The model is sensitive to initialization and batch order. We retain the declared seed 42 main result and disclose the weaker run.
- **Linear reconstruction:** PCA8 reaches PR AUC 0.0712 and finds 6 frauds in 100 reviews. The tested nonlinear AE is better here, although the fitting methods and parameterizations differ.
- **Hybrid branches:** Removing the AE leaves the GMM, which is the best overall ranking model. Removing the GMM leaves the weaker AE. This combination has a measured trade-off, not universal superiority.

## Why does the hybrid fail to beat the GMM at 100 reviews?

The GMM and AE score ranks have Spearman correlation **0.746**. All **24** frauds found by the AE in its top 100 are already among the GMM's **29** detected frauds at that capacity. The AE contributes **0 unique frauds** there, while the GMM contributes **5 unique frauds**. Equal averaging changes the order and loses some of the GMM's advantage. This overlap is specific to the top-100 policy; it does not prove there are no complementary cases at other thresholds.

## How certain is the comparison?

A 200-replicate paired, stratified row bootstrap gives a 95% percentile interval of **[0.0060, 0.0177]** for hybrid AP minus main-AE AP. This describes sampling variation conditional on these fitted models and the observed class counts. It ignores temporal dependence, training variation and benchmark reuse. The large seed effect is a reason to avoid a broad superiority claim.

## Training actually observed

Both the plain and main denoising AE reach the 30-epoch cap and restore epoch 30. The wide model stops at epoch 15 and restores epoch 10. Seed 43 reaches epoch 30 and restores epoch 28. We do not claim early stopping fired on every run.

## A defensible viva conclusion

“We implemented and evaluated both density and nonlinear reconstruction scoring. The GMM ranks fraud best on this reused benchmark. The hybrid improves the AE and can reduce false alerts relative to the GMM at declared validation policies, with a recall cost at the default policy. Our ablations show that better reconstruction and a larger bottleneck do not automatically produce better anomaly detection.”

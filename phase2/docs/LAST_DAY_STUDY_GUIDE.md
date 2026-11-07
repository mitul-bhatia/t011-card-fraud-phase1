# Learn the whole project in one revision session

Team T011, Section A, Project 26. Use this with the executed notebooks and the measured results in `../results/`. The aim is understanding. Reading this once cannot guarantee viva readiness.

## A practical three-hour revision plan

| Time | Work | Check yourself |
|---|---|---|
| 0-20 min | Read notebook 00 and this guide through the architecture explanation | Explain the project without using the word accuracy |
| 20-45 min | Read notebooks 01 and 02 and their plots | Explain which rows fit the scaler and why |
| 45-85 min | Read notebook 03 slowly | Explain every line of the training loop in your own words |
| 85-115 min | Read notebook 04 and the results summary | Calculate precision, recall and FPR from one confusion matrix |
| 115-145 min | Rehearse the presentation using its speaker notes | Explain where the hybrid helps and where it loses |
| 145-180 min | Each member answers five questions aloud and demonstrates their notebook | Let another member ask a follow-up question |

## The 30-second explanation

We prioritize card transactions for human review. Both models learn from the same earlier legitimate transactions. The GMM scores how unlikely a transaction looks under a density model. The autoencoder compresses and rebuilds it, then scores reconstruction error. Our hybrid combines normalized versions of both scores. We compare fraud ranking and the cost of false alerts on the same later test period.

An anomaly means unusual under a model. It does not prove fraud. This system orders a review queue; it does not automatically block cards.

## The data, without jargon

- A row is one transaction. `Class` gives the known answer: 0 legitimate, 1 fraud.
- V1-V28 are anonymous features created by the provider using PCA. We cannot recover their business meaning.
- `Amount` is the transaction amount. `Time` orders transactions by elapsed seconds.
- There are 284,807 original rows and 492 frauds. Removing 1,081 exact repeats leaves 283,726 rows and 473 frauds.
- Exact repeats might be genuine transactions because there are no transaction IDs. We retain Phase 1's removal assumption and disclose it.
- Earliest 60%: training, 170,235 rows with 342 frauds. Only its 169,893 legitimate rows fit the models.
- Next 20%: validation, 56,745 rows with 57 frauds. Legitimate validation rows control AE stopping, score normalization and thresholds.
- Last 20%: test, 56,746 rows with 74 frauds. Its fraud base rate is about 0.1304%.

Training learns the parameters. Validation controls choices before evaluation. Test measures later performance. Phase 1 already exposed this exact test benchmark, so we disclose its reuse rather than claiming a newly blind evaluation.

## Features: what enters both models?

Twenty-eight anonymous features plus `LogAmount = log(1 + Amount)` give 29 inputs. The log compresses very large amounts and safely handles zero. StandardScaler learns the mean and standard deviation from legitimate training rows only, then applies those same values everywhere.

For one feature, `scaled = (value - training_mean) / training_std`. A value one standard deviation above the training mean becomes 1. A value one standard deviation below becomes -1.

Class never enters the model. Time determines the split but does not enter the model. We do not add extra engineered features only for DL, because that would complicate the model comparison.

## Why the GMM comes first

A Gaussian describes a bell-shaped pattern with a center and spread. A mixture combines several such patterns. The mixture weights sum to one. K is the number of components, not the number of fraud categories.

EM alternates two steps. In the E-step it estimates how strongly each transaction belongs to each component. In the M-step it updates the centers, spreads and mixture weights using those memberships. It repeats until the fit changes little enough or reaches its iteration limit. Convergence does not prove a global optimum.

We tested K=1,2,4,8 in Phase 1. AIC and BIC reward fit and penalize additional parameters; lower is better. K=8 has the smallest training BIC in this tested grid. The criterion still falls at the grid edge, so we do not claim that 8 is globally optimal.

`GMM score = -log p(x)`. High score means low density under learned normal activity. Density is not probability of fraud. A density can exceed one, so negative log density can be negative.

Diagonal covariance estimates a separate variance per feature within a component. It improves speed and reduces parameter count. It ignores within-component cross-feature covariance, although mixtures can still represent some overall dependencies.

## Why add a neural network?

The GMM imposes a mixture of Gaussian shapes. The AE learns a nonlinear reconstruction function that may capture different relationships. This gives a second way to measure unusual activity. A second model is useful only if its scores improve an evaluation measure that matters.

A supervised fraud classifier learns a mapping from inputs to a fraud label. Our AE learns a mapping from normal inputs back to those same inputs. We need known legitimate labels to filter training data, so the overall setup is commonly called one-class or semi-supervised anomaly detection. Its weight-training objective does not use fraud targets.

## The architecture, layer by layer

| Layer | Width | Plain meaning | Why it is here |
|---|---:|---|---|
| Input | 29 | The scaled transaction features | Same information as the GMM |
| Encoder hidden | 16 | Learned combinations of the inputs | Starts compression |
| Bottleneck | 8 | A short learned representation | Limits capacity and forces information loss |
| Decoder hidden | 16 | Expands the short representation | Starts rebuilding |
| Output | 29 | Reconstructed feature values | Compare with the original input |

Each Linear layer computes weighted sums plus biases. Each hidden ReLU computes `max(0, value)`, which adds nonlinearity. The output stays linear because inputs are standardized real numbers, including negatives. A sigmoid would restrict outputs to 0-1 and would not match these targets.

The 8-unit bottleneck is undercomplete because 8 is less than 29. It discourages an identity map, but does not prevent the network from reconstructing fraud. The main network has 1,253 trainable parameters. Verify the exact count in notebook 03 rather than memorizing an unsupported number.

## A denoising AE, with a tiny example

We add small Gaussian noise to each training input. If a clean value is 1.00, its noisy input might be 1.08. The target stays 1.00. The model learns to recover a clean normal transaction from a slightly perturbed version.

Noise standard deviation 0.10 is measured in standardized feature units. The main experiment fixes it before test evaluation. No noise is added when computing validation or test anomaly scores.

If the input is `[1, 2]` and reconstruction is `[1.1, 1.8]`, squared errors are 0.01 and 0.04. Their mean is 0.025. In the real model, average over 29 features. This is reconstruction MSE. The training loss averages it over the batch as well.

## Every important line of the training loop

| Code idea | Meaning |
|---|---|
| `model.train()` | Put the model in training mode; mode matters for layers such as dropout if used |
| Batch size 1024 | Use 1,024 transactions for one optimizer update |
| `noisy = clean + noise * randn_like(clean)` | Corrupt only the input |
| `optimizer.zero_grad()` | Clear gradients from the previous batch |
| `rebuilt = model(noisy)` | Forward pass through encoder and decoder |
| `MSELoss(rebuilt, clean)` | Compare the output with the clean target |
| `loss.backward()` | Apply the chain rule to compute gradients |
| `optimizer.step()` | Adam changes the trainable weights using those gradients |
| One epoch | One full pass over the training rows |
| `model.eval()` | Put the model in evaluation mode |
| `torch.no_grad()` | Avoid recording gradients during scoring |
| Best-state copy | Keep the weights at the best accepted normal-validation loss |
| Early stopping | Stop after five epochs without improvement of at least 0.0001, or after 30 epochs |

Learning rate 0.001 controls update size. Weight decay 0.00001 discourages large weights. Noise, compression and early stopping provide other regularization. We did not add dropout, convolution, recurrence, or attention: these 29 tabular features do not supply an image grid or a transaction sequence.

`eval()` does not itself disable gradients; `no_grad()` does. The clean training and validation losses in the plot use the same scoring mode. Thus their gap is not just noise corruption. Time drift can also create a gap.

## The hybrid, with arithmetic

We first transform the GMM score with `asinh` and the AE error with `log1p`. These compress extreme magnitudes. For each transformed score, normal validation values supply its median and IQR. IQR is the 75th percentile minus the 25th percentile.

`normalized_score = (transformed_score - normal_validation_median) / normal_validation_IQR`

`hybrid_score = 0.5 * normalized_GMM + 0.5 * normalized_AE`

For example, if normalized GMM is 2 and normalized AE is 4, hybrid is 3. A negative normalized value simply means it lies below that normal-validation median.

Why normalize? Raw scales differ, so their direct sum would give arbitrary influence. Why use median/IQR? They reduce sensitivity to extreme values. Why equal weights? They give a simple fixed baseline without a fraud-label tuning search. Robust normalization still does not make the score a calibrated probability.

The rationale is complementary evidence: density rarity and reconstruction difficulty. We test that rationale with GMM-only, AE-only and hybrid results, score correlation, and the overlap of detected frauds. A combination that runs successfully can still have no demonstrated advantage. See `RESULTS_EXPLAINED.md` for the measured conclusion.

## Thresholds, ranking and the cost of alerts

A threshold turns a score into an alert. Our rule is strictly `score > cutoff`. For a 0.5% validation FPR budget, take the normal-validation 99.5th percentile using the higher observed value. This keeps the observed validation false-positive fraction at or below that budget. It does not guarantee the same future FPR.

Our fixed budgets are 0.1%, 0.5% and 1%. The default is 0.5%. A larger FPR budget generally allows more reviews and catches more fraud. A budget of 0.5% legitimate false positives does not mean exactly 0.5% of all transactions get reviewed, because detected frauds add to the queue.

Top-100 ranking instead sends exactly 100 transactions for review in the entire test batch. It does not use a cutoff chosen from test labels. Sort scores decreasingly and take the first 100. Use stable sorting to make ties reproducible.

## Metrics you must calculate aloud

TP: fraud with an alert. FP: legitimate with an alert. FN: fraud missed. TN: legitimate without an alert.

- Recall = TP/(TP+FN). Of all fraud, how much did we catch?
- Precision = TP/(TP+FP). Of all alerts, how many were fraud?
- FPR = FP/(FP+TN). Of all legitimate transactions, how many did we disturb?
- Precision@100 = fraud in the top 100 / 100. This is a review hit rate.
- PR curve = precision versus recall across thresholds.
- Trapezoidal PR AUC integrates that curve with straight-line interpolation. Higher means a better summary on this data.
- Average precision (AP) weights precision by increments in recall. It differs from trapezoidal PR AUC; show the label clearly.

Phase 1 example: TP=37, FP=172, FN=37, TN=56,500. Recall=50%; precision=37/209=17.70%; FPR=172/56,672=0.3035%. There are 209 alerts. These are GMM numbers, not the AE result.

Accuracy hides the rare fraud class. A zero-fraud detector can be over 99.8% accurate here. PR metrics also depend on base rate, so comparisons should use the same test set.

## What the ablations answer

| Comparison | Question | What to avoid claiming |
|---|---|---|
| Plain AE vs denoising AE | Does input noise help? | Every regularizer always improves fraud ranking |
| Bottleneck 8 vs 12 | Does weaker compression help? | This isolates only capacity without changing parameter count |
| Main seed 42 vs seed 43 | How sensitive is the result to randomness? | Two seeds establish statistical certainty |
| PCA8 vs AE | Is nonlinear reconstruction useful here? | These have identical fitting objectives and parameter counts |
| GMM and AE vs hybrid | Does combination add value? | Combining models guarantees extra marks |

Main model settings stay fixed even if an ablation scores higher on test. Switching to the highest test result would use the test as validation. The best normal reconstruction model can still have worse fraud ranking.

## Common viva questions and short answers

**1. Is this classification?** The model learns normal reconstruction or density; evaluation compares its scores with fraud labels. A threshold yields binary alerts.

**2. Why not a supervised classifier?** This project studies normal-only training and compares density with reconstruction. A classifier would be a useful separate baseline, but it answers a different training question.

**3. Do you use fraud labels at all?** Yes, to filter legitimate training/validation rows and evaluate. No fraud examples update AE weights or fit GMM parameters.

**4. Why avoid a random split?** Chronological splitting better mimics learning from the past and scoring the future. It still cannot remove every source of leakage in the provider's original PCA preparation.

**5. Why standardize if V1-V28 already came from PCA?** PCA features can still have different variances. MSE should not be dominated merely by differing scale.

**6. Why no Time feature?** Elapsed seconds since this file starts is not a stable business feature. We use it to order the split.

**7. Why only normal training rows?** That is the project hypothesis: learn legitimate activity and flag deviations.

**8. Why a small network?** It is easier to understand and constrains reconstruction capacity. Larger networks may learn to copy anomalous inputs too well.

**9. Why eight bottleneck units?** A simple undercomplete predeclared choice. The 12-unit experiment probes sensitivity; we do not claim eight is optimal.

**10. Why ReLU?** It supplies a simple nonlinear transformation and efficient gradients, although negative preactivations have zero gradient.

**11. Why linear output?** Scaled targets include negative and unbounded values. Sigmoid or tanh would constrain the reconstruction range.

**12. Why MSE?** It is simple, differentiable, and penalizes large reconstruction differences. It is sensitive to extreme values, which is also a limitation.

**13. Why Adam?** It adapts parameter update sizes using gradient moments. It is a practical fixed optimizer here, not evidence of superiority over all optimizers.

**14. What is backpropagation?** The chain rule calculates how the loss changes with each trainable weight.

**15. Why clear gradients?** PyTorch accumulates them by default. Clearing starts a fresh batch update.

**16. Is validation used for weight updates?** No. Its normal rows choose a checkpoint and policy settings. That still makes it part of model selection.

**17. Does early stopping always stop early?** No. Training may reach the 30-epoch cap. Read the actual history and restored epoch.

**18. Why can fraud reconstruct well?** Fraud can share normal feature structure, or the network can generalize reconstruction beyond normal training patterns.

**19. Why can a legitimate transaction score highly?** Rare legitimate behavior can differ from learned normal patterns. Distribution change can also raise scores.

**20. What is special about the hybrid?** It combines two distinct anomaly signals using a declared normalization and fixed weights. Its measured contribution determines whether that rationale helps.

**21. Is the hybrid a new neural architecture?** No. It is score-level fusion after separate GMM and AE scoring.

**22. Why not add the raw scores?** Their units and ranges differ. The sum could be dominated by a scale choice rather than useful evidence.

**23. Why not train a GMM on the bottleneck?** That is a plausible separate hybrid. Our simple score fusion preserves a direct same-feature Phase 1 baseline and is easier to explain. We did not implement a latent GMM.

**24. Are the scores fraud probabilities?** No. A score ranks unusual activity. Probability calibration would require a separate justified calibration design.

**25. Is the test untouched?** No Phase 2 tuning uses test labels, but the team already saw Phase 1 results on this test period. We explicitly call it a reused chronological benchmark.

**26. Can you deploy this in a bank?** This is a small historical research comparison. Deployment would require future evaluation, drift monitoring, business costs, latency checks and operational review.

**27. Why show bad ablations?** They reveal whether an intervention helps and prevent reporting only favorable experiments.

**28. What does the bootstrap mean?** It resamples legitimate and fraud test rows separately to estimate conditional uncertainty in paired AP differences. It does not refit models or account for temporal dependence.

**29. What if the confidence interval includes zero?** This analysis does not clearly distinguish the paired AP difference from zero. Even an interval excluding zero would not prove performance on new data.

**30. Why are feature-error explanations limited?** Anonymous V-features reveal numerical contributions, not the original business reason for fraud.

**31. What would you improve next?** Evaluate truly later data, repeat more seeds, examine drift and analyst costs, then test one additional model change at a time using validation.

**32. Did each member already implement their assigned section?** The role table retains agreed responsibilities. Each member must review, rerun and record their own actual work before claiming implementation or understanding.

## Your five-minute demonstration

1. Open notebook 00 and state the review-queue question.
2. Open notebook 02 and show that both models receive 29 inputs from the same legitimate training rows.
3. Open notebook 03 and explain the forward pass, clean target, loss, backward pass and optimizer step.
4. Show the PR curves, threshold trade-offs and top-100 result in notebook 04.
5. Explain the measured hybrid result, one failed or weaker ablation, and the reused-test limitation.

Before presenting, fill the actual-work record in `TEAM_ROLES.md`. AI assistance helped prepare these materials. Each member remains responsible for checking the work and following faculty disclosure rules.

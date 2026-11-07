# Requirements and verification review

Reviewed on 7 November 2026. Start at the shared [project index](../../index.html).

The supplied [course PDF](../../CSAI_Project_Instructions_Phase_1_and_Phase_2.pdf), especially page 2, defines the evaluation areas. The user's Project 26 brief defines the GMM/AE comparison and metrics. The user's later request defines the two phase folders and one shared index. Document instructions were used as requirements, not permission to send messages or submit work.

## Course requirements and prepared evidence

| Required area | What is prepared | Review outcome |
|---|---|---|
| Phase 1 literature review | Notebook 01 and literature guide | Sources and limits explained |
| Dataset quality and EDA | Phase 1 notebook 02 | Official CSV fingerprint, duplicates, base rates and chronological periods checked |
| Feature engineering | Phase 1 notebook 03 | 29 inputs; scaling learns only from legitimate training rows |
| ML theory and application | Phase 1 notebooks 04–05 | EM, diagonal GMM, AIC/BIC, thresholds and measured results |
| DL architecture logic | Phase 2 notebook 03 | 29–16–8–16–29, ReLU, linear output, 1,253 parameters and rationale |
| DL literature | Phase 2 literature guide and report | Denoising motivation distinguished from reproducing a published method |
| DL data and regularization | Phase 2 notebooks 01–03 | Same legitimate rows; noise, weight decay and normal-validation stopping |
| Theoretical rigor | Report and notebooks | Density, reconstruction MSE, normalization and threshold equations |
| Meaningful hybrid | Phase 2 notebook 04 | Implemented fixed score fusion; gains and losses reported |
| Ablations | Plain AE, bottleneck 12, seed 43, PCA8 and branch removal | All results retained, including weaker runs |
| Technical validation | Notebook assertions and two independent checkers | Data-to-score checks and metric/count checks passed |
| Repository and code quality | Separate phases, executed reading pages, requirements and runners | Local files checked; remote repository access still needs team review |
| Presentations | Both phase decks; Phase 2 PDF and speaker script | Phase 2 PDF slides visually reviewed; native PowerPoint editing remains unverified |
| IEEE/LaTeX reports | Both phase PDFs and editable sources | Both sources compile in two passes; rebuilt text matches delivered PDFs; Phase 2's four pages visually reviewed |
| Individual viva | Study guides, examples, speaking notes and questions | Prepared for study; each student must demonstrate understanding |
| Five members and individual contribution | Original names, IDs and role areas retained | Actual-work record remains for each member to complete truthfully |

## Project 26 checks

| User requirement | Evidence checked |
|---|---|
| Legitimate-only training | 169,893 normal rows; independent scaler refit and shared-array comparison |
| GMM selection by AIC/BIC | K=1,2,4,8; converged K=8 selected by minimum training BIC |
| Undercomplete or denoising AE | Four trained runs; weights reloaded independently and stopping histories replayed |
| Same mixed test set | 56,746 rows and 74 frauds; labels and GMM scores match Phase 1 |
| PR metrics, review capacity and false positives | Seven score columns, PR area/AP, all review capacities and confusion matrices recomputed |
| Validation thresholds | All cutoffs independently recomputed from normal validation; no test-fitted cutoff |
| Hybrid value | GMM ranks best; fusion improves AE and changes the false-alert trade-off |
| Simple separate folders and navigation | `phase1/`, `phase2/`, one root index, linked reading copies and results |

## Latest checks

- All ten main notebooks ran again, each in a fresh Python process. Read-only introduction notebooks have no code cells to execute.
- Both main summaries match the saved pre-restructure summaries within numerical tolerance. Neither model settings nor the conclusion changed during verification.
- The separate six-code-cell exploratory GMM notebook also ran in a fresh process. Its expanded-grid results reproduced, and its K=32 result remains exploratory; it does not replace the declared K=8 comparison.
- The independent model checker traced the raw CSV through legitimate-only scaling to the saved arrays. Reloaded GMM and four AE checkpoints reproduced the seven test score columns, PCA reference, hybrid normalization and all validation cutoffs.
- The saved-artifact checker independently recomputes ranking metrics, review capacities and threshold confusion matrices. It checks notebook execution, local links, identities, report readability and presentation structure.
- The previous fresh-environment, extracted-package run reproduced both phases. The repeatable verification command also rebuilds the package, validates its checksums and checks the extracted reading pages.
- The shared index and AE training reading page opened in native Chrome. The browser automation host remains unavailable, and screenshot capture failed. This confirms page loading and accessible content, not complete desktop/mobile visual rendering.
- The Phase 2 slide PDF was rendered and all twenty slides reviewed again. The report was rendered and all four pages reviewed again. This does not claim a native PowerPoint editing test.

Reference checks: the [Vincent paper page](https://www.jmlr.org/papers/v11/vincent10a.html), [Saito/Rehmsmeier paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432) and [versioned AP documentation](https://scikit-learn.org/1.4/modules/generated/sklearn.metrics.average_precision_score.html) were accessible. The Sakurada/Yairi DOI returned HTTP 403, so the existing full-text limitation remains disclosed.

## Repeat the checks

From the repository root, with dependencies installed and the official CSV in `phase1/data/raw/`:

```bash
python phase2/scripts/verify_project.py --rerun
```

Omit `--rerun` to check current saved work. The command stops at a failed step and saves logs under `.build/verification/`. Local data/model checks require regenerated artifacts; a portable copy without them receives saved-artifact checks only. The checker reports this distinction explicitly.

## Remaining team actions

- Each member records actual work and rehearses the complete pipeline.
- Check current faculty communications for the Phase 2 date, upload form and limits; these are not supplied in the PDF.
- Confirm faculty can access the intended remote repository and open the deck in PowerPoint before presenting.

No local check can establish a student's individual understanding or guarantee marks. The test is the previously exposed Phase 1 benchmark, and the hybrid does not beat the GMM's overall ranking. Publishing the repository is separate from submitting coursework through a faculty form.

# Deliverable review

## Code and measured results

- All ten main notebooks across both phases have saved outputs and no cell errors. Each runner uses a separate Python process for each notebook.
- After the folder restructure, the complete ZIP was extracted into a clean copy. The root requirements installed in a fresh environment, and both phases ran in order from the official CSV. Both phases reproduced their saved metrics.
- A fresh-copy run regenerated features, trained all four AE variants and rebuilt the original GMM grid without the local GMM binary. Every saved ranking metric reproduced within numerical tolerance.
- The artifact checker independently recalculates PR area, AP, precision at 100 and every threshold confusion matrix from saved scores.
- Validation FPR budgets, dataset identity, shared features and historical GMM parity are checked.
- All weaker experiments remain visible. No new tuning uses the known test labels.

## Presentation

- Twenty slides use the earlier deck's dark teal, teal and amber visual direction.
- Ten editable charts include measured PR curves, a real validation reconstruction, learning curves, review capacity and threshold trade-offs.
- Every slide has matching speaking notes. A PDF reading copy and a separate plain-language script support rehearsal.
- The deck passed package, geometry, font and embedded chart-data checks. All slide previews were visually reviewed, and the PDF copy was checked after conversion.
- Preview and PDF checks do not claim a native Microsoft PowerPoint editing test.

## Report and study material

- The four-page IEEE report includes DL rationale, training, hybrid equations, ablations, trade-offs, references and limitations.
- The last-day guide explains the training loop and metric denominators, with a three-hour revision plan and viva questions.
- The short first-read guide explains why Phase 2 is DL, its noisy input and clean target, tensor shapes, weight updates and actual loss reduction. Notebook 03 traces four legitimate inputs through the network before training. The updated cover and teaching slides lead with the neural model; GMM and score fusion follow as comparisons.
- The same five members and role areas appear throughout. Individual actual-work entries remain for the named members to complete after doing their review.
- The result conclusion is consistent: GMM ranks best; hybrid improves AE and changes the false-alert trade-off. The test benchmark is reused from Phase 1.

## Folder and portable copy

Start at `../START_HERE.md` or `../../index.html#phase2`. The canonical report and presentation are in `../output/`; notebooks, results and report source have separate folders. Earlier presentation revisions and build files stay outside the deliverable folder.

The complete ZIP includes both phases and the single root index. It excludes the raw CSV, split arrays and fitted models. The README explains how to download the dataset and regenerate those files.

The root page's 44 local resources returned HTTP 200, and the artifact checker checked links in all exported reading pages. On 7 November, the root page and AE training reading page opened in native Chrome. Screenshot capture failed, so complete visual browser rendering is not marked as verified.

The [latest requirement review](VERIFICATION_REVIEW.md) records the repeated notebook run, independent model reload checks, report/slide inspection and remaining team actions. Use `python phase2/scripts/verify_project.py --rerun` from the repository root to repeat the full local workflow.

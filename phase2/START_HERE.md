# Phase 2: start here

Team T011, Section A, Project 26. The team and responsibility areas stay the same.

## Understand Phase 2 first

**Phase 2 is deep learning.** Its main model is a denoising autoencoder: a neural network that learns to rebuild legitimate transactions. Four learned Linear layers and hidden ReLU activations transform 29 inputs into 8 compressed values and back into 29 outputs. Backpropagation and Adam update its 1,253 parameters.

Start with **[Deep learning in 10 minutes](output/DEEP_LEARNING_IN_10_MINUTES.html)**. Then read notebook 03. The GMM is the earlier ML baseline; the hybrid is an extension after the AE is trained.

## Present and submit

1. [PowerPoint presentation](output/Phase2_Presentation.pptx): 20 slides, editable charts and speaking notes.
2. [Presentation PDF](output/Phase2_Presentation.pdf): easy to view and rehearse.
3. [IEEE report](output/Phase2_Report.pdf): methods, measured results, ablations and limitations.

## Study in this order

1. [Read the short DL explanation](output/DEEP_LEARNING_IN_10_MINUTES.html), then [open the notebooks](../index.html#phase2).
2. [Use the last-day study guide](output/LAST_DAY_STUDY_GUIDE.html) for the complete theory and viva questions.
3. [Rehearse the presentation script](output/PRESENTATION_SCRIPT.html).
4. [Understand the result trade-offs](output/RESULTS_EXPLAINED.html).

The main idea: both models learn from legitimate transactions. The GMM scores unusual density. The autoencoder scores reconstruction error. The hybrid combines normalized scores.

The measured conclusion: GMM ranks best here. At the default validation policy, hybrid makes 40 fewer false alerts than GMM and catches two fewer frauds. The test benchmark was already used in Phase 1.

## Team review

Use [the retained role table](docs/TEAM_ROLES.md) and [the faculty checklist](docs/PHASE2_CHECKLIST.md). Each member must review their section, record their actual work and understand the complete pipeline.

For code and the full folder structure, see [README](README.md). A [complete portable package](output/Card_Fraud_Triage_Complete.zip) includes both phases, reading copies and final artifacts. The dataset and generated model files stay outside the archive; download the official CSV as explained in the README.

# A short DL literature review

## Sakurada and Yairi, 2014

**Anomaly Detection Using Autoencoders with Nonlinear Dimensionality Reduction.** MLSDA, pp. 4-11. DOI: https://doi.org/10.1145/2689746.2689747

This is the assignment's suggested autoencoder anomaly-detection reading. It motivates exploring nonlinear reconstruction as an alternative to a distributional model. Our report does not reproduce its experiments or transfer any reported performance to card fraud. Publisher full text was not available through the research tool during preparation, so no unverified numerical or detailed experimental claim is attributed to this paper.

## Vincent et al., 2010

**Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion.** JMLR 11(110), pp. 3371-3408. https://www.jmlr.org/papers/v11/vincent10a.html

The paper trains networks to recover clean inputs from corrupted versions and studies learned representations. Our experiment borrows the clean-target denoising idea. It uses a small jointly trained tabular network, not the paper's full stacked pretraining procedure. Its classification results are not evidence that our fraud anomaly ranking must improve. Our plain-versus-noise ablation checks the local effect.

## Dempster, Laird and Rubin, 1977

**Maximum Likelihood from Incomplete Data via the EM Algorithm.** JRSS-B. https://doi.org/10.1111/j.2517-6161.1977.tb01600.x

EM explains alternating hidden-membership estimates and parameter updates. It supports the retained GMM baseline. A converged likelihood fit does not guarantee fraud separation or the global optimum. Training AIC/BIC choose legitimate density complexity; test PR metrics assess the different goal of fraud ranking.

## Saito and Rehmsmeier, 2015

**The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets.** PLOS ONE. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432

The paper motivates examining precision and recall when positive cases are rare. We therefore show PR curves and review workload. We also report FPR because an analyst's tolerance for legitimate alerts matters. No metric is universally sufficient: compare on the same base rate and include operational counts.

## Implementation references

- scikit-learn 1.4, AP definition: https://scikit-learn.org/1.4/modules/generated/sklearn.metrics.average_precision_score.html. AP uses recall increments and differs from trapezoidal PR integration. Both are named separately in the saved tables.
- scikit-learn mixture guide: https://scikit-learn.org/1.4/modules/mixture.html. Diagonal covariance offers a smaller parameterization at the cost of within-component covariance structure.
- PyTorch MSELoss: https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html. Used to compare reconstructed and clean targets.
- PyTorch Adam: https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html. Used with a fixed learning rate and weight decay.
- ULB/Worldline dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud. Provider PCA details, two-day scope and anonymous variables limit interpretation.

## The gap this project tests

A GMM models normal density with Gaussian components. A nonlinear AE models normal reconstruction. They can fail on different transactions, but they can also agree on the same errors. We preserve the same data representation and evaluate both individually before combining them through normalized score fusion. The ablations test denoising, compression, randomness, linear reconstruction and removal of each hybrid branch. This is an empirical comparison, not a claim of algorithmic novelty.

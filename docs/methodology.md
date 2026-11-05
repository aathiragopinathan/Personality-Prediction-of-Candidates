# Experiment methodology

1. Generate fictional observations with a fixed seed, or supply another schema-compatible **synthetic** CSV.
2. Validate ranges, categories, target coverage, unique nonmissing IDs, and provenance. Require at least ten observations per class for five-fold training evaluation after the holdout split.
3. Reserve a stratified 25% test set. Save IDs so separation can be inspected.
4. Compare four pipelines on identical five-fold stratified partitions of the remaining training rows. Fit all learned preprocessing within each fold. Select the highest mean macro F1; ties use the listed model order.
5. Refit the selected pipeline and majority baseline on all training rows. Evaluate both on the held-out set once.
6. Report per-class errors, uncertainty, and feature importance. Save the complete fitted pipeline, dataset hash, split manifest, and package versions.
7. Verify that reloaded inference matches in-memory inference. Example inference records are generated with a different seed and have no target column.

Macro F1 gives each class equal weight; accuracy alone can obscure a weak minority class. Balanced accuracy averages class recall. Log loss assesses probability quality without claiming calibration. The bootstrap interval resamples a fixed test set and does not describe uncertainty from training or distribution change. Permutation importance can be negative due to noise and measures association, not causal effect.

The logistic ablation holds estimator settings and splits fixed while adding pair means. These redundant linear features change regularization geometry and are not guaranteed to help. A claim that feature engineering improved performance must be backed by the comparison CSV, not by the existence of engineered columns.

Tests cover schema failures, deterministic generation, missingness handling, immutability of input features, training-only imputation statistics, ID split disjointness, selection consistency, complete report generation, and prediction parity after serialization. CI runs these tests on Python 3.11 and 3.14. No real-world validity or fairness test is implied by passing software tests.

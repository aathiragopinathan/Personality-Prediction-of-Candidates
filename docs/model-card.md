# Model card

**Intended use:** educational demonstration and portfolio review of a tabular ML workflow on fictional data. **Excluded use:** real personality assessment; screening, ranking, recommending, or rejecting candidates; psychological or medical inference.

**Model:** selected by highest training-only five-fold macro F1 among a majority-class dummy, raw-feature logistic regression, engineered-feature logistic regression, and engineered-feature random forest. See `metrics.json` for the selected model in a particular run. Hyperparameters are fixed in source, not tuned to meet historical claims.

**Inputs:** six synthetic behavioral/skill scores and one fictional activity preference. Engineered features average planning/analytical skill, collaboration/communication skill, and curiosity/adaptability. Both components must be present; otherwise the pair feature remains missing for imputation.

**Preprocessing:** median numeric imputation, missingness indicators for features with missing training values, standard scaling, constant categorical imputation, and one-hot encoding. Entirely missing numeric features are retained with the imputer's zero fill. The public schema rejects unrecognized activity categories even though the underlying encoder tolerates unknown values defensively.

**Outputs:** a synthetic class and its class probabilities. These are uncalibrated model outputs, not psychometric certainty. No candidate score, hiring recommendation, or suitability rank is produced.

**Evaluation:** a stratified 25% test split; accuracy, balanced accuracy, macro F1, per-class metrics, log loss, confusion matrix, and 1,000-resample percentile bootstrap interval for accuracy. Raw-input permutation importance uses ten repeats and macro F1. See the report for measured results.

**Limits:** generated labels are structurally related to generated features. This deliberately easy-to-inspect world cannot validate personality measurement or generalization to real populations. Test-set explanations are descriptive only. Demographic fairness, calibration, psychometric reliability, construct validity, external validity, and hiring benefit have not been established. Selection among several models can introduce training-CV optimism; the test set must remain untouched by subsequent tuning.

Before any study involving real people, a separate research design would need validated targets and instruments, informed consent and data governance, appropriate independent review, representative sampling, demographic and intersectional audits, reliability assessment, external validation, and human review. This repository does not implement that study or a deployment service.

# Portfolio and resume framing

Use claims supported by this implementation:

> Built a reproducible Python classification experiment with scikit-learn pipelines, missing-value handling, categorical encoding, and behavioral/skill feature engineering on explicitly synthetic data.

> Compared logistic regression and random forest models against a majority baseline using stratified five-fold cross-validation; produced held-out macro F1, per-class metrics, confusion matrices, and permutation feature importance.

> Packaged the experiment as a tested CLI with dataset fingerprints, auditable train/test splits, saved preprocessing/model artifacts, and documented responsible-use limitations.

If quoting a numerical result, cite the checked-in report and explicitly call it **synthetic held-out performance**. Do not present this new implementation as the original CV project or as independently proving its 92% accuracy or 25% hiring improvement. Do not claim deployment, fairness mitigation, validated personality measurement, or actual hiring impact.

For an interview, demonstrate `generate → train → predict`, explain why preprocessing belongs inside cross-validation, compare macro F1 with accuracy, inspect the feature ablation, and discuss why recovering a synthetic generator is different from measuring personality.

Original evidence still missing: dataset origin and size, original feature definitions, target construction, original algorithms and libraries, split/CV protocol, baseline, hiring-outcome metric, privacy/consent measures, fairness evidence, deployment status, individual contributions, and dates.

"""Training-only model selection, untouched test evaluation, and audit artifacts."""

import hashlib
import json
import platform
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from .data import FEATURES, TARGET, validate_data
from .features import make_pipeline


def evaluate(model, X, y):
    predicted = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro")),
        "log_loss": float(log_loss(y, model.predict_proba(X), labels=model.classes_)),
        "classification_report": classification_report(
            y, predicted, output_dict=True, zero_division=0
        ),
    }


def train(data_path, output_dir, seed=42):
    frame = pd.read_csv(data_path)
    validate_data(frame)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    training, testing = train_test_split(
        frame, test_size=0.25, stratify=frame[TARGET], random_state=seed
    )
    X_train, y_train = training[FEATURES], training[TARGET]
    X_test, y_test = testing[FEATURES], testing[TARGET]
    candidates = {
        "majority_baseline": make_pipeline(DummyClassifier(strategy="most_frequent"), False),
        "logistic_raw": make_pipeline(LogisticRegression(max_iter=1500, C=1.0), False),
        "logistic_engineered": make_pipeline(LogisticRegression(max_iter=1500, C=1.0)),
        "random_forest_engineered": make_pipeline(
            RandomForestClassifier(
                n_estimators=160,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=seed,
                n_jobs=1,
            )
        ),
    }
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    comparison = []
    for name, pipeline in candidates.items():
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=folds,
            scoring={"macro_f1": "f1_macro", "accuracy": "accuracy"},
        )
        comparison.append(
            {
                "model": name,
                "cv_macro_f1_mean": float(scores["test_macro_f1"].mean()),
                "cv_macro_f1_std": float(scores["test_macro_f1"].std()),
                "cv_accuracy_mean": float(scores["test_accuracy"].mean()),
            }
        )
    # The baseline remains eligible: model complexity must earn its place.
    best = max(comparison, key=lambda row: row["cv_macro_f1_mean"])["model"]
    model = candidates[best].fit(X_train, y_train)
    baseline = candidates["majority_baseline"].fit(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    baseline_metrics = evaluate(baseline, X_test, y_test)
    predictions = model.predict(X_test)
    rng = np.random.default_rng(seed)
    correct = np.asarray(predictions == y_test)
    bootstrap = [float(rng.choice(correct, len(correct), replace=True).mean()) for _ in range(1000)]
    metrics["accuracy_bootstrap_95_interval"] = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    importance = permutation_importance(
        model, X_test, y_test, scoring="f1_macro", n_repeats=10, random_state=seed
    )
    pd.DataFrame(
        {
            "feature": FEATURES,
            "macro_f1_decrease_mean": importance.importances_mean,
            "macro_f1_decrease_std": importance.importances_std,
        }
    ).sort_values("macro_f1_decrease_mean", ascending=False).to_csv(
        output / "feature_importance.csv", index=False
    )
    pd.DataFrame(comparison).to_csv(output / "model_comparison.csv", index=False)
    test_predictions = testing[["candidate_id", TARGET]].copy()
    test_predictions["prediction"] = predictions
    for index, label in enumerate(model.classes_):
        test_predictions[f"probability_{label}"] = model.predict_proba(X_test)[:, index]
    test_predictions.to_csv(output / "test_predictions.csv", index=False)
    display = ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, labels=model.classes_, cmap="Blues", colorbar=False
    )
    display.ax_.set_title("Synthetic work styles · held-out test set")
    display.figure_.tight_layout()
    display.figure_.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(display.figure_)
    manifest = {
        "purpose": "Synthetic educational experiment; no real personality or hiring validity",
        "seed": seed,
        "dataset_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "rows": len(frame),
        "training_rows": len(training),
        "test_rows": len(testing),
        "class_counts": frame[TARGET].value_counts().to_dict(),
        "missing_counts": frame[FEATURES].isna().sum().to_dict(),
        "selected_model": best,
        "selection_metric": "training 5-fold CV macro F1",
        "test": metrics,
        "baseline_test": baseline_metrics,
        "versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    (output / "metrics.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "split_manifest.json").write_text(
        json.dumps(
            {
                "train_ids": training.candidate_id.tolist(),
                "test_ids": testing.candidate_id.tolist(),
            },
            indent=2,
        )
        + "\n"
    )
    joblib.dump(
        {
            "pipeline": model,
            "features": FEATURES,
            "schema_version": 1,
            "data_origin": "synthetic",
            "sklearn_version": sklearn.__version__,
        },
        output / "model.joblib",
    )
    report = f"""# Synthetic experiment results

This is a newly implemented educational experiment, not evidence of personality validity.

- Dataset: {len(frame):,} fictional observations; seed {seed}.
- Split: {len(training)} training / {len(testing)} test; unique candidate IDs.
- Selected model: **{best}**, using training-only 5-fold macro F1.
- Held-out accuracy: **{metrics["accuracy"]:.1%}**.
- Held-out macro F1: **{metrics["macro_f1"]:.3f}**.
- Balanced accuracy: **{metrics["balanced_accuracy"]:.3f}**.
- Log loss: **{metrics["log_loss"]:.3f}**.
- Majority baseline accuracy: **{baseline_metrics["accuracy"]:.1%}**.
- Accuracy 95% bootstrap interval: {metrics["accuracy_bootstrap_95_interval"][0]:.1%}–{metrics["accuracy_bootstrap_95_interval"][1]:.1%}.

The interval describes resampling uncertainty for this fixed model and synthetic test set;
it does not include training or population uncertainty. Probabilities are uncalibrated model
outputs. Permutation importance measures predictive reliance, not causal influence.
The raw/engineered logistic comparison is an ablation; feature engineering is not assumed
to improve results. The test set must not be reused for tuning.

![Confusion matrix](confusion_matrix.png)

See `model_comparison.csv`, `feature_importance.csv`, and `metrics.json` for details.
The historical 92% test accuracy and 25% hiring improvement are unverified source claims.
No hiring outcome is generated or evaluated here. No fairness or psychometric validation
has been performed, and this model must not be used to screen, rank, or reject people.
"""
    (output / "REPORT.md").write_text(report)
    return manifest


def predict(model_path, data_path, output_path):
    """Load only a trusted local artifact: joblib uses executable pickle serialization."""
    artifact = joblib.load(model_path)
    if artifact.get("schema_version") != 1 or artifact.get("features") != FEATURES:
        raise ValueError("Incompatible model artifact schema.")
    if artifact.get("sklearn_version") != sklearn.__version__:
        raise ValueError("Use the scikit-learn version recorded in the model artifact.")
    frame = pd.read_csv(data_path)
    validate_data(frame, labeled=False)
    model = artifact["pipeline"]
    output = frame[["candidate_id"]].copy()
    output["synthetic_work_style_prediction"] = model.predict(frame[FEATURES])
    probabilities = model.predict_proba(frame[FEATURES])
    for index, label in enumerate(model.classes_):
        output[f"probability_{label}"] = probabilities[:, index]
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    return output

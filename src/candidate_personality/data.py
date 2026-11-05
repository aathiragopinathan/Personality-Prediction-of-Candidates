"""Explicit schema and a reproducible, intentionally artificial data-generating process."""

import numpy as np
import pandas as pd

NUMERIC = [
    "planning",
    "collaboration",
    "curiosity",
    "adaptability",
    "analytical_skill",
    "communication_skill",
]
CATEGORICAL = ["preferred_activity"]
FEATURES = NUMERIC + CATEGORICAL
ACTIVITIES = ["analysis", "discussion", "experimentation"]
LABELS = ["analytical", "collaborative", "exploratory"]
TARGET = "synthetic_work_style"


def generate_data(rows=1200, seed=42):
    """Create fictional observations. Labels are arbitrary rules plus latent noise."""
    if rows < 120:
        raise ValueError("Use at least 120 rows for stratified evaluation.")
    rng = np.random.default_rng(seed)
    values = rng.beta(2, 2, size=(rows, len(NUMERIC))) * 4 + 1
    frame = pd.DataFrame(values.round(3), columns=NUMERIC)
    activity = rng.choice(ACTIVITIES, rows)
    frame["preferred_activity"] = activity
    scores = np.column_stack(
        [
            0.8 * values[:, 0] + values[:, 4] + 0.5 * (activity == "analysis"),
            values[:, 1] + 0.8 * values[:, 5] + 0.5 * (activity == "discussion"),
            values[:, 2] + 0.8 * values[:, 3] + 0.5 * (activity == "experimentation"),
        ]
    )
    scores += rng.normal(0, 0.65, scores.shape)
    frame[TARGET] = np.array(LABELS)[scores.argmax(axis=1)]
    # Missingness is introduced after labels: the unobserved values remain latent.
    frame[NUMERIC] = frame[NUMERIC].mask(rng.random(values.shape) < 0.04)
    frame.loc[rng.random(rows) < 0.03, "preferred_activity"] = np.nan
    frame.insert(0, "candidate_id", [f"SYN-{i:06d}" for i in range(rows)])
    frame.insert(1, "data_origin", "synthetic")
    return frame


def validate_data(frame, *, labeled=True):
    """Reject malformed data and keep identity/target columns outside model inputs."""
    required = ["candidate_id", "data_origin"] + FEATURES + ([TARGET] if labeled else [])
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Dataset must contain rows.")
    if frame["candidate_id"].isna().any() or frame["candidate_id"].duplicated().any():
        raise ValueError("candidate_id must be nonmissing and unique (one row per candidate).")
    if not frame["data_origin"].eq("synthetic").all():
        raise ValueError("This reference implementation accepts synthetic data only.")
    for column in NUMERIC:
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid = frame[column].notna() & (values.isna() | ~values.between(1, 5))
        if invalid.any():
            raise ValueError(f"{column} must contain numbers in [1, 5] or missing values.")
    if not frame["preferred_activity"].dropna().isin(ACTIVITIES).all():
        raise ValueError(f"preferred_activity must be one of {ACTIVITIES} or missing.")
    if labeled:
        if frame[TARGET].isna().any() or set(frame[TARGET].unique()) != set(LABELS):
            raise ValueError(f"Training data must include exactly these labels: {LABELS}")
        if frame[TARGET].value_counts().min() < 10:
            raise ValueError("Each label needs at least 10 observations.")

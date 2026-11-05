import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from candidate_personality.data import FEATURES, NUMERIC, generate_data, validate_data
from candidate_personality.features import DERIVED, WorkStyleFeatures, make_pipeline


def test_generator_reproducibility_and_contract():
    first = generate_data(180, seed=7)
    pd.testing.assert_frame_equal(first, generate_data(180, seed=7))
    assert not first.equals(generate_data(180, seed=8))
    validate_data(first)
    assert first[NUMERIC].isna().any().any()


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda df: df.drop(columns="planning"), "Missing columns"),
        (lambda df: df.assign(candidate_id="duplicate"), "unique"),
        (lambda df: df.assign(planning=6), "numbers in"),
        (lambda df: df.assign(planning="invalid"), "numbers in"),
        (lambda df: df.assign(planning=np.inf), "numbers in"),
        (lambda df: df.assign(data_origin="real"), "synthetic data only"),
        (lambda df: df.assign(preferred_activity="unrecognized"), "preferred_activity"),
        (lambda df: df.assign(synthetic_work_style="invented"), "exactly these labels"),
    ],
)
def test_invalid_inputs_fail_with_useful_errors(mutation, message):
    with pytest.raises(ValueError, match=message):
        validate_data(mutation(generate_data(180)))


def test_feature_engineering_preserves_missingness_and_input():
    original = generate_data(180)[FEATURES]
    original.loc[0, "planning"] = np.nan
    saved = original.copy(deep=True)
    transformed = WorkStyleFeatures().fit_transform(original)
    pd.testing.assert_frame_equal(original, saved)
    assert np.isnan(transformed.loc[0, DERIVED[0]])
    np.testing.assert_allclose(
        transformed[DERIVED[1]],
        (original.collaboration + original.communication_skill) / 2,
        equal_nan=True,
    )


def test_preprocessing_uses_training_statistics_only():
    frame = generate_data(180)
    train = frame.iloc[:140]
    held_out = frame.iloc[140:][FEATURES].copy()
    pipeline = make_pipeline(LogisticRegression(max_iter=1000), engineered=False)
    pipeline.fit(train[FEATURES], train.synthetic_work_style)
    imputer = pipeline.named_steps["preprocess"].named_transformers_["numeric"]["impute"]
    np.testing.assert_allclose(imputer.statistics_, train[NUMERIC].median().to_numpy())
    saved_statistics = imputer.statistics_.copy()
    held_out[NUMERIC] = 5.0
    pipeline.predict(held_out)
    np.testing.assert_array_equal(imputer.statistics_, saved_statistics)


def test_entirely_missing_numeric_feature_is_supported():
    frame = generate_data(180)
    frame["planning"] = np.nan
    pipeline = make_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(frame[FEATURES], frame.synthetic_work_style)
    assert np.isfinite(pipeline.predict_proba(frame[FEATURES])).all()

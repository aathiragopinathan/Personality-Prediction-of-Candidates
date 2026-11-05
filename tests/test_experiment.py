import json

import joblib
import numpy as np
import pandas as pd
import pytest

from candidate_personality.data import FEATURES, TARGET, generate_data
from candidate_personality.experiment import predict, train


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory):
    folder = tmp_path_factory.mktemp("experiment")
    frame = generate_data(180, seed=11)
    source = folder / "data.csv"
    frame.to_csv(source, index=False)
    output = folder / "run"
    manifest = train(source, output, seed=11)
    return folder, frame, output, manifest


def test_end_to_end_evaluation_and_split(completed_run):
    _folder, frame, output, manifest = completed_run
    splits = json.loads((output / "split_manifest.json").read_text())
    assert set(splits["train_ids"]).isdisjoint(splits["test_ids"])
    assert set(splits["train_ids"] + splits["test_ids"]) == set(frame.candidate_id)
    assert manifest["test_rows"] == 45
    comparison = pd.read_csv(output / "model_comparison.csv")
    expected = comparison.loc[comparison.cv_macro_f1_mean.idxmax(), "model"]
    assert manifest["selected_model"] == expected
    assert len(manifest["dataset_sha256"]) == 64
    assert 0 <= manifest["test"]["accuracy"] <= 1
    assert (output / "confusion_matrix.png").stat().st_size > 0
    assert "unverified source claims" in (output / "REPORT.md").read_text()


def test_serialized_predictions_match_and_exclude_identity(completed_run):
    folder, frame, output, _manifest = completed_run
    inference = frame.drop(columns=TARGET).head(12).copy()
    inference["candidate_id"] = [f"NEW-{i}" for i in range(len(inference))]
    source = folder / "inference.csv"
    inference.to_csv(source, index=False)
    results = predict(output / "model.joblib", source, folder / "predictions.csv")
    model = joblib.load(output / "model.joblib")["pipeline"]
    np.testing.assert_array_equal(
        results.synthetic_work_style_prediction, model.predict(inference[FEATURES])
    )
    probability_columns = results.filter(like="probability_")
    np.testing.assert_allclose(probability_columns.sum(axis=1), 1)
    assert results.candidate_id.tolist() == inference.candidate_id.tolist()

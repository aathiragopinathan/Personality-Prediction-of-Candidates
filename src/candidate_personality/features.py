"""Feature construction and fold-local preprocessing."""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, NUMERIC

DERIVED = ["planning_analysis", "collaboration_communication", "curiosity_adaptability"]


class WorkStyleFeatures(TransformerMixin, BaseEstimator):
    """Pair means require both inputs, leaving incomplete pairs for imputation."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = X.copy()
        for name, pair in zip(
            DERIVED,
            [
                ("planning", "analytical_skill"),
                ("collaboration", "communication_skill"),
                ("curiosity", "adaptability"),
            ],
        ):
            result[name] = (result[pair[0]] + result[pair[1]]) / 2
        return result


def make_pipeline(estimator, engineered=True):
    numeric = NUMERIC + (DERIVED if engineered else [])
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(
                                strategy="median", add_indicator=True, keep_empty_features=True
                            ),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "category",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL,
            ),
        ]
    )
    return Pipeline(
        [
            ("features", WorkStyleFeatures() if engineered else "passthrough"),
            ("preprocess", preprocessing),
            ("model", estimator),
        ]
    )

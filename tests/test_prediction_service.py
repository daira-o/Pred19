import numpy as np
import pandas as pd

from pred19.features import REQUIRED_FEATURES
from pred19.inference.prediction import predict_record


class SyntheticPipeline:
    feature_names_in_ = np.array(REQUIRED_FEATURES)
    classes_ = np.array([0, 1])

    def predict_proba(self, frame):
        assert tuple(frame.columns) == REQUIRED_FEATURES
        return np.array([[0.3, 0.7]])


def test_prediction_uses_positive_class_and_configured_threshold():
    row = pd.Series(dict.fromkeys(REQUIRED_FEATURES, 1.0))
    result = predict_record(SyntheticPipeline(), row, threshold=0.65, model_version="test-only")
    assert result.probability == 0.7
    assert result.predicted_class == "Positive"
    assert result.threshold == 0.65

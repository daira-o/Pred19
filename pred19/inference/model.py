from __future__ import annotations

import joblib
import streamlit as st

from pred19.features import REQUIRED_FEATURES
from pred19.settings import MODEL_PATH


class ModelLoadError(RuntimeError):
    pass


@st.cache_resource(show_spinner="Loading the Pred19 model…")
def load_model():
    if not MODEL_PATH.is_file():
        raise ModelLoadError("The trained Pred19 pipeline is not available on this deployment.")
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as exc:
        raise ModelLoadError("The trained Pred19 pipeline could not be loaded safely.") from exc

    if not callable(getattr(model, "predict_proba", None)):
        raise ModelLoadError("The saved model does not support probability predictions.")

    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        actual = tuple(str(name) for name in names)
        if len(actual) != len(REQUIRED_FEATURES) or set(actual) != set(REQUIRED_FEATURES):
            raise ModelLoadError("The saved model's input features do not match the Pred19 specification.")
    return model

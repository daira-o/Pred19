from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from pred19.features import REQUIRED_FEATURES


@dataclass(frozen=True)
class PredictionResult:
    probability: float
    predicted_class: str
    threshold: float
    model_version: str
    timestamp: datetime


class PredictionError(RuntimeError):
    pass


def _positive_class_index(model) -> int:
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classifier = model.named_steps.get("classifier")
        classes = getattr(classifier, "classes_", None)
    if classes is None:
        raise PredictionError("The model does not identify its positive class.")
    values = list(classes)
    for candidate in (1, True, "1", "positive", "Positive"):
        if candidate in values:
            return values.index(candidate)
    raise PredictionError("The model's positive class could not be identified safely.")


def predict_record(model, row: pd.Series, threshold: float, model_version: str) -> PredictionResult:
    names = getattr(model, "feature_names_in_", None)
    order = tuple(str(name) for name in names) if names is not None else REQUIRED_FEATURES
    model_input = pd.DataFrame([{name: row[name] for name in order}], columns=order)
    try:
        probabilities = np.asarray(model.predict_proba(model_input))
        positive_probability = float(probabilities[0, _positive_class_index(model)])
    except PredictionError:
        raise
    except Exception as exc:
        raise PredictionError("The selected record could not be evaluated by the model.") from exc
    if not np.isfinite(positive_probability) or not 0 <= positive_probability <= 1:
        raise PredictionError("The model returned an invalid probability.")
    return PredictionResult(
        probability=positive_probability,
        predicted_class="Positive" if positive_probability >= threshold else "Negative",
        threshold=threshold,
        model_version=model_version,
        timestamp=datetime.now(timezone.utc),
    )

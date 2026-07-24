from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from pred19.settings import METRICS_PATH, ROC_PATH


@dataclass(frozen=True)
class PerformanceArtifacts:
    metrics: dict | None
    roc_curve: pd.DataFrame | None

    @property
    def available(self) -> bool:
        return self.metrics is not None and self.roc_curve is not None


def load_performance_artifacts() -> PerformanceArtifacts:
    metrics = None
    roc = None
    if METRICS_PATH.is_file():
        try:
            loaded_metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded_metrics, dict):
                metrics = loaded_metrics
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    if ROC_PATH.is_file():
        try:
            loaded_roc = pd.read_csv(ROC_PATH)
            required = ["fpr", "tpr", "threshold"]
            if set(required).issubset(loaded_roc.columns):
                loaded_roc[required] = loaded_roc[required].apply(pd.to_numeric, errors="coerce")
                if not loaded_roc[["fpr", "tpr"]].isna().any().any():
                    roc = loaded_roc
        except (OSError, ValueError, TypeError, pd.errors.ParserError):
            pass
    return PerformanceArtifacts(metrics, roc)


def resolve_threshold(model, metrics: dict | None) -> float | None:
    for attribute in ("decision_threshold_", "decision_threshold"):
        value = getattr(model, attribute, None)
        if value is not None:
            try:
                value = float(value)
                if 0 <= value <= 1:
                    return value
            except (TypeError, ValueError):
                pass
    for attribute in ("pred19_metadata", "metadata_"):
        metadata = getattr(model, attribute, None)
        if isinstance(metadata, dict) and "decision_threshold" in metadata:
            try:
                value = float(metadata["decision_threshold"])
                if 0 <= value <= 1:
                    return value
            except (TypeError, ValueError):
                pass
    if metrics and "decision_threshold" in metrics:
        try:
            value = float(metrics["decision_threshold"])
            return value if 0 <= value <= 1 else None
        except (TypeError, ValueError):
            return None
    return None


def resolve_model_version(model, metrics: dict | None) -> str:
    for attribute in ("model_version_", "model_version"):
        value = getattr(model, attribute, None)
        if value:
            return str(value)
    for attribute in ("pred19_metadata", "metadata_"):
        metadata = getattr(model, attribute, None)
        if isinstance(metadata, dict) and metadata.get("model_version"):
            return str(metadata["model_version"])
    if metrics and metrics.get("model_version"):
        return str(metrics["model_version"])
    return "Not specified"

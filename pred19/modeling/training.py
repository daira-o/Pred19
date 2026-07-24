from __future__ import annotations

import argparse
import csv
import io
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

from pred19.features import REQUIRED_FEATURES
from pred19.settings import METRICS_PATH, MODEL_PATH, ROC_PATH


RANDOM_STATE = 42
DECISION_THRESHOLD = 0.4


@dataclass(frozen=True)
class DataSplits:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


class TrainingDataError(ValueError):
    """Error caused by an invalid or incomplete training CSV."""


def read_training_csv(path: Path, target_column: str) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Load the six notebook features from a previously cleaned CSV."""
    try:
        text = path.read_bytes().decode("utf-8-sig")
        header = [value.strip() for value in next(csv.reader(io.StringIO(text)))]
    except FileNotFoundError as exc:
        raise TrainingDataError(f"Training CSV not found: {path}") from exc
    except (UnicodeDecodeError, csv.Error, StopIteration) as exc:
        raise TrainingDataError("The training file must be a valid UTF-8 CSV.") from exc

    duplicate_headers = sorted({name for name in header if name and header.count(name) > 1})
    if duplicate_headers:
        raise TrainingDataError("Duplicate columns: " + ", ".join(duplicate_headers))

    try:
        frame = pd.read_csv(io.StringIO(text))
    except (pd.errors.ParserError, ValueError) as exc:
        raise TrainingDataError("The training CSV could not be parsed.") from exc
    frame.columns = header

    missing_columns = [name for name in (*REQUIRED_FEATURES, target_column) if name not in frame.columns]
    if missing_columns:
        raise TrainingDataError("Required training columns are missing: " + ", ".join(missing_columns))

    features = frame.loc[:, REQUIRED_FEATURES].copy()
    for column in REQUIRED_FEATURES:
        normalized = features[column].astype("string").str.strip().str.replace(",", ".", regex=False)
        features[column] = pd.to_numeric(normalized, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    if features.isna().any().any():
        raise TrainingDataError(
            "The training features still contain missing values. "
            "Run pred19.data.preparation first."
        )

    target = pd.to_numeric(frame[target_column], errors="coerce")
    if target.isna().any() or set(target.unique().tolist()) != {0, 1}:
        raise TrainingDataError("The target column must contain both binary classes coded as 0 and 1.")

    quality = {
        "row_count": int(len(frame)),
        "feature_names": list(REQUIRED_FEATURES),
        "missing_feature_values": 0,
    }
    return features, target.astype(int), quality


def make_splits(X: pd.DataFrame, y: pd.Series) -> DataSplits:
    """Reproduce the notebook's stratified 80/20 split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return DataSplits(X_train, X_test, y_train, y_test)


def build_pipeline() -> Pipeline:
    """Build the RobustScaler and XGBoost pipeline used in the notebook."""
    preprocessor = ColumnTransformer(
        transformers=[("scaler", RobustScaler(), list(REQUIRED_FEATURES))],
        remainder="drop",
    )
    classifier = XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss")
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def fit_model(X_train: pd.DataFrame, y_train: pd.Series, n_jobs: int) -> tuple[Pipeline, dict, float]:
    search = GridSearchCV(
        estimator=build_pipeline(),
        param_grid={
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [3, 5, 7],
            "classifier__learning_rate": [0.01, 0.05, 0.1],
            "classifier__subsample": [0.8, 1.0],
        },
        cv=5,
        scoring="roc_auc",
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict, pd.DataFrame]:
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= DECISION_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()

    def ratio(numerator: int, denominator: int) -> float | None:
        return float(numerator / denominator) if denominator else None

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "sensitivity": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "negative_predictive_value": ratio(tn, tn + fn),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "decision_threshold": DECISION_THRESHOLD,
        "evaluation_sample_size": int(len(y_test)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    fpr, tpr, thresholds = roc_curve(y_test, probabilities, pos_label=1)
    return metrics, pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})


def _atomic_joblib_dump(value, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=destination.name, suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        joblib.dump(value, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text_write(text: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=destination.name, suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def train_and_export(data_path: Path, target_column: str, model_version: str, n_jobs: int) -> dict:
    X, y, data_quality = read_training_csv(data_path, target_column)
    splits = make_splits(X, y)
    model, best_params, cross_validation_auc = fit_model(splits.X_train, splits.y_train, n_jobs)
    metrics, roc = evaluate(model, splits.X_test, splits.y_test)

    trained_at = datetime.now(timezone.utc).isoformat()
    threshold_method = "Fixed at 0.4 in the original notebook"
    evaluation_method = (
        "Notebook reproduction: imputation performed on the cleaned dataset before a stratified 80/20 split; "
        "5-fold GridSearchCV on the training partition; final XGBoost evaluated on the test partition at 0.4."
    )
    metadata = {
        "model_version": model_version,
        "decision_threshold": DECISION_THRESHOLD,
        "feature_names": list(REQUIRED_FEATURES),
        "positive_class": 1,
        "trained_at_utc": trained_at,
        "threshold_selection_method": threshold_method,
        "best_parameters": best_params,
        "training_cross_validation_roc_auc": cross_validation_auc,
        "data_quality_summary": data_quality,
        "partition_sizes": {"training": int(len(splits.y_train)), "test": int(len(splits.y_test))},
        "library_versions": {
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        },
    }
    model.pred19_metadata = metadata
    model.decision_threshold_ = DECISION_THRESHOLD
    model.model_version_ = model_version

    metrics.update(
        {
            "evaluation_method": evaluation_method,
            "model_version": model_version,
            "evaluated_at_utc": trained_at,
            "threshold_selection_method": threshold_method,
            "training_cross_validation_roc_auc": cross_validation_auc,
            "best_parameters": best_params,
        }
    )

    _atomic_joblib_dump(model, MODEL_PATH)
    _atomic_text_write(json.dumps(metrics, indent=2, allow_nan=False) + "\n", METRICS_PATH)
    _atomic_text_write(roc.to_csv(index=False), ROC_PATH)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and export the XGBoost model from the notebook.")
    parser.add_argument("--data", type=Path, required=True, help="Path to data_clean.csv.")
    parser.add_argument("--target", default="target", help="Binary target column (default: target).")
    parser.add_argument("--model-version", default="notebook-xgb-1.0", help="Version stored with the model.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel workers used by GridSearchCV.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        metrics = train_and_export(args.data, args.target, args.model_version, args.n_jobs)
    except TrainingDataError as exc:
        print(f"Training data error: {exc}")
        return 2
    except Exception as exc:
        print(f"Training failed: {type(exc).__name__}: {exc}")
        return 1
    print("PRED19 artifacts exported.")
    print(f"Test sample size: {metrics['evaluation_sample_size']}")
    print(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Decision threshold: {metrics['decision_threshold']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

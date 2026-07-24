from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

from pred19.features import REQUIRED_FEATURES


DROP_COLUMNS = ("Unnamed: 0", "Suspect")
NUMERIC_COLUMNS = (
    "Age", "CA", "CK", "CREA", "ALP", "GGT", "GLU", "AST", "ALT", "LDH", "PCR",
    "KAL", "NAT", "UREA", "WBC", "RBC", "HGB", "HCT", "MCV", "MCH", "MCHC",
    "PLT1", "NE", "LY", "MO", "EO", "BA", "NET", "LYT", "MOT", "EOT", "BAT",
)
DROP_AFTER_CLEANING = ("CK", "UREA")
KNN_COLUMNS = ("ALP", "GGT", "BA", "NET", "LYT", "MOT", "NE", "LY", "BAT", "EOT", "MO", "EO")
MEAN_COLUMNS = (
    "LDH", "AST", "GLU", "PCR", "ALT", "CA", "KAL", "CREA", "NAT", "HCT", "HGB", "RBC",
    "WBC", "MCV", "MCH", "MCHC", "PLT1", "Age",
)


class DataPreparationError(ValueError):
    pass


def _atomic_write(text: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=destination.name, suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(text, encoding="utf-8", newline="")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _to_numeric(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.replace(",", ".", regex=False)
    converted = pd.to_numeric(normalized, errors="coerce")
    return converted.mask(converted.notna() & ~np.isfinite(converted)).astype("float64")


def prepare_data(source: Path, data_directory: Path, inference_directory: Path) -> dict:
    """Reproduce the notebook cleaning steps and write local CSV outputs."""
    if not source.is_file():
        raise DataPreparationError(f"Source CSV not found: {source}")
    try:
        frame = pd.read_csv(source)
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
        raise DataPreparationError("The source file could not be parsed as a UTF-8 CSV.") from exc

    duplicate_headers = frame.columns[frame.columns.duplicated()].tolist()
    if duplicate_headers:
        raise DataPreparationError("Duplicate headers: " + ", ".join(map(str, duplicate_headers)))
    frame.columns = [str(column).strip() for column in frame.columns]
    required = [*REQUIRED_FEATURES, "target"]
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        raise DataPreparationError("Required columns are missing: " + ", ".join(missing_columns))

    source_rows = len(frame)
    cleaned = frame.drop(columns=[column for column in DROP_COLUMNS if column in frame.columns]).copy()
    for column in NUMERIC_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = _to_numeric(cleaned[column])
    if "Sex" in cleaned.columns:
        cleaned["Sex"] = _to_numeric(cleaned["Sex"])
    converted_target = _to_numeric(cleaned["target"])
    targets = set(converted_target.dropna().unique().tolist())
    if targets != {0.0, 1.0} or converted_target.isna().any():
        raise DataPreparationError("The target column must contain only complete binary values 0 and 1.")
    cleaned["target"] = converted_target.astype("Int64")

    before_deduplication = len(cleaned)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    duplicates_removed = before_deduplication - len(cleaned)

    cleaned = cleaned.drop(columns=[column for column in DROP_AFTER_CLEANING if column in cleaned.columns])
    missing_before = {column: int(cleaned[column].isna().sum()) for column in REQUIRED_FEATURES}

    knn_columns = [column for column in KNN_COLUMNS if column in cleaned.columns]
    cleaned[knn_columns] = KNNImputer(n_neighbors=5).fit_transform(cleaned[knn_columns])
    for column in MEAN_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].fillna(cleaned[column].mean())
    if "Sex" in cleaned.columns and cleaned["Sex"].isna().any():
        cleaned["Sex"] = cleaned["Sex"].fillna(cleaned["Sex"].mode().iloc[0])

    if cleaned.loc[:, REQUIRED_FEATURES].isna().any().any():
        raise DataPreparationError("Notebook imputation did not produce complete model inputs.")

    inference = cleaned.loc[:, REQUIRED_FEATURES].copy()

    data_directory.mkdir(parents=True, exist_ok=True)
    inference_directory.mkdir(parents=True, exist_ok=True)
    clean_path = data_directory / "data_clean.csv"
    inference_path = inference_directory / "inference_input.csv"
    report_path = data_directory / "cleaning_report.json"
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_rows": int(source_rows),
        "clean_rows": int(len(cleaned)),
        "inference_rows": int(len(inference)),
        "exact_duplicate_rows_removed": int(duplicates_removed),
        "dropped_columns": [column for column in DROP_COLUMNS if column in frame.columns],
        "dropped_high_missingness_columns": list(DROP_AFTER_CLEANING),
        "inference_columns": list(REQUIRED_FEATURES),
        "missing_values_before_imputation": missing_before,
        "missing_values_after_imputation": {
            column: int(inference[column].isna().sum()) for column in REQUIRED_FEATURES
        },
        "imputation": "KNN (5 neighbors), mean, and mode as defined in medas.ipynb",
        "clean_output": str(clean_path),
        "inference_output": str(inference_path),
    }
    _atomic_write(cleaned.to_csv(index=False), clean_path)
    _atomic_write(inference.to_csv(index=False), inference_path)
    _atomic_write(json.dumps(report, indent=2, allow_nan=False) + "\n", report_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare private Pred19 training and inference CSV files.")
    parser.add_argument("--source", type=Path, default=Path("data/data.csv"))
    parser.add_argument("--data-directory", type=Path, default=Path("data"))
    parser.add_argument("--inference-directory", type=Path, default=Path("inference"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = prepare_data(args.source, args.data_directory, args.inference_directory)
    except DataPreparationError as exc:
        print(f"Data preparation failed: {exc}")
        return 2
    print("Data preparation completed without displaying clinical records.")
    print(f"Clean rows: {report['clean_rows']}")
    print(f"Inference rows: {report['inference_rows']}")
    print(f"Exact duplicates removed: {report['exact_duplicate_rows_removed']}")
    print("Imputation: KNN, mean, and mode as defined in medas.ipynb")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

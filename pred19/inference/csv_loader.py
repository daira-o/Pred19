"""Safe CSV parsing for inference inputs."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CSVLoadResult:
    dataframe: pd.DataFrame | None
    errors: tuple[str, ...]
    duplicate_columns: tuple[str, ...] = ()


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("The file must use UTF-8 text encoding.") from exc


def load_csv_bytes(data: bytes) -> CSVLoadResult:
    if not data:
        return CSVLoadResult(None, ("The uploaded file is empty.",))

    try:
        text = _decode(data)
        header = next(csv.reader(io.StringIO(text)))
    except (ValueError, csv.Error, StopIteration) as exc:
        return CSVLoadResult(None, (str(exc) or "The CSV header could not be read.",))

    normalized = [name.strip() for name in header]
    duplicates = tuple(sorted({name for name in normalized if name and normalized.count(name) > 1}))
    if duplicates:
        return CSVLoadResult(
            None,
            ("Duplicate CSV columns were found: " + ", ".join(duplicates) + ".",),
            duplicates,
        )

    try:
        frame = pd.read_csv(io.StringIO(text))
    except (pd.errors.ParserError, UnicodeError, ValueError) as exc:
        return CSVLoadResult(None, (f"The CSV could not be parsed: {exc}",))

    frame.columns = normalized
    if frame.empty:
        return CSVLoadResult(None, ("The CSV contains no patient records.",))
    return CSVLoadResult(frame, ())

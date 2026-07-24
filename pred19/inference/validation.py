from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pred19.features import REQUIRED_FEATURES


@dataclass(frozen=True)
class ValidationResult:
    dataframe: pd.DataFrame | None
    errors: tuple[str, ...]
    invalid_rows: tuple[int, ...]

    @property
    def is_valid(self) -> bool:
        return self.dataframe is not None and not self.errors


def validate_inputs(frame: pd.DataFrame) -> ValidationResult:
    missing_columns = [code for code in REQUIRED_FEATURES if code not in frame.columns]
    if missing_columns:
        return ValidationResult(
            dataframe=None,
            errors=("Required columns are missing: " + ", ".join(missing_columns) + ".",),
            invalid_rows=(),
        )

    validated = frame.copy()
    original = validated.loc[:, REQUIRED_FEATURES]
    numeric = pd.DataFrame(index=original.index)
    malformed = pd.DataFrame(False, index=original.index, columns=REQUIRED_FEATURES)
    for column in REQUIRED_FEATURES:
        source = original[column]
        normalized = source.astype("string").str.strip().str.replace(",", ".", regex=False)
        missing_source = source.isna() | normalized.isna() | normalized.eq("")
        converted_column = pd.to_numeric(normalized, errors="coerce")
        non_finite = converted_column.notna() & ~np.isfinite(converted_column)
        malformed[column] = (~missing_source & converted_column.isna()) | non_finite
        numeric[column] = converted_column.mask(non_finite)

    invalid_mask = malformed.any(axis=1) | numeric.isna().any(axis=1)
    invalid_rows = tuple((numeric.index[invalid_mask] + 1).tolist())
    # Assign column-by-column so pandas can replace strict string-extension
    # columns with numeric dtypes instead of attempting an unsafe in-place cast.
    for column in REQUIRED_FEATURES:
        validated[column] = numeric[column]

    return ValidationResult(
        dataframe=validated,
        errors=(),
        invalid_rows=invalid_rows,
    )

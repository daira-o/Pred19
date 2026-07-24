import pandas as pd

from pred19.inference.csv_loader import load_csv_bytes
from pred19.inference.validation import validate_inputs


VALID_HEADER = b"PCR,LDH,WBC,CA,HCT,EO\n"


def test_duplicate_columns_are_rejected():
    result = load_csv_bytes(b"PCR,LDH,WBC,CA,HCT,EO,PCR\n1,2,3,4,5,6,7\n")
    assert result.dataframe is None
    assert result.duplicate_columns == ("PCR",)


def test_missing_required_column_is_rejected():
    loaded = load_csv_bytes(b"PCR,LDH,WBC,CA,HCT\n1,2,3,4,5\n")
    result = validate_inputs(loaded.dataframe)
    assert result.dataframe is None
    assert "EO" in result.errors[0]


def test_non_numeric_record_is_marked_ineligible():
    loaded = load_csv_bytes(VALID_HEADER + b"1,2,3,4,5,6\n1,bad,3,4,5,6\n")
    result = validate_inputs(loaded.dataframe)
    assert result.errors == ()
    assert result.invalid_rows == (2,)


def test_missing_value_is_rejected_for_exported_notebook_model():
    loaded = load_csv_bytes(VALID_HEADER + b"1,2,,4,5,6\n")
    result = validate_inputs(loaded.dataframe)
    assert result.invalid_rows == (1,)
    assert pd.isna(result.dataframe.loc[0, "WBC"])


def test_decimal_comma_is_converted_to_numeric():
    loaded = load_csv_bytes(VALID_HEADER + b'"1,2",200,6.4,2.3,40,0.2\n')
    result = validate_inputs(loaded.dataframe)
    assert result.invalid_rows == ()
    assert float(result.dataframe.loc[0, "PCR"]) == 1.2


def test_numeric_strings_are_converted():
    loaded = load_csv_bytes(VALID_HEADER + b"1.2,200,6.4,2.3,40,0.2\n")
    result = validate_inputs(loaded.dataframe)
    assert result.is_valid
    assert result.invalid_rows == ()
    assert float(result.dataframe.loc[0, "PCR"]) == 1.2

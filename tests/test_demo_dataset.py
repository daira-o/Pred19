import pandas as pd

from pred19.features import REQUIRED_FEATURES
from pred19.inference.validation import validate_inputs
from pred19.settings import DEMO_DATA_PATH


def test_bundled_demo_dataset_is_present_and_processable():
    assert DEMO_DATA_PATH.is_file()
    frame = pd.read_csv(DEMO_DATA_PATH)
    assert len(frame) >= 1
    assert set(REQUIRED_FEATURES).issubset(frame.columns)
    assert {"SIMULATED_NAME", "SIMULATED_ID"}.issubset(frame.columns)

    validation = validate_inputs(frame)
    assert validation.errors == ()
    assert validation.invalid_rows == ()

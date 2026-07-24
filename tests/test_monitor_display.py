import pandas as pd

from pred19.features import FEATURES
from pred19.ui.components.input_cards import range_status
from pred19.ui.components.prediction_panel import risk_band


def test_risk_display_bands_cover_probability_range():
    assert risk_band(0.0)[0] == "Low"
    assert risk_band(0.25)[0] == "Moderate"
    assert risk_band(0.59)[0] == "Moderate"
    assert risk_band(0.60)[0] == "High"
    assert risk_band(1.0)[0] == "High"


def test_laboratory_status_uses_configured_reference():
    calcium = next(feature for feature in FEATURES if feature.code == "CA")
    assert range_status(2.0, calcium)[0] == "Low"
    assert range_status(2.4, calcium)[0] == "Normal"
    assert range_status(2.8, calcium)[0] == "Elevated"
    assert range_status(pd.NA, calcium)[0] == "No value"

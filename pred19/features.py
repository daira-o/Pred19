"""Canonical definitions for the six model input features."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureDefinition:
    code: str
    name: str
    group: str
    unit: str
    expected_range: str
    expected_min: float
    expected_max: float


FEATURES = (
    FeatureDefinition("PCR", "C-reactive protein", "Inflammation and cell damage", "mg/L", "0–10", 0.0, 10.0),
    FeatureDefinition("LDH", "Lactate dehydrogenase", "Inflammation and cell damage", "U/L", "120–246", 120.0, 246.0),
    FeatureDefinition("WBC", "White blood cell count", "Inflammation and cell damage", "×10⁹/L", "4–11", 4.0, 11.0),
    FeatureDefinition("CA", "Calcium", "Metabolic stability", "mmol/L", "2.2–2.6", 2.2, 2.6),
    FeatureDefinition("HCT", "Haematocrit", "Transport and rheology", "%", "36–55", 36.0, 55.0),
    FeatureDefinition("EO", "Eosinophils", "Immune regulation", "%", "0–6", 0.0, 6.0),
)

REQUIRED_FEATURES = tuple(feature.code for feature in FEATURES)

from pred19.features import FEATURES, REQUIRED_FEATURES


def test_feature_configuration_matches_notebook_inputs():
    assert tuple(feature.code for feature in FEATURES) == REQUIRED_FEATURES
    assert REQUIRED_FEATURES == ("PCR", "LDH", "WBC", "CA", "HCT", "EO")


def test_monitor_features_define_display_units_and_expected_ranges():
    for feature in FEATURES:
        assert feature.unit
        assert feature.expected_range
        assert feature.expected_min < feature.expected_max

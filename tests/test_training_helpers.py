from pathlib import Path

import pandas as pd

from pred19.features import REQUIRED_FEATURES
from pred19.modeling.training import (
    DECISION_THRESHOLD,
    build_pipeline,
    evaluate,
    make_splits,
    read_training_csv,
)


def test_training_reader_preserves_clean_rows_and_feature_order(tmp_path: Path):
    rows = []
    for index in range(120):
        target = index % 2
        rows.append(
            {
                "PCR": f"{1 + index / 100:.2f}".replace(".", ","),
                "LDH": 180 + index,
                "WBC": 5 + index / 100,
                "CA": 2.2,
                "HCT": 40 + target,
                "EO": 0.3 + target,
                "target": target,
            }
        )
    rows.append(rows[0].copy())
    path = tmp_path / "synthetic.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    X, y, quality = read_training_csv(path, "target")

    assert tuple(X.columns) == REQUIRED_FEATURES
    assert len(X) == 121
    assert len(y) == 121
    assert quality["row_count"] == 121
    assert X["PCR"].dtype.kind == "f"


def test_splits_are_disjoint_and_stratified():
    X = pd.DataFrame({column: range(200) for column in REQUIRED_FEATURES})
    y = pd.Series([0, 1] * 100)
    splits = make_splits(X, y)
    assert len(splits.X_train) == 160
    assert len(splits.X_test) == 40
    assert splits.y_train.mean() == splits.y_test.mean() == 0.5


def test_threshold_matches_notebook():
    assert DECISION_THRESHOLD == 0.4


def test_complete_pipeline_scales_and_predicts_complete_values():
    X = pd.DataFrame(
        {
            column: [float(index + offset) for index in range(80)]
            for offset, column in enumerate(REQUIRED_FEATURES)
        }
    )
    y = pd.Series([0, 1] * 40)
    model = build_pipeline().set_params(
        classifier__n_estimators=5,
        classifier__max_depth=2,
        classifier__learning_rate=0.1,
        classifier__subsample=1.0,
    )
    model.fit(X, y)
    assert tuple(model.feature_names_in_) == REQUIRED_FEATURES
    probabilities = model.predict_proba(X.iloc[[3]])
    assert probabilities.shape == (1, 2)
    metrics, roc = evaluate(model, X, y)
    assert metrics["decision_threshold"] == 0.4
    assert 0 <= metrics["roc_auc"] <= 1
    assert list(roc.columns) == ["fpr", "tpr", "threshold"]

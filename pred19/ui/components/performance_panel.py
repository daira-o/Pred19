"""Population-level model-performance presentation."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def _display_metric(label: str, value, percent: bool = False) -> None:
    if value is None:
        st.metric(label, "Not available")
        return
    try:
        number = float(value)
        st.metric(label, f"{number:.1%}" if percent else f"{number:g}")
    except (TypeError, ValueError):
        st.metric(label, str(value))


def render_performance(artifacts) -> None:
    st.subheader("Reproduced test-set evaluation")
    st.caption("These results describe the current model on the notebook's holdout split, not the selected record.")
    if not artifacts.available:
        st.warning("Model-performance artifacts are not available.")
        return

    metrics = artifacts.metrics
    first = st.columns(4)
    with first[0]:
        _display_metric("ROC-AUC", metrics.get("roc_auc"), True)
    with first[1]:
        _display_metric("Sensitivity", metrics.get("sensitivity"), True)
    with first[2]:
        _display_metric("Specificity", metrics.get("specificity"), True)
    with first[3]:
        _display_metric("Decision threshold", metrics.get("decision_threshold"))

    roc = artifacts.roc_curve.loc[:, ["fpr", "tpr"]].copy()
    roc["series"] = "ROC curve"
    baseline = pd.DataFrame(
        {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0], "series": ["No-discrimination", "No-discrimination"]}
    )
    chart_data = pd.concat([roc, baseline], ignore_index=True)
    st.vega_lite_chart(
        chart_data,
        {
            "height": 360,
            "mark": {"type": "line", "strokeWidth": 3},
            "encoding": {
                "x": {
                    "field": "fpr",
                    "type": "quantitative",
                    "title": "False positive rate",
                    "scale": {"domain": [0, 1]},
                },
                "y": {
                    "field": "tpr",
                    "type": "quantitative",
                    "title": "True positive rate",
                    "scale": {"domain": [0, 1]},
                },
                "color": {
                    "field": "series",
                    "type": "nominal",
                    "scale": {
                        "domain": ["ROC curve", "No-discrimination"],
                        "range": ["#087e8b", "#80909d"],
                    },
                    "legend": {"title": None, "orient": "bottom-right"},
                },
                "strokeDash": {
                    "field": "series",
                    "type": "nominal",
                    "scale": {
                        "domain": ["ROC curve", "No-discrimination"],
                        "range": [[1, 0], [6, 6]],
                    },
                    "legend": None,
                },
                "tooltip": [
                    {"field": "series", "type": "nominal", "title": "Series"},
                    {"field": "fpr", "type": "quantitative", "format": ".3f", "title": "FPR"},
                    {"field": "tpr", "type": "quantitative", "format": ".3f", "title": "TPR"},
                ],
            },
        },
        width="stretch",
    )

    details = st.columns(2)
    details[0].markdown(f"**Evaluation sample size**  \n{metrics.get('evaluation_sample_size', 'Not available')}")
    details[1].markdown(f"**Evaluation method**  \n{metrics.get('evaluation_method', 'Not available')}")

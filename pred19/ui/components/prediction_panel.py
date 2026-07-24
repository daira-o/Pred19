"""Risk-score presentation for a single inference result."""

import html

import streamlit as st


def risk_band(probability: float) -> tuple[str, str, str]:
    if probability < 0.25:
        return (
            "Low",
            "low",
            "The model score is in the lower display band for this observation.",
        )
    if probability < 0.60:
        return (
            "Moderate",
            "moderate",
            "The model score is in the intermediate display band and warrants contextual review.",
        )
    return (
        "High",
        "high",
        "The model score is in the upper display band and should be reviewed with the full clinical context.",
    )


def render_prediction(result) -> None:
    probability_pct = result.probability * 100
    category, category_class, interpretation = risk_band(result.probability)
    marker_position = min(max(probability_pct, 1.0), 99.0)
    st.markdown('<div id="risk" class="pred19-section-anchor"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="pred19-risk-panel">'
        '<div>'
        '<div class="pred19-risk-eyebrow">Estimated COVID-19 risk</div>'
        f'<div class="pred19-risk-number">{probability_pct:.1f}<small>%</small></div>'
        f'<span class="pred19-risk-category pred19-risk-{category_class}">{category} risk</span>'
        '</div>'
        '<div>'
        '<div class="pred19-gauge-labels"><span>Low</span><span>Moderate</span><span>High</span></div>'
        f'<div class="pred19-gauge" role="img" aria-label="Risk score {probability_pct:.1f} percent">'
        f'<span class="pred19-gauge-marker" style="left:calc({marker_position:.1f}% - 2px)"></span></div>'
        f'<div class="pred19-risk-copy">{html.escape(interpretation)}</div>'
        f'<div class="pred19-risk-time">Last assessment · '
        f'{result.timestamp.strftime("%d %b %Y at %H:%M:%S UTC")}</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

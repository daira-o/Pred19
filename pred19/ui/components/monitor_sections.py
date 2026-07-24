import html

import pandas as pd
import streamlit as st

from pred19.ui.components.input_cards import abnormal_features


def render_explanation(row: pd.Series, category: str | None) -> None:
    deviations = abnormal_features(row)
    context = (
        ", ".join(deviations)
        if deviations
        else "No configured reference-range deviations are visible"
    )
    category_text = category or "Unavailable"
    st.markdown(
        '<div id="explanation" class="pred19-section-anchor"></div>'
        '<div class="pred19-section-heading"><h2>Result explanation</h2>'
        '<p>What the displayed score means</p></div>'
        '<div class="pred19-explanation">'
        f'<strong>{html.escape(category_text)} display band.</strong> '
        'The score is calculated from PCR, LDH, WBC, CA, HCT and EO. '
        f'{html.escape(context)}. Reference-range status provides context only and is not a '
        'patient-level explanation of the model prediction.'
        '</div>',
        unsafe_allow_html=True,
    )
    with st.expander("How to read this monitor"):
        st.write(
            "The percentage is the model output for the positive class. Low, Moderate and High are "
            "display bands for this demo; they are not diagnostic categories. The exported model does "
            "not include patient-level feature attribution, so the interface does not claim that an "
            "individual laboratory value caused the result."
        )


def render_clinical_disclaimer() -> None:
    st.markdown(
        '<div class="pred19-disclaimer"><strong>Clinical demo.</strong> '
        'PRED19 is a research demonstration and not a validated medical device. The displayed score '
        'does not establish a diagnosis or recommend treatment. Laboratory intervals vary by method '
        'and institution and must be interpreted in clinical context.</div>',
        unsafe_allow_html=True,
    )

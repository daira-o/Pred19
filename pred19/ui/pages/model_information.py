import streamlit as st

from pred19.features import FEATURES
from pred19.inference.artifacts import load_performance_artifacts
from pred19.ui.components.sidebar import render_page_header, render_sidebar


render_sidebar()
render_page_header(
    "Governance and scope",
    "Model Information",
    "Technical boundaries and intended use of the PRED19 synthetic demonstration.",
)
st.subheader("Intended use")
st.write(
    "Pred19 provides a probability-based research output from six laboratory inputs. It is not a "
    "validated medical device, does not establish a diagnosis, and does not provide treatment recommendations."
)

st.subheader("Required inputs")
for feature in FEATURES:
    st.markdown(
        f"- **{feature.name}** (`{feature.code}`): {feature.expected_range} {feature.unit}"
    )
st.caption(
    "The units and adult display intervals configure the simulated monitor; "
    "they are not model inputs and should be replaced by the reporting laboratory's intervals."
)
st.markdown(
    "Display references: [MedlinePlus](https://medlineplus.gov/lab-tests/) · "
    "[NHS haematology ranges](https://www.gloshospitals.nhs.uk/our-services/"
    "services-we-offer/pathology/haematology/haematology-reference-ranges/) · "
    "[NHS biochemistry test directory](https://www.cuh.nhs.uk/our-services/pathology/"
    "pathology-tests/pathology-tests-a-to-z/biochemistry-tests/blood-tests-l-q/)"
)

artifacts = load_performance_artifacts()
st.subheader("Version and evaluation status")
if artifacts.metrics:
    st.write(f"Model version: **{artifacts.metrics.get('model_version', 'Not specified')}**")
    st.write(f"Evaluation method: **{artifacts.metrics.get('evaluation_method', 'Not available')}**")
else:
    st.warning("Model-performance artifacts are not available.")

st.subheader("Interpretation limits")
st.write(
    "The probability is the XGBoost positive-class output at the notebook's 0.4 decision threshold. "
    "The dashboard does not present laboratory values as causes or as patient-level model explanations."
)

from __future__ import annotations

import pandas as pd
import streamlit as st

from pred19.inference.artifacts import (
    load_performance_artifacts,
    resolve_model_version,
    resolve_threshold,
)
from pred19.inference.csv_loader import load_csv_bytes
from pred19.inference.model import ModelLoadError, load_model
from pred19.inference.prediction import PredictionError, predict_record
from pred19.inference.validation import validate_inputs
from pred19.settings import DEMO_DATA_PATH, MAX_UPLOAD_MB, TIMESTAMP_COLUMNS
from pred19.ui.components.input_cards import render_input_cards
from pred19.ui.components.monitor_sections import render_clinical_disclaimer, render_explanation
from pred19.ui.components.patient_identity import render_patient_identity
from pred19.ui.components.patient_selector import select_patient
from pred19.ui.components.prediction_panel import render_prediction, risk_band
from pred19.ui.components.sidebar import render_monitor_navigation, render_page_header, render_sidebar
from pred19.ui.components.validation_panel import render_data_quality, render_validation


render_sidebar()
render_page_header(
    "Clinical assistance",
    "PRED19 Patient Monitor",
    "Clinical overview of one observation at a time",
)
render_monitor_navigation()

with st.sidebar:
    st.divider()
    st.markdown('<div class="pred19-sidebar-label">Patient data</div>', unsafe_allow_html=True)
    use_demo = st.toggle(
        "Synthetic dataset",
        value=True,
        help="Use the bundled non-clinical patient records.",
    )
    uploaded = None
    with st.expander("Upload CSV", expanded=not use_demo):
        if use_demo:
            st.caption("Turn off the synthetic dataset to upload observations.")
        else:
            uploaded = st.file_uploader(
                "CSV observations",
                type=("csv",),
                help="Required columns: PCR, LDH, WBC, CA, HCT and EO.",
                label_visibility="collapsed",
            )
            st.caption(f"CSV · Up to {MAX_UPLOAD_MB} MB")

if not use_demo and uploaded is None:
    st.markdown(
        '<div class="pred19-empty"><div class="pred19-monitor-title">Monitor ready</div>'
        '<strong>No patient observations loaded</strong>'
        '<p>Activate the synthetic dataset or open Upload CSV in the sidebar.</p></div>',
        unsafe_allow_html=True,
    )
    st.stop()

if use_demo:
    if not DEMO_DATA_PATH.is_file():
        st.error("The bundled synthetic dataset is not available.")
        st.stop()
    data = DEMO_DATA_PATH.read_bytes()
    source_label = "Synthetic demo"
else:
    data = uploaded.getvalue()
    source_label = uploaded.name

if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
    render_validation((f"The file is larger than the {MAX_UPLOAD_MB} MB limit.",))
    st.stop()

loaded = load_csv_bytes(data)
if loaded.errors:
    render_validation(loaded.errors)
    st.stop()

validation = validate_inputs(loaded.dataframe)
if validation.dataframe is None:
    render_validation(validation.errors, validation.invalid_rows)
    st.stop()

frame = validation.dataframe
invalid_positions = {number - 1 for number in validation.invalid_rows}
eligible = [position for position in range(len(frame)) if position not in invalid_positions]

with st.sidebar:
    with st.container(key="patient_selector"):
        st.markdown('<div class="pred19-sidebar-label">Patient</div>', unsafe_allow_html=True)
        position = select_patient(frame, eligible, allow_simulated_names=use_demo)
if position is None:
    st.stop()

row = frame.iloc[position]
observation_timestamp = None
for column in frame.columns:
    if str(column).strip().lower() in TIMESTAMP_COLUMNS and not pd.isna(row[column]):
        observation_timestamp = str(row[column])
        break

prediction_result = None
prediction_message = None
try:
    model = load_model()
    artifacts = load_performance_artifacts()
    threshold = resolve_threshold(model, artifacts.metrics)
    if threshold is None:
        prediction_message = "The model decision threshold is unavailable."
    else:
        version = resolve_model_version(model, artifacts.metrics)
        prediction_result = predict_record(model, row, threshold, version)
except (ModelLoadError, PredictionError) as exc:
    prediction_message = str(exc)

render_patient_identity(
    row,
    position,
    simulated=use_demo,
    evaluated_at=prediction_result.timestamp if prediction_result else None,
    analysis_available=prediction_result is not None,
)

if prediction_result is not None:
    render_prediction(prediction_result)
    risk_category = risk_band(prediction_result.probability)[0]
else:
    st.markdown('<div id="risk" class="pred19-section-anchor"></div>', unsafe_allow_html=True)
    st.warning(prediction_message)
    risk_category = None

render_input_cards(row, observation_timestamp)
render_data_quality(
    source_label=source_label,
    invalid_rows=validation.invalid_rows,
    selected_record_valid=position not in invalid_positions,
)
with st.expander("Validation details", expanded=bool(validation.invalid_rows)):
    render_validation(validation.errors, validation.invalid_rows, show_heading=False)

render_explanation(row, risk_category)
render_clinical_disclaimer()

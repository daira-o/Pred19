"""Compact patient identity header for the demonstration monitor."""

from __future__ import annotations

import html
from datetime import datetime

import pandas as pd
import streamlit as st


def _text(row: pd.Series, column: str, fallback: str = "Not documented") -> str:
    value = row.get(column)
    if value is None or pd.isna(value) or not str(value).strip():
        return fallback
    return str(value).strip()


def patient_display(row: pd.Series, position: int, simulated: bool) -> dict[str, str]:
    if simulated:
        return {
            "name": _text(row, "SIMULATED_NAME", f"Simulated patient {position + 1}"),
            "patient_id": _text(row, "SIMULATED_ID", f"SIM-{position + 1:03d}"),
            "age": _text(row, "SIMULATED_AGE"),
            "sex": _text(row, "SIMULATED_SEX"),
            "department": _text(row, "SIMULATED_DEPARTMENT"),
        }
    return {
        "name": f"Record {position + 1}",
        "patient_id": "Non-identifying record",
        "age": "Not documented",
        "sex": "Not documented",
        "department": "Uploaded observation",
    }


def render_patient_identity(
    row: pd.Series,
    position: int,
    simulated: bool,
    evaluated_at: datetime | None,
    analysis_available: bool,
) -> None:
    patient = patient_display(row, position, simulated)
    badge = (
        '<span class="pred19-badge pred19-badge-synthetic">Synthetic</span>'
        if simulated
        else ""
    )
    status_class = "complete" if analysis_available else "unavailable"
    status_label = "Analysis complete" if analysis_available else "Analysis unavailable"
    evaluation = (
        evaluated_at.strftime("%d %b %Y · %H:%M UTC")
        if evaluated_at is not None
        else "Not available"
    )
    st.markdown(
        '<div class="pred19-clinical-header">'
        '<div class="pred19-patient-primary">'
        f'<div class="pred19-patient-name">{badge}{html.escape(patient["name"])}</div>'
        f'<div class="pred19-patient-subline">{html.escape(patient["patient_id"])}'
        f' · {html.escape(patient["department"])}</div></div>'
        '<div class="pred19-header-facts">'
        f'<div class="pred19-header-fact"><span>Patient</span>'
        f'<strong>{html.escape(patient["age"])} yr · {html.escape(patient["sex"])}</strong></div>'
        f'<div class="pred19-header-fact"><span>Evaluated</span><strong>{evaluation}</strong></div>'
        f'<span class="pred19-badge pred19-badge-{status_class}">{status_label}</span>'
        '</div></div>',
        unsafe_allow_html=True,
    )

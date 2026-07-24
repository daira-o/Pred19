from __future__ import annotations

import html
from collections import OrderedDict

import pandas as pd
import streamlit as st

from pred19.features import FEATURES


def _format_value(value) -> str:
    if pd.isna(value):
        return "Missing"
    return f"{float(value):,.3f}".rstrip("0").rstrip(".")


def range_status(value, feature) -> tuple[str, str]:
    if pd.isna(value):
        return "No value", "unknown"
    if feature.expected_min is None or feature.expected_max is None:
        return "No reference", "unknown"
    numeric = float(value)
    if numeric < feature.expected_min:
        return "Low", "low"
    if numeric > feature.expected_max:
        return "Elevated", "high"
    return "Normal", "normal"


def _lab_card(row: pd.Series, feature, show_group: bool = False) -> str:
    value = row[feature.code]
    status, status_class = range_status(value, feature)
    unit = feature.unit or "Not documented"
    reference = (
        f"{feature.expected_range} {feature.unit}"
        if feature.expected_range and feature.unit
        else "Not documented"
    )
    group_line = (
        f'<div class="pred19-lab-group">{html.escape(feature.group)}</div>'
        if show_group
        else ""
    )
    return (
        '<div class="pred19-lab-card">'
        f"{group_line}"
        '<div class="pred19-lab-head">'
        f'<div class="pred19-lab-code">{html.escape(feature.code)}</div>'
        f'<span class="pred19-status pred19-status-{status_class}">{status}</span></div>'
        f'<div class="pred19-lab-name" title="{html.escape(feature.name)}">'
        f'{html.escape(feature.name)}</div>'
        '<div class="pred19-lab-reading">'
        f'<span class="pred19-lab-value">{_format_value(value)}</span>'
        f'<span class="pred19-lab-unit">{html.escape(unit)}</span></div>'
        '<div class="pred19-lab-reference"><span>Reference</span>'
        f'<strong>{html.escape(reference)}</strong></div></div>'
    )


def abnormal_features(row: pd.Series) -> list[str]:
    abnormal = []
    for feature in FEATURES:
        status, _ = range_status(row[feature.code], feature)
        if status in {"Low", "Elevated"}:
            abnormal.append(f"{feature.code} ({status.lower()})")
    return abnormal


def render_input_cards(row: pd.Series, observation_timestamp: str | None) -> None:
    groups: OrderedDict[str, list] = OrderedDict()
    for feature in FEATURES:
        groups.setdefault(feature.group, []).append(feature)

    st.markdown(
        '<div id="laboratory-values" class="pred19-section-anchor"></div>'
        '<div class="pred19-section-heading"><h2>Laboratory values</h2>'
        '<p>Six inputs used by the prediction pipeline</p></div>',
        unsafe_allow_html=True,
    )
    primary_group = "Inflammation and cell damage"
    primary_features = groups.pop(primary_group, [])
    if primary_features:
        st.markdown(
            f'<div class="pred19-group-title">{primary_group}</div>'
            '<div class="pred19-lab-grid">'
            + "".join(_lab_card(row, feature) for feature in primary_features)
            + "</div>",
            unsafe_allow_html=True,
        )
    secondary_features = [feature for features in groups.values() for feature in features]
    if secondary_features:
        st.markdown(
            '<div class="pred19-lab-grid">'
            + "".join(_lab_card(row, feature, show_group=True) for feature in secondary_features)
            + "</div>",
            unsafe_allow_html=True,
        )

    if observation_timestamp:
        st.caption(f"Observation timestamp: {observation_timestamp}")

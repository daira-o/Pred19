"""Validation and data-quality presentation components."""

import html

import streamlit as st


def render_validation(
    errors: tuple[str, ...],
    invalid_rows: tuple[int, ...] = (),
    show_heading: bool = True,
) -> None:
    if show_heading:
        st.subheader("Input validation")
    if errors:
        for error in errors:
            st.error(error)
        return
    if invalid_rows:
        preview = ", ".join(f"Record {number}" for number in invalid_rows[:10])
        suffix = "…" if len(invalid_rows) > 10 else ""
        st.warning(
            f"{len(invalid_rows)} record(s) cannot be processed because a required value is missing or "
            f"non-numeric: {preview}{suffix}."
        )
        return
    st.success("All required columns and values passed validation.")


def render_data_quality(
    *,
    source_label: str,
    invalid_rows: tuple[int, ...],
    selected_record_valid: bool,
) -> None:
    valid_label = "Complete" if selected_record_valid else "Review required"
    excluded = str(len(invalid_rows))
    st.markdown(
        '<div id="data-quality" class="pred19-section-anchor"></div>'
        '<div class="pred19-section-heading"><h2>Data quality</h2>'
        '<p>Validation status for the selected observation</p></div>'
        '<div class="pred19-quality-grid">'
        '<div class="pred19-compact-panel"><span>Selected record</span>'
        f'<strong>{valid_label}</strong><p>All six model inputs are numeric and available.</p></div>'
        '<div class="pred19-compact-panel"><span>Dataset source</span>'
        f'<strong>{html.escape(source_label)}</strong><p>Used only for the current session.</p></div>'
        '<div class="pred19-compact-panel"><span>Excluded records</span>'
        f'<strong>{excluded}</strong><p>Rows unavailable for inference after validation.</p></div>'
        '</div>',
        unsafe_allow_html=True,
    )

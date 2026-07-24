"""Patient-record selector used by the Streamlit presentation layer."""

import streamlit as st


def select_patient(frame, eligible_positions: list[int], allow_simulated_names: bool = False) -> int | None:
    if not eligible_positions:
        st.warning("No records contain a processable set of model inputs.")
        return None
    simulated_names = (
        frame["SIMULATED_NAME"]
        if allow_simulated_names and "SIMULATED_NAME" in frame.columns
        else None
    )
    labels = {}
    for position in eligible_positions:
        name = str(simulated_names.iloc[position]).strip() if simulated_names is not None else ""
        if name and name.lower() not in {"nan", "<na>"}:
            labels[position] = name
        else:
            labels[position] = f"Record {position + 1}"
    return st.selectbox(
        "Select patient",
        options=eligible_positions,
        format_func=lambda position: labels[position],
        help="Record numbers are generated from row position and contain no patient identifiers.",
    )

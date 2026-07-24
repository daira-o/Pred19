"""Shared navigation and page-heading components."""

import streamlit as st


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<div class="pred19-brand"><strong>PRED19</strong>'
            '<span>Clinical Assistance Monitor</span></div>',
            unsafe_allow_html=True,
        )


def render_page_header(kicker: str, title: str, description: str) -> None:
    st.markdown(
        f'<div class="pred19-monitor-top"><div>'
        f'<div class="pred19-monitor-title">{kicker}</div>'
        f'<h1 style="margin:.1rem 0;font-size:1.65rem">{title}</h1>'
        f'<div style="color:#627382;font-size:.8rem">{description}</div></div></div>',
        unsafe_allow_html=True,
    )


def render_monitor_navigation() -> None:
    st.markdown(
        '<div class="pred19-jump-nav" aria-label="Monitor sections">'
        '<a href="#risk">Risk</a>'
        '<a href="#laboratory-values">Laboratory values</a>'
        '<a href="#data-quality">Data quality</a>'
        '<a href="#explanation">Explanation</a>'
        '</div>',
        unsafe_allow_html=True,
    )

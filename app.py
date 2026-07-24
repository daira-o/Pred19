import streamlit as st

from pred19.settings import APP_TITLE
from pred19.ui.components.styles import render_global_styles


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="✚",
    layout="wide",
    initial_sidebar_state="auto",
)
render_global_styles()

pages = [
    st.Page("pred19/ui/pages/patient_monitor.py", title="Patient Monitor", icon="🩺", default=True),
    st.Page("pred19/ui/pages/model_performance.py", title="Model Performance", icon="📈"),
    st.Page("pred19/ui/pages/model_information.py", title="Model Information", icon="ℹ️"),
]

st.navigation(pages).run()

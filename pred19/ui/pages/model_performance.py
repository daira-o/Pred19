from pred19.inference.artifacts import load_performance_artifacts
from pred19.ui.components.performance_panel import render_performance
from pred19.ui.components.sidebar import render_page_header, render_sidebar


render_sidebar()
render_page_header(
    "Notebook reproduction",
    "Model Performance",
    "Current exported-model results on the notebook's holdout split, separate from individual inference.",
)
render_performance(load_performance_artifacts())

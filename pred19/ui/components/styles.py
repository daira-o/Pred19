"""Centralized Streamlit presentation styles."""

import streamlit as st


def render_global_styles() -> None:
    st.markdown(
        """
        <style>
          :root {
            --p19-ink:#142536;
            --p19-muted:#627382;
            --p19-line:#dfe6eb;
            --p19-bg:#f4f7f9;
            --p19-panel:#ffffff;
            --p19-teal:#087e8b;
            --p19-teal-dark:#075e68;
            --p19-teal-soft:#e4f2f1;
            --p19-green:#287a57;
            --p19-amber:#ae6d14;
            --p19-red:#b74545;
          }
          html { scroll-behavior:smooth; }
          .stApp { background:var(--p19-bg); color:var(--p19-ink); }
          .block-container {
            max-width:1180px;
            padding-top:.75rem;
            padding-bottom:3rem;
          }
          h1, h2, h3 { color:var(--p19-ink); letter-spacing:-.025em; }
          h2 { margin-top:2.2rem; }
          section[data-testid="stSidebar"] {
            background:#102a3c;
            min-width:270px;
            max-width:270px;
          }
          section[data-testid="stSidebar"] * { color:#f5fafc; }
          section[data-testid="stSidebar"] hr { margin:.65rem 0; }
          section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
            background:rgba(255,255,255,.07);
            border:1px dashed rgba(255,255,255,.32);
            border-radius:10px;
          }
          section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
            background:#fff;
          }
          section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button * {
            color:var(--p19-ink) !important;
          }
          .st-key-patient_selector input {
            color:var(--p19-ink) !important;
            -webkit-text-fill-color:var(--p19-ink) !important;
          }
          .pred19-brand { padding:.15rem 0 .4rem; }
          .pred19-brand strong { display:block; font-size:1.08rem; letter-spacing:.04em; }
          .pred19-brand span { color:#b9c9d3; font-size:.78rem; }
          .pred19-sidebar-label {
            color:#b9c9d3;
            font-size:.68rem;
            font-weight:800;
            letter-spacing:.09em;
            text-transform:uppercase;
            margin:.3rem 0 .1rem;
          }
          .pred19-monitor-top {
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap:1rem;
            margin:.15rem 0 .45rem;
          }
          .pred19-monitor-title {
            color:var(--p19-teal-dark);
            font-size:.77rem;
            font-weight:850;
            letter-spacing:.12em;
            text-transform:uppercase;
          }
          .pred19-jump-nav { display:flex; flex-wrap:wrap; gap:.35rem; margin-bottom:.55rem; }
          .pred19-jump-nav a {
            color:#4f6574 !important;
            text-decoration:none;
            background:#e9eef2;
            border-radius:999px;
            padding:.28rem .62rem;
            font-size:.7rem;
            font-weight:750;
          }
          .pred19-clinical-header {
            position:sticky;
            top:.35rem;
            z-index:20;
            display:flex;
            justify-content:space-between;
            align-items:center;
            gap:1rem;
            background:rgba(255,255,255,.96);
            border:1px solid var(--p19-line);
            border-radius:14px;
            padding:12px 16px;
            box-shadow:0 8px 24px rgba(20,37,54,.08);
            backdrop-filter:blur(10px);
          }
          .pred19-patient-primary { min-width:0; }
          .pred19-patient-name {
            color:var(--p19-ink);
            font-size:1.15rem;
            font-weight:850;
            white-space:nowrap;
            overflow:hidden;
            text-overflow:ellipsis;
          }
          .pred19-patient-subline {
            color:var(--p19-muted);
            font-size:.77rem;
            margin-top:.18rem;
          }
          .pred19-header-facts {
            display:flex;
            align-items:center;
            justify-content:flex-end;
            gap:1rem;
            flex-wrap:wrap;
          }
          .pred19-header-fact span {
            display:block;
            color:#7b8a96;
            font-size:.61rem;
            font-weight:800;
            letter-spacing:.07em;
            text-transform:uppercase;
          }
          .pred19-header-fact strong { font-size:.79rem; }
          .pred19-badge {
            display:inline-flex;
            align-items:center;
            gap:.28rem;
            border-radius:999px;
            padding:.23rem .5rem;
            font-size:.65rem;
            font-weight:850;
          }
          .pred19-badge::before {
            content:"";
            width:6px;
            height:6px;
            border-radius:50%;
            background:currentColor;
          }
          .pred19-badge-synthetic { background:#e3f1ee; color:#176957; }
          .pred19-badge-complete { background:#e4f2ec; color:#287a57; }
          .pred19-badge-unavailable { background:#f6e9e9; color:#a33d3d; }
          .pred19-section-anchor { scroll-margin-top:6rem; }
          .pred19-risk-panel {
            display:grid;
            grid-template-columns:minmax(210px,.8fr) minmax(300px,1.45fr);
            gap:1.5rem;
            align-items:center;
            background:var(--p19-panel);
            border:1px solid var(--p19-line);
            border-radius:18px;
            padding:22px 24px;
            box-shadow:0 10px 28px rgba(20,37,54,.07);
            margin:.8rem 0 1.1rem;
          }
          .pred19-risk-eyebrow {
            color:var(--p19-muted);
            font-size:.69rem;
            font-weight:800;
            letter-spacing:.08em;
            text-transform:uppercase;
          }
          .pred19-risk-number {
            color:var(--p19-ink);
            font-size:3.15rem;
            font-weight:850;
            letter-spacing:-.055em;
            line-height:1;
            margin:.35rem 0;
          }
          .pred19-risk-number small { font-size:1.2rem; letter-spacing:0; }
          .pred19-risk-category {
            display:inline-block;
            border-radius:7px;
            padding:.3rem .65rem;
            font-size:.76rem;
            font-weight:850;
          }
          .pred19-risk-low { background:#e5f2ed; color:var(--p19-green); }
          .pred19-risk-moderate { background:#fff1dc; color:var(--p19-amber); }
          .pred19-risk-high { background:#f8e5e5; color:var(--p19-red); }
          .pred19-gauge {
            position:relative;
            height:14px;
            overflow:hidden;
            border-radius:999px;
            background:linear-gradient(90deg,#5aa77f 0 25%,#e8b655 25% 60%,#d85f5f 60% 100%);
            margin:.45rem 0 1rem;
          }
          .pred19-gauge-marker {
            position:absolute;
            top:-5px;
            width:4px;
            height:24px;
            border-radius:3px;
            background:#12263a;
            box-shadow:0 0 0 3px #fff;
          }
          .pred19-gauge-labels {
            display:flex;
            justify-content:space-between;
            color:#748491;
            font-size:.64rem;
            font-weight:700;
          }
          .pred19-risk-copy {
            color:#3f5362;
            font-size:.87rem;
            line-height:1.55;
            margin:.45rem 0 .55rem;
          }
          .pred19-risk-time { color:#748491; font-size:.72rem; }
          .pred19-section-heading {
            display:flex;
            align-items:flex-end;
            justify-content:space-between;
            gap:1rem;
            margin:2.2rem 0 .75rem;
          }
          .pred19-section-heading h2 { font-size:1.25rem; margin:0; }
          .pred19-section-heading p {
            color:var(--p19-muted);
            font-size:.77rem;
            margin:0;
            text-align:right;
          }
          .pred19-group-title {
            color:#536979;
            font-size:.67rem;
            font-weight:850;
            letter-spacing:.08em;
            text-transform:uppercase;
            margin:.45rem 0 .35rem;
          }
          .pred19-lab-grid {
            display:grid;
            grid-template-columns:repeat(3,minmax(0,1fr));
            gap:.65rem;
            margin-bottom:.8rem;
          }
          .pred19-lab-card {
            background:var(--p19-panel);
            border:1px solid var(--p19-line);
            border-radius:12px;
            padding:12px 13px;
            min-width:0;
          }
          .pred19-lab-group {
            color:#7a8994;
            font-size:.56rem;
            font-weight:850;
            letter-spacing:.06em;
            text-transform:uppercase;
            white-space:nowrap;
            overflow:hidden;
            text-overflow:ellipsis;
            margin-bottom:.28rem;
          }
          .pred19-lab-head {
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap:.5rem;
          }
          .pred19-lab-code {
            color:var(--p19-teal-dark);
            font-size:.69rem;
            font-weight:900;
            letter-spacing:.06em;
          }
          .pred19-lab-name {
            color:#526575;
            font-size:.71rem;
            white-space:nowrap;
            overflow:hidden;
            text-overflow:ellipsis;
            margin-top:.12rem;
          }
          .pred19-lab-reading {
            display:flex;
            align-items:baseline;
            gap:.3rem;
            margin:.35rem 0;
          }
          .pred19-lab-value { font-size:1.38rem; font-weight:850; color:var(--p19-ink); }
          .pred19-lab-unit { font-size:.67rem; font-weight:700; color:#71818e; }
          .pred19-lab-reference {
            display:flex;
            justify-content:space-between;
            gap:.4rem;
            padding-top:.35rem;
            border-top:1px solid #edf1f3;
            color:#748491;
            font-size:.65rem;
          }
          .pred19-lab-reference strong { color:#435765; text-align:right; }
          .pred19-status {
            display:inline-flex;
            align-items:center;
            gap:.26rem;
            border-radius:999px;
            padding:.2rem .45rem;
            font-size:.61rem;
            font-weight:850;
            white-space:nowrap;
          }
          .pred19-status::before {
            content:"";
            width:5px;
            height:5px;
            border-radius:50%;
            background:currentColor;
          }
          .pred19-status-normal { background:#e5f2ed; color:var(--p19-green); }
          .pred19-status-high, .pred19-status-low { background:#fff0df; color:#a35f0c; }
          .pred19-status-unknown { background:#edf1f4; color:#647684; }
          .pred19-quality-grid, .pred19-explanation-grid {
            display:grid;
            grid-template-columns:repeat(3,minmax(0,1fr));
            gap:.7rem;
          }
          .pred19-compact-panel {
            background:#fff;
            border:1px solid var(--p19-line);
            border-radius:12px;
            padding:13px 14px;
          }
          .pred19-compact-panel span {
            display:block;
            color:#758591;
            font-size:.64rem;
            font-weight:800;
            letter-spacing:.07em;
            text-transform:uppercase;
          }
          .pred19-compact-panel strong {
            display:block;
            color:var(--p19-ink);
            font-size:.93rem;
            margin:.25rem 0;
          }
          .pred19-compact-panel p { color:#5b6d7a; font-size:.72rem; margin:0; line-height:1.4; }
          .pred19-explanation {
            background:#edf5f4;
            border:1px solid #cfe3e1;
            border-radius:14px;
            padding:16px 18px;
            color:#294752;
            line-height:1.55;
          }
          .pred19-disclaimer {
            margin-top:2rem;
            background:#fff8e9;
            border-left:3px solid #d8a345;
            border-radius:9px;
            padding:12px 15px;
            color:#5d4b28;
            font-size:.75rem;
            line-height:1.5;
          }
          .pred19-empty {
            max-width:680px;
            background:#fff;
            border:1px solid var(--p19-line);
            border-radius:16px;
            padding:22px;
            margin:1.5rem auto;
            text-align:center;
          }
          .pred19-empty strong { display:block; font-size:1.05rem; margin:.3rem 0; }
          .pred19-empty p { color:var(--p19-muted); font-size:.82rem; }
          @media (max-width:900px) {
            .block-container { padding-left:1rem; padding-right:1rem; }
            .pred19-clinical-header { position:relative; top:0; align-items:flex-start; }
            .pred19-header-facts { gap:.65rem; }
            .pred19-risk-panel { grid-template-columns:1fr; gap:1rem; }
            .pred19-lab-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
            .pred19-quality-grid, .pred19-explanation-grid { grid-template-columns:1fr; }
          }
          @media (max-width:600px) {
            section[data-testid="stSidebar"] { min-width:250px; max-width:250px; }
            .pred19-monitor-top, .pred19-clinical-header, .pred19-section-heading {
              align-items:flex-start;
              flex-direction:column;
            }
            .pred19-jump-nav {
              display:grid;
              grid-template-columns:repeat(2,minmax(0,1fr));
              width:100%;
            }
            .pred19-jump-nav a { text-align:center; }
            .pred19-header-facts { justify-content:flex-start; }
            .pred19-lab-grid { grid-template-columns:1fr; }
            .pred19-section-heading p { text-align:left; }
            .pred19-risk-number { font-size:2.65rem; }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

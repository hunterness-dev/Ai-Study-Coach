"""
AI-Study-Coach — Streamlit frontend.

Run: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AI Study Coach",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2a2a4a 100%);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        color: #e0e0ff;
        margin-bottom: 0.5rem;
    }
    .metric-card h3 { margin: 0; font-size: 0.9rem; color: #9090cc; }
    .metric-card p  { margin: 0; font-size: 1.8rem; font-weight: 700; color: #c0c0ff; }
    .source-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 99px;
        font-size: 0.75rem;
        font-weight: 600;
        background: #6c63ff33;
        color: #a29bfe;
        border: 1px solid #6c63ff66;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎓 AI Study Coach")
    st.caption("Adaptive learning powered by ML + RL")
    st.divider()

    st.subheader("Log a Study Session")
    subject = st.text_input("Subject", placeholder="e.g. Mathematics")
    hours = st.number_input("Hours studied", min_value=0.1, max_value=24.0, value=2.0, step=0.5)
    score = st.number_input("Score / Grade (%)", min_value=0.0, max_value=100.0, value=75.0, step=0.5)

    if st.button("📥 Save Session", use_container_width=True, type="primary"):
        if not subject.strip():
            st.warning("Please enter a subject name.")
        else:
            try:
                resp = requests.post(
                    f"{API_BASE}/log",
                    json={"subject": subject.strip(), "hours": hours, "score": score},
                    timeout=10,
                )
                if resp.status_code == 201:
                    st.success("Session logged ✓")
                    st.rerun()
                else:
                    st.error(f"API error {resp.status_code}: {resp.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the API. Is the backend running?")

# ── Main Panel ────────────────────────────────────────────────────────────────
st.header("📊 Your Study Plan")

col_btn, _ = st.columns([2, 8])
with col_btn:
    refresh = st.button("🔄 Generate Plan", use_container_width=True)

if refresh or True:  # auto-load
    try:
        resp = requests.get(f"{API_BASE}/plan", timeout=10)

        if resp.status_code == 200:
            data = resp.json()

            source_label = "RL Agent" if data.get("source") == "rl" else "Rule-Based Scheduler"
            st.markdown(
                f"Allocation source: <span class='source-badge'>{source_label}</span>",
                unsafe_allow_html=True,
            )
            st.divider()

            allocs: dict = data.get("allocations", {})
            preds: dict = data.get("predicted_scores", {})
            feats: dict = data.get("features", {})

            if not allocs:
                st.info("No data yet. Log some study sessions using the sidebar.")
            else:
                # ── Metrics row
                cols = st.columns(len(allocs))
                for col, (subj, hrs) in zip(cols, allocs.items()):
                    with col:
                        st.markdown(
                            f"""<div class='metric-card'>
                            <h3>{subj}</h3>
                            <p>{hrs} hrs</p>
                            </div>""",
                            unsafe_allow_html=True,
                        )

                st.divider()

                # ── Details table
                rows = []
                for subj in allocs:
                    feat = feats.get(subj, {})
                    rows.append(
                        {
                            "Subject": subj,
                            "Allocated (h)": allocs.get(subj, "-"),
                            "Avg Score": feat.get("avg_score", "-"),
                            "Trend (slope)": feat.get("trend", "-"),
                            "Efficiency": feat.get("efficiency", "-"),
                            "Sessions": feat.get("session_count", "-"),
                            "Predicted Score": preds.get(subj, "—"),
                        }
                    )

                st.subheader("Subject Details")
                st.dataframe(rows, use_container_width=True)

        elif resp.status_code == 422:
            st.info("📭 " + resp.json().get("detail", "No data yet."))
        else:
            st.error(f"API error {resp.status_code}: {resp.text}")

    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot reach the backend API. Make sure it is running on " + API_BASE)

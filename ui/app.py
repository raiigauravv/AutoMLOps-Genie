"""
AutoMLOps Genie — Main UI
A polished Streamlit interface for the AutoML pipeline.
"""

import os
import sys
import streamlit as st
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from genie_agent.genie_main import genie_respond
    from pipelines.pipeline_builder import load_recent_mlflow_runs, get_shap_plot
    automl_available = True
except ImportError as e:
    automl_available = False
    import_error = str(e)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoMLOps Genie",
    page_icon="🧞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background: #0f1117; }

    /* Gradient hero banner */
    .hero {
        background: linear-gradient(135deg, #1a1f35 0%, #0d1b2a 50%, #1a2744 100%);
        border: 1px solid rgba(99,179,237,.18);
        border-radius: 16px;
        padding: 36px 40px 28px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(99,179,237,.12) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero h1 {
        font-size: 2.1rem; font-weight: 800;
        background: linear-gradient(135deg, #63b3ed, #90cdf4);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0 0 8px;
    }
    .hero p { color: rgba(255,255,255,.6); font-size: .95rem; margin: 0; }

    /* Pill badges */
    .badge {
        display: inline-block;
        background: rgba(99,179,237,.12);
        border: 1px solid rgba(99,179,237,.25);
        color: #90cdf4;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: .75rem;
        font-weight: 600;
        margin-right: 6px;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1f35, #0d1b2a);
        border: 1px solid rgba(99,179,237,.2);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-card .value {
        font-size: 1.8rem; font-weight: 800;
        background: linear-gradient(135deg, #63b3ed, #90cdf4);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .metric-card .label {
        font-size: .75rem; color: rgba(255,255,255,.5);
        text-transform: uppercase; letter-spacing: .5px;
        margin-top: 4px;
    }

    /* Step cards in sidebar */
    .step {
        background: rgba(99,179,237,.06);
        border-left: 3px solid #63b3ed;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        margin-bottom: 10px;
        font-size: .85rem;
        color: rgba(255,255,255,.8);
    }

    /* Result card */
    .result-card {
        background: rgba(72,187,120,.08);
        border: 1px solid rgba(72,187,120,.3);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }

    /* Hide Streamlit default menu */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧞 AutoMLOps Genie")
    st.markdown("---")

    st.markdown("**How it works**")
    st.markdown('<div class="step">📁 <b>1.</b> Upload your CSV dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="step">💬 <b>2.</b> Describe your prediction task in plain English</div>', unsafe_allow_html=True)
    st.markdown('<div class="step">🚀 <b>3.</b> Genie trains multiple models automatically</div>', unsafe_allow_html=True)
    st.markdown('<div class="step">📊 <b>4.</b> Review the leaderboard and feature importance</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Sample prompts**")
    st.code("Predict Churn using all features")
    st.code("Predict charges using all features")
    st.code("Predict is_fraud using all features")

    st.markdown("---")
    st.markdown("**Powered by**")
    st.markdown("""
    <span class="badge">🤖 GPT Function Calling</span><br><br>
    <span class="badge">⚡ AutoGluon</span><br><br>
    <span class="badge">📈 MLflow</span><br><br>
    <span class="badge">🔍 SHAP</span>
    """, unsafe_allow_html=True)

    # Previous experiments (collapsible)
    st.markdown("---")
    with st.expander("📈 Experiment History"):
        if automl_available:
            try:
                runs_df = load_recent_mlflow_runs(limit=5)
                if not runs_df.empty:
                    st.dataframe(runs_df, use_container_width=True, hide_index=True)
                else:
                    st.caption("No experiments yet — run your first pipeline above.")
            except Exception:
                st.caption("History unavailable.")
        else:
            st.caption("AutoML not available.")

# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🧞 AutoMLOps Genie</h1>
    <p>Describe your ML task in plain English. Genie trains, evaluates, and explains the best model — automatically.</p>
    <br>
    <span class="badge">GPT Function Calling</span>
    <span class="badge">AutoGluon AutoML</span>
    <span class="badge">MLflow Observability</span>
    <span class="badge">SHAP Explainability</span>
</div>
""", unsafe_allow_html=True)

# ── Main layout ───────────────────────────────────────────────────────────────
col_upload, col_task = st.columns([1, 1], gap="large")

# ── Left: Upload ──────────────────────────────────────────────────────────────
with col_upload:
    st.markdown("### 📁 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload any CSV with column headers in the first row.",
        label_visibility="collapsed",
    )

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state["df"] = df

            # Stats row
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f'<div class="metric-card"><div class="value">{len(df):,}</div><div class="label">Rows</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><div class="value">{len(df.columns)}</div><div class="label">Columns</div></div>', unsafe_allow_html=True)
            with c3:
                nulls = df.isnull().sum().sum()
                st.markdown(f'<div class="metric-card"><div class="value">{nulls}</div><div class="label">Nulls</div></div>', unsafe_allow_html=True)

            st.markdown("")
            st.markdown("**Columns available:**")
            cols_html = " ".join([f'<span class="badge">{c}</span>' for c in df.columns])
            st.markdown(cols_html, unsafe_allow_html=True)

            st.markdown("")
            with st.expander("👀 Preview data"):
                st.dataframe(df.head(10), use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.session_state.pop("df", None)
    else:
        st.session_state.pop("df", None)
        # Sample dataset buttons
        st.markdown("**Or try a sample dataset:**")
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        sc1, sc2, sc3 = st.columns(3)
        for btn, fname, label in [
            (sc1, "churn_sample.csv", "📉 Churn"),
            (sc2, "fraud_sample.csv", "🔍 Fraud"),
            (sc3, "insurance_sample.csv", "🏥 Insurance"),
        ]:
            with btn:
                if st.button(label, use_container_width=True):
                    path = os.path.join(data_dir, fname)
                    if os.path.exists(path):
                        st.session_state["df"] = pd.read_csv(path)
                        st.session_state["sample_name"] = fname
                        st.rerun()

# ── Right: Task + Run ─────────────────────────────────────────────────────────
with col_task:
    st.markdown("### 🎯 Describe Your ML Task")

    df = st.session_state.get("df")

    if df is not None:
        suggested = df.columns[-1]
        prompt = st.text_area(
            "What do you want to predict?",
            value=f"Predict {suggested} using all features",
            height=120,
            help="Mention the column name you want to predict.",
        )

        run_btn = st.button("🚀 Run AutoML Pipeline", type="primary", use_container_width=True)

        if run_btn:
            if not prompt.strip():
                st.warning("Please describe your task first.")
            elif not automl_available:
                st.error(f"AutoML unavailable: {import_error}")
            else:
                # Case-insensitive column matching
                col_map = {c.lower(): c for c in df.columns}
                target_column = None
                for word in prompt.split():
                    cw = word.strip(".,!?`").lower()
                    if cw in col_map:
                        target_column = col_map[cw]
                        break

                if target_column is None:
                    st.error(f"Couldn't find a column name in your prompt. Available: {', '.join(df.columns)}")
                else:
                    st.info(f"🎯 Target identified: **{target_column}** — starting AutoML training...")

                    with st.spinner("⚡ Training models with AutoGluon..."):
                        try:
                            result, parsed, model_info_path, leaderboard, model_dir = genie_respond(prompt, df)
                            st.session_state["result"] = {
                                "result": result,
                                "parsed": parsed,
                                "model_info_path": model_info_path,
                                "leaderboard": leaderboard,
                                "model_dir": model_dir,
                                "df": df,
                            }
                        except Exception as e:
                            st.error(f"Training failed: {e}")
    else:
        st.markdown("")
        st.info("👈 Upload a dataset or pick a sample on the left to get started.")

# ── Results ───────────────────────────────────────────────────────────────────
res = st.session_state.get("result")
if res:
    st.markdown("---")
    st.markdown("## 📊 Results")

    # Success banner
    st.markdown(f'<div class="result-card">✅ <b>{res["result"]}</b></div>', unsafe_allow_html=True)

    tab_leaderboard, tab_shap, tab_download = st.tabs(["🏆 Model Leaderboard", "🔍 Feature Importance", "⬇️ Download"])

    with tab_leaderboard:
        lb = res.get("leaderboard")
        if lb is not None and not lb.empty:
            st.markdown("**All models ranked by validation score:**")

            # Highlight best model
            best = lb.iloc[0]["model"]
            st.success(f"🥇 Best model: **{best}**")

            # Color the leaderboard
            st.dataframe(
                lb.style.highlight_max(subset=["score_val"], color="#1a3a2a"),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning("Leaderboard not available.")

    with tab_shap:
        parsed = res.get("parsed", {})
        model_dir = res.get("model_dir")
        df_res = res.get("df")
        target_col = parsed.get("target")

        if df_res is not None and target_col:
            fig = None

            # Try SHAP / AutoGluon importance first
            if model_dir:
                with st.spinner("Generating feature importance..."):
                    fig = get_shap_plot(model_dir, df_res, parsed)

            # Guaranteed fallback: correlation-based importance (always works)
            if fig is None:
                from pipelines.pipeline_builder import create_correlation_plot
                import matplotlib.pyplot as plt
                import pandas as pd
                import numpy as np
                from sklearn.preprocessing import LabelEncoder

                try:
                    df_enc = df_res.copy()
                    for col in df_enc.select_dtypes(include=["object"]).columns:
                        if col != target_col:
                            le = LabelEncoder()
                            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
                    if df_enc[target_col].dtype == object:
                        le = LabelEncoder()
                        df_enc[target_col] = le.fit_transform(df_enc[target_col].astype(str))

                    corr = df_enc.corr()[target_col].abs().drop(target_col).sort_values(ascending=True)

                    fig, ax = plt.subplots(figsize=(9, max(4, len(corr) * 0.5)))
                    colors = ["#63b3ed" if v >= corr.median() else "#4a5568" for v in corr.values]
                    ax.barh(corr.index, corr.values, color=colors, alpha=0.85)
                    ax.set_xlabel("Absolute Correlation with Target", color="white", fontsize=11)
                    ax.set_title(f"Feature Importance — Correlation with '{target_col}'",
                                 color="white", fontsize=13, fontweight="bold", pad=14)
                    ax.tick_params(colors="white")
                    ax.spines[["top", "right"]].set_visible(False)
                    ax.spines[["left", "bottom"]].set_color("#4a5568")
                    fig.patch.set_facecolor("#0d1b2a")
                    ax.set_facecolor("#0d1b2a")
                    plt.tight_layout()
                except Exception as e:
                    fig = None
                    st.warning(f"Could not generate feature importance: {e}")

            if fig:
                st.pyplot(fig, use_container_width=True)
                st.caption("Blue bars = above-median correlation with target. "
                           "Longer bar = stronger influence on prediction.")
        else:
            st.warning("Run the pipeline first to see feature importance.")

    with tab_download:
        model_info_path = res.get("model_info_path")
        if model_info_path and os.path.exists(model_info_path):
            with open(model_info_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Model Info",
                    f,
                    file_name="model_info.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            st.code(open(model_info_path).read())
        else:
            st.warning("Model file not available.")

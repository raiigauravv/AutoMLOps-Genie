import streamlit as st
import pandas as pd
import os
import sys

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try importing AutoML functions
try:
    from genie_agent.genie_main import genie_respond
    from pipelines.pipeline_builder import load_recent_mlflow_runs, get_shap_plot
    automl_available = True
except ImportError as e:
    automl_available = False
    import_error = str(e)

st.set_page_config(
    page_title="AutoMLOps Genie", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'show_guide' not in st.session_state:
    st.session_state.show_guide = True

# Header
st.title("🧞 AutoMLOps Genie")
st.markdown("**Automated machine learning with explainable AI**")
st.markdown("")

# How it works guide (dismissible)
if st.session_state.show_guide:
    guide_col1, guide_col2 = st.columns([6, 1])
    
    with guide_col1:
        st.info("""
        **How it works:**  
        1. Upload your CSV file (first row must be column names)  
        2. Enter what you want to predict using the exact column name  
        3. Click "Run AutoML" and get your trained model
        """)
    
    with guide_col2:
        if st.button("✕", help="Close guide"):
            st.session_state.show_guide = False
            st.rerun()

st.markdown("")

# Main interface
col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    # File uploader
    st.markdown("### 📁 Upload Your Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type=["csv"],
        help="Upload a CSV file with column headers in the first row"
    )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            
            # Show success and data info
            st.success(f"✓ Dataset loaded: {len(df)} rows × {len(df.columns)} columns")
            
            # Display column names as helpful info
            st.markdown("**Available columns:**")
            col_display = ", ".join([f"`{col}`" for col in df.columns])
            st.markdown(f"{col_display}")
            
            # Auto-suggest a prompt based on the last column
            suggested_target = df.columns[-1]
            st.info(f"💡 **Suggested prompt:** Predict `{suggested_target}` using all features")
            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.session_state.df = None
    
    # Get current dataset
    df = st.session_state.df
    
    if df is not None:
        st.markdown("")
        st.markdown("### 🎯 Describe Your ML Task")
        
        # Task description
        suggested_target = df.columns[-1]
        prompt = st.text_area(
            "What do you want to predict?",
            placeholder=f"e.g., Predict {suggested_target} using all features",
            height=100,
            help="Use the exact column name from your data as the target variable"
        )
        
        st.markdown("")
        
        # Run button
        run_button = st.button("🚀 Run AutoML", type="primary", use_container_width=True)
        
        if run_button:
            if not prompt:
                st.warning("Please describe your ML task first")
            elif not automl_available:
                st.error(f"AutoML not available: {import_error}")
            else:
                # Validate prompt against columns
                target_column = None
                if prompt:
                    words = prompt.split()
                    if "predict" in prompt.lower():
                        for word in words:
                            clean_word = word.strip(".,!?").strip("`")
                            if clean_word in df.columns:
                                target_column = clean_word
                                break
                
                if target_column is None:
                    available_cols = ", ".join([f"`{col}`" for col in df.columns])
                    st.error(f"Please use an exact column name from your data. Available columns: {available_cols}")
                else:
                    # Run AutoML
                    with st.spinner("🔄 Training models... This may take a minute"):
                        try:
                            result, parsed, model_info_path, leaderboard, model_dir = genie_respond(prompt, df)
                            
                            # Results section
                            st.markdown("")
                            st.markdown("---")
                            st.markdown("## 📊 Results")
                            
                            # Success message
                            st.success("✅ AutoML training completed successfully!")
                            
                            # Model Performance
                            if leaderboard is not None and not leaderboard.empty:
                                st.markdown("")
                                st.markdown("### 🏆 Model Performance")
                                st.markdown("*Top performing models ranked by accuracy*")
                                
                                # Show top 5 models
                                top_models = leaderboard.head(5)
                                st.dataframe(top_models, use_container_width=True, hide_index=True)
                                
                                # Show all models in expander if more than 5
                                if len(leaderboard) > 5:
                                    with st.expander(f"View all {len(leaderboard)} models"):
                                        st.dataframe(leaderboard, use_container_width=True, hide_index=True)
                            
                            st.markdown("")
                            
                            # Action cards
                            action_col1, action_col2 = st.columns(2)
                            
                            # Model download card
                            with action_col1:
                                st.markdown("#### ⬇️ Download Model")
                                if model_info_path and os.path.exists(model_info_path):
                                    with open(model_info_path, "rb") as f:
                                        st.download_button(
                                            "Download Model Info (.txt)", 
                                            f, 
                                            file_name=os.path.basename(model_info_path),
                                            mime="text/plain",
                                            use_container_width=True,
                                            type="primary"
                                        )
                                    st.markdown("*Model information and loading instructions*")
                                else:
                                    st.warning("Model file not available")
                            
                            # Feature importance card  
                            with action_col2:
                                st.markdown("#### 🔍 Feature Importance")
                                fig = get_shap_plot(model_dir, df, parsed)
                                if fig is not None:
                                    st.pyplot(fig, use_container_width=True)
                                    st.markdown("*Most influential features for predictions*")
                                else:
                                    st.warning("Feature importance not available")
                            
                            # Data preview
                            st.markdown("")
                            with st.expander("📋 View Your Data", expanded=False):
                                st.dataframe(df.head(100), use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"An error occurred during training: {str(e)}")
                            st.markdown("Please check your data and prompt, then try again.")

st.markdown("")
st.markdown("")

# Optional: Experiment history (very minimized)
if st.session_state.df is not None and automl_available:
    with st.expander("📈 Previous Experiments", expanded=False):
        try:
            runs_df = load_recent_mlflow_runs()
            if not runs_df.empty:
                st.dataframe(runs_df.head(3), use_container_width=True, hide_index=True)
            else:
                st.info("No previous experiments")
        except:
            st.info("Experiment history not available")

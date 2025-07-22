# Simple version to debug the blank page issue
import streamlit as st
import pandas as pd
import os
import sys

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="AutoMLOps Genie", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Test basic session state
if 'test' not in st.session_state:
    st.session_state.test = True

st.title("🧞 AutoMLOps Genie")
st.write("**Automated machine learning with explainable AI**")

st.write("If you can see this, the app is working!")

# Test file uploader
uploaded_file = st.file_uploader("Test Upload", type=["csv"])

if uploaded_file:
    st.write("File uploaded successfully!")

st.button("Test Button")

# Quick Gemini Prompt for Architecture Diagram

Copy and paste this prompt to Gemini:

---

**Create a system architecture diagram for AutoMLOps-Genie:**

An AutoML platform with these components:
1. **Streamlit UI** (file upload, prompt input, results display)
2. **LLM Agent** (OpenAI GPT-4 for prompt parsing)
3. **AutoML Pipeline** (AutoGluon for model training)  
4. **SHAP Interpretability** (feature importance)
5. **MLflow Tracking** (experiment logging)

**Data Flow:**
User uploads CSV + enters prompt → LLM parses intent → AutoGluon trains models → SHAP explains results → User downloads model

**Tech Stack:** Python 3.12, Streamlit, AutoGluon, OpenAI API, SHAP, MLflow

**Show:** Component boxes, data flow arrows, tech labels, fallback paths, and color-coded layers (UI=blue, ML=green, Storage=orange, External=red).

Create both high-level overview and detailed technical diagrams.

---

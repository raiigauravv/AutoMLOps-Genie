# AutoMLOps-Genie System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AutoMLOps-Genie Platform                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      USER INTERFACE LAYER                       │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │               Streamlit Web App                             │ │
│  │         (ui/minimal_app.py)                                 │ │
│  │                                                             │ │
│  │  📄 File Upload  💬 Prompt Input  📊 Results  📥 Download   │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PROMPT INTELLIGENCE LAYER                     │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                  LLM Agent                                  │ │
│  │         (genie_agent/genie_main.py)                         │ │
│  │                                                             │ │
│  │  Input: prompt + columns                                    │ │
│  │  Output: target, problem_type, features                     │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼                    ┌─────────────┐
┌─────────────────────────────────────────────────────────────────┐ │ OpenAI API  │
│                     AUTOML PIPELINE LAYER                       │ │  (GPT-4)    │
│  ┌─────────────────────────────────────────────────────────────┐ │ └─────────────┘
│  │                Pipeline Builder                             │ │        ▲
│  │        (pipelines/pipeline_builder.py)                      │ │        │
│  │                                                             │ │        │
│  │  🤖 AutoGluon Training                                      │ │────────┘
│  │  🔄 Fallback Strategies                                     │ │
│  │  💾 Model Persistence                                       │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                 MODEL INTERPRETABILITY LAYER                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                 SHAP Integration                            │ │
│  │           (within pipeline_builder.py)                      │ │
│  │                                                             │ │
│  │  📈 Feature Importance                                      │ │
│  │  🎯 Model Explanations                                      │ │
│  │  📊 Visualizations                                          │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                  EXPERIMENT TRACKING LAYER                      │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                  MLflow Integration                         │ │
│  │                                                             │ │
│  │  📝 Experiment Logging                                      │ │
│  │  🏷️  Model Versioning                                       │ │
│  │  📊 Metrics Tracking                                        │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DATA STORAGE LAYER                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Sample Data   │  │  Model Storage  │  │   MLflow Runs   │ │
│  │     (data/)     │  │   (models/)     │  │   (mlruns/)     │ │
│  │                 │  │                 │  │                 │ │
│  │  📄 CSV files   │  │  🤖 .pkl files  │  │  📈 Metrics     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

TECHNOLOGY STACK:
• Frontend: Streamlit 1.47.0
• ML Framework: AutoGluon 1.2  
• LLM: OpenAI GPT-4
• Interpretability: SHAP 0.48.0
• Tracking: MLflow 3.1.3
• Runtime: Python 3.12

DATA FLOW:
1. User uploads CSV + prompt → Streamlit UI
2. Prompt + columns → LLM Agent → OpenAI API
3. Parsed intent → Pipeline Builder → AutoGluon
4. Trained model → SHAP explanations
5. Results + visualizations → Streamlit UI
6. User downloads model artifacts
```

# AutoMLOps Genie 🧞

An intelligent AutoML platform that builds, trains, and deploys machine learning models from natural language prompts. Describe your task in plain English — Genie handles everything else.

🔗 **Live Demo:** https://automlops-genie-kst4vvfmga-uc.a.run.app

![Python](https://img.shields.io/badge/Python-3.10-blue)
![GCP](https://img.shields.io/badge/Deployed-GCP%20Cloud%20Run-4285F4)
![MLflow](https://img.shields.io/badge/Observability-MLflow-blue)
![Tests](https://img.shields.io/badge/Tests-18%20passing-brightgreen)

---

## ✨ Features

- **LLM Function Calling** — Gemini's `tool_config` API forces structured output via a `function_calling_config` mode, replacing fragile prompt engineering with validated JSON extraction
- **Automated ML Pipeline** — AutoGluon 1.2 runs its automated model search (`medium_quality_faster_train` preset, stacking disabled) to train and compare candidate models
- **MLflow Observability** — Every run logs target column, task type, best model name, and accuracy/RMSE. Experiment history is surfaced live in the UI
- **Feature Importance** — SHAP explainability with correlation-based fallback; rendered as an animated Chart.js bar chart
- **React SPA Frontend** — Polished drag-and-drop UI built with React 18, Tailwind CSS, and Chart.js served by FastAPI
- **GCP Deployment** — Containerised with Docker; built via Cloud Build and deployed to Cloud Run via GitHub Actions CI/CD
- **Automated Testing** — 18 pytest tests covering function-calling validation, task-type detection, MLflow logging, and pipeline contracts

---

## 🏗️ Architecture

```
User (natural language prompt + CSV)
        │
        ▼
┌─────────────────────────────┐
│  React SPA (frontend/)      │  ← drag & drop upload, polling UI
│  FastAPI (server.py)        │  ← background job queue, REST API
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  genie_agent/genie_main.py  │
│  Gemini Function Calling    │  ← tool_config forces the call
│  configure_ml_pipeline()    │  ← returns {target, type}
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ pipelines/pipeline_builder  │
│  AutoGluon TabularPredictor │  ← automated model search
│  MLflow logging             │  ← metrics, params, artifacts
│  SHAP feature importance    │  ← explainability
└──────────────┬──────────────┘
               │
               ▼
      GCP Cloud Run
    (Docker + GitHub Actions CI/CD)
```

---

## 🚀 Quick Start (Local)

### Prerequisites
- Python 3.10
- Gemini API key (free tier: https://aistudio.google.com/apikey)

```bash
git clone https://github.com/raiigauravv/AutoMLOps-Genie.git
cd AutoMLOps-Genie

python3.10 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

echo "GEMINI_API_KEY=..." > .env

# Start the FastAPI server + React SPA
uvicorn server:app --host 0.0.0.0 --port 8501 --reload
# Open http://localhost:8501
```

---

## 🐳 Docker

```bash
docker build -t automlops-genie .
docker run -p 8501:8501 -e GEMINI_API_KEY=... automlops-genie
# Open http://localhost:8501
```

---

## ☁️ Deploy to GCP Cloud Run

### CI/CD via GitHub Actions

Add these repository secrets:
- `GCP_SA_KEY` — JSON key for a service account with Cloud Run, Artifact Registry, Cloud Build, and logging-viewer roles
- `GCP_PROJECT_ID` — your GCP project ID
- `GEMINI_API_KEY` — used at deploy time to configure the live service

Every push to `main`:
1. Runs the full pytest suite
2. Submits a Cloud Build job that builds and pushes a Docker image tagged with the commit SHA to Artifact Registry
3. Deploys the image to Cloud Run automatically

See `.github/workflows/gcp-deploy.yml` for the full pipeline.

---

## 🧪 Automated Tests

```bash
pytest tests/ -v
```

```
tests/test_function_calling.py::test_uses_tools_parameter          PASSED
tests/test_function_calling.py::test_tool_choice_forces_correct_function PASSED
tests/test_function_calling.py::test_classification_prompt         PASSED
tests/test_function_calling.py::test_regression_prompt             PASSED
tests/test_function_calling.py::test_fraud_detection_prompt        PASSED
tests/test_pipeline.py::test_binary_target_detected_as_classification PASSED
tests/test_pipeline.py::test_logs_target_and_task_type_params      PASSED
tests/test_pipeline.py::test_logs_accuracy_metric_for_classification PASSED
... 18 tests passing
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | ✅ Yes | Gemini function calling |

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Gemini 2.0 Flash (function calling) |
| AutoML | AutoGluon 1.2 |
| Backend | FastAPI + uvicorn |
| Frontend | React 18 + Tailwind CSS + Chart.js |
| Observability | MLflow |
| Explainability | SHAP |
| Deployment | Docker + GCP Cloud Run |
| CI/CD | GitHub Actions + Cloud Build |
| Testing | pytest |

---

## 📁 Project Structure

```
AutoMLOps-Genie/
├── server.py                   # FastAPI backend + background job queue
├── frontend/
│   └── index.html              # React SPA (drag & drop, leaderboard, charts)
├── genie_agent/
│   └── genie_main.py           # Gemini function calling pipeline parser
├── pipelines/
│   └── pipeline_builder.py     # AutoGluon + MLflow pipeline
├── tests/
│   ├── conftest.py             # Test stubs for heavy ML dependencies
│   ├── test_function_calling.py
│   └── test_pipeline.py
├── .github/workflows/
│   └── gcp-deploy.yml          # CI/CD pipeline (test → Cloud Build → Cloud Run)
├── Dockerfile
└── requirements.txt
```

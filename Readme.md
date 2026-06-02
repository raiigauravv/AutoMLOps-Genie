# AutoMLOps Genie 🧞

An intelligent AutoML platform that builds, trains, and deploys machine learning models from natural language prompts. Describe your task in plain English — Genie handles everything else.

🔗 **Live Demo:** https://automlops-genie.greenglacier-5e31ee4e.westus2.azurecontainerapps.io

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Azure](https://img.shields.io/badge/Deployed-Azure%20Container%20Apps-0089D6)
![MLflow](https://img.shields.io/badge/Observability-MLflow-blue)
![Tests](https://img.shields.io/badge/Tests-18%20passing-brightgreen)

---

## ✨ Features

- **LLM Function Calling** — OpenAI `tools=` API forces structured output via `tool_choice`, replacing fragile prompt engineering with validated JSON extraction
- **Automated ML Pipeline** — AutoGluon 1.2 trains and compares GBM, Random Forest, Extra Trees, KNN, and stacked ensembles automatically
- **MLflow Observability** — Every run logs target column, task type, best model name, and accuracy/RMSE. Experiment history is surfaced live in the UI
- **Feature Importance** — SHAP explainability with correlation-based fallback; rendered as an animated Chart.js bar chart
- **React SPA Frontend** — Polished drag-and-drop UI built with React 18, Tailwind CSS, and Chart.js served by FastAPI
- **Azure Deployment** — Containerised with Docker; deployed to Azure Container Apps via GitHub Actions CI/CD
- **Automated Testing** — 18 pytest tests covering function-calling validation, task-type detection, MLflow logging, and pipeline contracts

---

## 📉 90% Reduction in Manual Pipeline Cycles

A standard ML workflow requires ~10 manual steps. AutoMLOps Genie automates 9 of them:

| Step | Manual Workflow | AutoMLOps Genie |
|------|----------------|-----------------|
| 1 | Load & inspect data | ✅ Automatic |
| 2 | Handle missing values | ✅ AutoGluon |
| 3 | Encode categorical features | ✅ AutoGluon |
| 4 | Scale / normalise features | ✅ AutoGluon |
| 5 | Train/validation split | ✅ AutoGluon |
| 6 | Choose model architecture | ✅ AutoGluon (multi-model) |
| 7 | Tune hyperparameters | ✅ AutoGluon |
| 8 | Evaluate and compare models | ✅ Live leaderboard |
| 9 | Log metrics and artifacts | ✅ MLflow |
| 10 | Interpret feature importance | ✅ SHAP / Chart.js |
| **User action** | 10 manual steps | **1 natural-language prompt** |

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
│  OpenAI Function Calling    │  ← tools= API, tool_choice enforced
│  configure_ml_pipeline()    │  ← returns {target, type}
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ pipelines/pipeline_builder  │
│  AutoGluon TabularPredictor │  ← multi-model AutoML
│  MLflow logging             │  ← metrics, params, artifacts
│  SHAP feature importance    │  ← explainability
└──────────────┬──────────────┘
               │
               ▼
    Azure Container Apps
    (Docker + GitHub Actions CI/CD)
```

---

## 🚀 Quick Start (Local)

### Prerequisites
- Python 3.10
- OpenAI API key

```bash
git clone https://github.com/raiigauravv/AutoMLOps-Genie.git
cd AutoMLOps-Genie

python3.10 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

echo "OPENAI_API_KEY=sk-..." > .env

# Start the FastAPI server + React SPA
uvicorn server:app --host 0.0.0.0 --port 8501 --reload
# Open http://localhost:8501
```

---

## 🐳 Docker

```bash
docker build -t automlops-genie .
docker run -p 8501:8501 -e OPENAI_API_KEY=sk-... automlops-genie
# Open http://localhost:8501
```

---

## ☁️ Deploy to Azure

### One-command deployment

```bash
az login
OPENAI_API_KEY=sk-... ./deploy_azure.sh
```

The script creates an Azure Container Registry, builds and pushes the Docker image, and deploys to Azure Container Apps with external HTTPS ingress.

### CI/CD via GitHub Actions

Add `AZURE_CREDENTIALS` as a GitHub secret (output of `az ad sp create-for-rbac`).

Every push to `main`:
1. Runs the full pytest suite
2. Builds a new Docker image tagged with the commit SHA
3. Deploys to Azure Container Apps automatically

---

## 🧪 Automated Tests

```bash
pytest tests/ -v
```

```
tests/test_function_calling.py::test_uses_tools_parameter          PASSED
tests/test_function_calling.py::test_tool_choice_forces_function   PASSED
tests/test_function_calling.py::test_classification_prompt         PASSED
tests/test_function_calling.py::test_regression_prompt             PASSED
tests/test_function_calling.py::test_fraud_detection_prompt        PASSED
tests/test_pipeline.py::test_binary_target_detected                PASSED
tests/test_pipeline.py::test_logs_target_and_task_type_params      PASSED
tests/test_pipeline.py::test_logs_accuracy_metric                  PASSED
... 18 tests passing
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ Yes | GPT-3.5-turbo function calling |

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | OpenAI GPT-3.5-turbo (function calling) |
| AutoML | AutoGluon 1.2 |
| Backend | FastAPI + uvicorn |
| Frontend | React 18 + Tailwind CSS + Chart.js |
| Observability | MLflow |
| Explainability | SHAP |
| Deployment | Docker + Azure Container Apps |
| CI/CD | GitHub Actions |
| Testing | pytest |

---

## 📁 Project Structure

```
AutoMLOps-Genie/
├── server.py                   # FastAPI backend + background job queue
├── frontend/
│   └── index.html              # React SPA (drag & drop, leaderboard, charts)
├── genie_agent/
│   └── genie_main.py           # OpenAI function calling pipeline parser
├── pipelines/
│   └── pipeline_builder.py     # AutoGluon + MLflow pipeline
├── tests/
│   ├── conftest.py             # Test stubs for heavy ML dependencies
│   ├── test_function_calling.py
│   └── test_pipeline.py
├── .github/workflows/
│   └── azure-deploy.yml        # CI/CD pipeline
├── Dockerfile
├── deploy_azure.sh
└── requirements.txt
```

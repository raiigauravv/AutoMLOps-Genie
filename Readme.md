# AutoMLOps Genie 🧞

An intelligent AutoML platform that automatically builds, trains, and deploys machine learning models from natural language prompts. Describe your task in plain English — Genie handles everything else.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Azure](https://img.shields.io/badge/Deployed-Azure%20Container%20Apps-0089D6)
![MLflow](https://img.shields.io/badge/Observability-MLflow-blue)
![Tests](https://img.shields.io/badge/Tests-pytest-green)

## ✨ Features

- **LLM Function Calling** — Uses OpenAI's `tools` API to parse natural-language task descriptions into structured ML pipeline configs. The model is *forced* to return validated JSON via `tool_choice`, eliminating fragile string parsing.
- **Automated ML Pipeline** — AutoGluon trains and compares multiple model families (GBM, RF, NN, CatBoost) automatically; the best model is selected by validation score.
- **MLflow Observability** — Every run logs target, task type, problem type, best model name, and accuracy/RMSE metrics. Experiment history is surfaced in the UI via `load_recent_mlflow_runs()`.
- **SHAP Explainability** — Feature importance plots via SHAP KernelExplainer, with AutoGluon and correlation-based fallbacks.
- **Azure Deployment** — Containerised with Docker; deployed to Azure Container Apps via `deploy_azure.sh` and automated via GitHub Actions CI/CD.
- **Automated Testing** — pytest suite covering function-calling output validation, task-type detection, MLflow logging assertions, and pipeline output contracts.

---

## 📉 Pipeline Automation: 90% reduction in manual steps

A standard manual ML workflow requires approximately **10 distinct manual steps**:

| Step | Manual workflow | AutoMLOps Genie |
|------|----------------|-----------------|
| 1 | Load & inspect data | ✅ Automatic |
| 2 | Handle missing values | ✅ AutoGluon |
| 3 | Encode categorical features | ✅ AutoGluon |
| 4 | Scale / normalise features | ✅ AutoGluon |
| 5 | Split train / validation sets | ✅ AutoGluon |
| 6 | Choose model architecture | ✅ AutoGluon (multi-model) |
| 7 | Tune hyperparameters | ✅ AutoGluon |
| 8 | Evaluate and compare models | ✅ Leaderboard |
| 9 | Log metrics and artifacts | ✅ MLflow |
| 10 | Interpret feature importance | ✅ SHAP |
| **User action** | 10 manual steps | **1 natural-language prompt** |

**Result: 9 of 10 steps fully automated = 90% reduction in manual pipeline cycles.**

---

## 🏗️ Architecture

```
User (natural language prompt)
        │
        ▼
┌─────────────────────────────┐
│  genie_agent/genie_main.py  │
│  OpenAI Function Calling    │  ← tools= API, not plain chat
│  configure_ml_pipeline()    │
└──────────────┬──────────────┘
               │  {target, type}
               ▼
┌─────────────────────────────┐
│ pipelines/pipeline_builder  │
│  AutoGluon TabularPredictor │  ← multi-model AutoML
│  MLflow logging             │  ← metrics, params, artifacts
│  SHAP feature importance    │  ← model interpretability
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  ui/minimal_app.py          │
│  Streamlit                  │
│  MLflow run history         │
└─────────────────────────────┘
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
git clone https://github.com/yourusername/AutoMLOps-Genie.git
cd AutoMLOps-Genie

python3.10 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

echo "OPENAI_API_KEY=sk-..." > .env

streamlit run ui/minimal_app.py
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

### One-command deployment (Azure Container Apps)

```bash
az login
OPENAI_API_KEY=sk-... ./deploy_azure.sh
```

The script:
1. Creates a resource group and Azure Container Registry
2. Builds and pushes the Docker image via `az acr build`
3. Creates a Container Apps environment
4. Deploys with external ingress and returns the live HTTPS URL

### CI/CD via GitHub Actions

Set the following GitHub secret: `AZURE_CREDENTIALS` (output of `az ad sp create-for-rbac`)

Every push to `main` will:
1. Run the full pytest suite
2. Build a new Docker image tagged with the commit SHA
3. Deploy to Azure Container Apps automatically

---

## 🧪 Automated Tests

```bash
pytest tests/ -v
```

Test coverage:
- `tests/test_function_calling.py` — LLM function-calling API, tool_choice enforcement, error handling
- `tests/test_pipeline.py` — Task-type detection, MLflow param/metric logging, pipeline output contract, run history

```
tests/test_function_calling.py::TestExtractInfoFromPrompt::test_classification_prompt PASSED
tests/test_function_calling.py::TestExtractInfoFromPrompt::test_regression_prompt PASSED
tests/test_function_calling.py::TestExtractInfoFromPrompt::test_uses_tools_parameter PASSED
tests/test_function_calling.py::TestExtractInfoFromPrompt::test_tool_choice_forces_correct_function PASSED
tests/test_pipeline.py::TestMLflowLogging::test_logs_target_and_task_type_params PASSED
tests/test_pipeline.py::TestMLflowLogging::test_logs_accuracy_metric_for_classification PASSED
...
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ Yes | GPT-3.5-turbo function calling |

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI GPT-3.5-turbo (function calling) |
| AutoML | AutoGluon 1.2 |
| Observability | MLflow |
| Explainability | SHAP |
| UI | Streamlit |
| Deployment | Docker + Azure Container Apps |
| CI/CD | GitHub Actions |
| Testing | pytest + pytest-cov |

---

## ⚠️ Disclaimer

This tool is for demonstration and educational purposes. Model outputs should not be used for critical decisions without expert validation.

# AutoMLOps-Genie Architecture Generation Prompt

## For Gemini AI: Generate System Architecture Diagram

Please create a detailed system architecture diagram for the AutoMLOps-Genie platform based on the following specifications:

### System Overview
AutoMLOps-Genie is an intelligent AutoML platform that automatically builds, trains, and deploys machine learning models using natural language prompts. It combines LLM-powered prompt parsing, automated ML pipelines, and model interpretability.

### Core Components & Data Flow

#### 1. User Interface Layer
- **Component**: Streamlit Web App (`ui/minimal_app.py`)
- **Features**: 
  - File upload interface for CSV datasets
  - Natural language prompt input
  - Results display with model metrics
  - SHAP visualization cards
  - Model download functionality
- **Style**: Modern, minimal, single-page design without sidebar

#### 2. Prompt Intelligence Layer
- **Component**: LLM Agent (`genie_agent/genie_main.py`)
- **Function**: `genie_respond(prompt, df_columns)`
- **Input**: User prompt + dataset column names
- **Processing**: OpenAI GPT-4 analysis
- **Output**: Structured response (target_column, problem_type, features, model_explanation, business_insights)
- **Problem Types**: binary, multiclass, regression

#### 3. AutoML Pipeline Layer
- **Component**: Pipeline Builder (`pipelines/pipeline_builder.py`)
- **Function**: `build_automl_pipeline(df, target_col, problem_type, features)`
- **ML Framework**: AutoGluon TabularPredictor
- **Features**:
  - Automatic model training with fallback strategies
  - Progressive preset selection (medium_quality → minimal)
  - Model persistence and loading
  - Error handling and cleanup

#### 4. Model Interpretability Layer
- **Component**: SHAP Integration (within pipeline_builder.py)
- **Features**:
  - Feature importance calculation
  - SHAP value visualization
  - Model explanation generation
  - Fallback mechanisms for different model types

#### 5. Experiment Tracking Layer
- **Component**: MLflow Integration
- **Features**:
  - Experiment logging
  - Model versioning
  - Metrics tracking
  - Artifact storage

#### 6. Data Storage Layer
- **Components**:
  - `data/` - Sample datasets (churn, fraud, insurance)
  - `models/` - Trained model artifacts
  - Local file system storage

### Technical Architecture Requirements

#### Technology Stack
- **Frontend**: Streamlit 1.47.0
- **ML Framework**: AutoGluon 1.2
- **LLM Integration**: OpenAI GPT-4 (API)
- **Interpretability**: SHAP 0.48.0
- **Experiment Tracking**: MLflow 3.1.3
- **Data Processing**: Pandas, NumPy
- **Environment**: Python 3.12

#### Data Flow Sequence
1. User uploads CSV file → Streamlit UI
2. User enters natural language prompt → Streamlit UI
3. Streamlit calls `genie_respond()` → LLM Agent
4. LLM Agent processes prompt + columns → OpenAI API
5. Parsed response returned → Streamlit UI
6. Streamlit calls `build_automl_pipeline()` → Pipeline Builder
7. AutoGluon trains models → Model artifacts saved
8. SHAP generates explanations → Feature importance
9. Results displayed → Streamlit UI
10. User downloads model → Local storage

#### Error Handling & Fallbacks
- Multi-attempt training strategy
- Progressive preset fallback (best_quality → medium_quality → minimal)
- SHAP fallback mechanisms for different model types
- Comprehensive error logging and user feedback

#### Security & Configuration
- Environment variables (.env) for API keys
- .gitignore protection for sensitive files
- Local model storage with unique timestamps

### Architectural Patterns
- **Microservices-like separation** of concerns (UI, LLM, AutoML, Interpretability)
- **Pipeline pattern** for ML workflow
- **Strategy pattern** for fallback mechanisms
- **Factory pattern** for model creation

### Deployment Architecture
- **Local Development**: Streamlit server
- **Production Ready**: Can be deployed to Streamlit Cloud, Heroku, or containerized
- **Scalability**: Stateless design allows horizontal scaling

### Key Architectural Decisions
1. **Streamlit** chosen for rapid UI development and ML integration
2. **AutoGluon** selected for enterprise-grade AutoML capabilities
3. **OpenAI GPT-4** for robust natural language understanding
4. **SHAP** for model-agnostic interpretability
5. **MLflow** for experiment management and reproducibility

## Diagram Requirements

Please generate a comprehensive architecture diagram that includes:

1. **Component boxes** for each layer/service
2. **Data flow arrows** showing the sequence of operations
3. **Technology labels** for each component
4. **Input/Output specifications** for key functions
5. **External dependencies** (OpenAI API, file system)
6. **Error handling paths** and fallback mechanisms
7. **Color coding** for different types of components (UI, ML, Storage, etc.)

The diagram should be suitable for:
- Technical documentation
- System design presentations
- Developer onboarding
- Architecture review discussions

Please create both a high-level overview diagram and a detailed technical diagram showing internal component interactions.

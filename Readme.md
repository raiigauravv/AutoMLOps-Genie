# AutoMLOps-Genie 🧞‍♂️

An intelligent AutoML platform that automatically builds, trains, and deploys machine learning models with natural language prompts.

## Features

- **Natural Language Interface**: Describe your ML task in plain English
- **Automated ML Pipeline**: Automatic data preprocessing, model selection, and training
- **Multiple Problem Types**: Binary classification, multi-class classification, and regression
- **Model Interpretability**: SHAP-based feature importance analysis
- **Modern UI**: Clean, minimal Streamlit interface
- **MLflow Integration**: Experiment tracking and model management

## Quick Start

### Prerequisites

- Python 3.12 (AutoGluon compatibility requirement)
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AutoMLOps-Genie.git
cd AutoMLOps-Genie
```

2. Create and activate virtual environment:
```bash
python3.12 -m venv venv312
source venv312/bin/activate  # On Windows: venv312\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

### Usage

1. Start the application:
```bash
streamlit run ui/minimal_app.py
```

2. Upload your CSV dataset
3. Enter a natural language prompt describing your ML task
4. Let the Genie build and train your model automatically!

## Project Structure

```
AutoMLOps-Genie/
├── ui/
│   └── minimal_app.py          # Streamlit interface
├── genie_agent/
│   └── genie_main.py           # LLM-powered prompt parsing
├── pipelines/
│   └── pipeline_builder.py     # AutoML pipeline implementation
├── data/                       # Sample datasets
├── models/                     # Trained model storage
└── requirements.txt            # Dependencies
```

## Supported ML Tasks

- **Classification**: Binary and multi-class prediction tasks
- **Regression**: Continuous value prediction
- **Automatic Detection**: The system automatically detects problem type

## Technologies Used

- **AutoGluon**: Automated machine learning framework
- **Streamlit**: Web interface
- **OpenAI GPT**: Natural language processing
- **SHAP**: Model interpretability
- **MLflow**: Experiment tracking
- **Pandas/NumPy**: Data manipulation

## Sample Prompts

- "Predict customer churn based on the dataset"
- "Build a model to forecast sales revenue"
- "Create a classifier for fraud detection"
- "Analyze which features are most important for the target"

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For questions and support, please open an issue on GitHub.

# Dockerfile — AutoMLOps Genie
# Builds a container image deployable to Azure Container Apps / Azure App Service.
#
# Build:   docker build -t automlops-genie .
# Run:     docker run -p 8501:8501 -e OPENAI_API_KEY=sk-... automlops-genie
# Azure:   see deploy_azure.sh

FROM python:3.10-slim

# System dependencies (AutoGluon needs libgomp for LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer-cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir pytest

# Copy application source
COPY . .

# Streamlit configuration — bind to 0.0.0.0 so Azure can reach it
RUN mkdir -p /root/.streamlit && cat > /root/.streamlit/config.toml <<'EOF'
[server]
headless = true
address = "0.0.0.0"
port = 8501
enableCORS = false
enableXsrfProtection = false
EOF

# Create directories for model artifacts and MLflow tracking
RUN mkdir -p models mlruns

EXPOSE 8501

# Health check — Azure App Service uses this to decide when the container is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "ui/minimal_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

#!/bin/bash
# deploy_azure.sh — Deploy AutoMLOps Genie to Azure Container Apps
#
# Prerequisites:
#   1. Azure CLI installed: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli
#   2. Logged in: az login
#   3. OPENAI_API_KEY environment variable set
#
# Usage:
#   chmod +x deploy_azure.sh
#   OPENAI_API_KEY=sk-... ./deploy_azure.sh

set -e

# ── Configuration ─────────────────────────────────────────────────────────────
RESOURCE_GROUP="automlops-genie-rg"
LOCATION="eastus"
ACR_NAME="automlopsgenieacr"          # Azure Container Registry name (must be globally unique)
APP_NAME="automlops-genie"
CONTAINER_APP_ENV="automlops-env"
IMAGE_TAG="latest"

echo "===== AutoMLOps Genie — Azure Deployment ====="

# ── 1. Create resource group ──────────────────────────────────────────────────
echo "[1/7] Creating resource group: $RESOURCE_GROUP in $LOCATION"
az group create \
  --name "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --output none

# ── 2. Create Azure Container Registry ───────────────────────────────────────
echo "[2/7] Creating Azure Container Registry: $ACR_NAME"
az acr create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$ACR_NAME" \
  --sku Basic \
  --admin-enabled true \
  --output none

# ── 3. Build and push image to ACR ───────────────────────────────────────────
echo "[3/7] Building and pushing Docker image to ACR"
az acr build \
  --registry "$ACR_NAME" \
  --image "$APP_NAME:$IMAGE_TAG" \
  .

# ── 4. Get ACR credentials ────────────────────────────────────────────────────
echo "[4/7] Retrieving ACR credentials"
ACR_USERNAME=$(az acr credential show --name "$ACR_NAME" --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name "$ACR_NAME" --query "passwords[0].value" -o tsv)
ACR_LOGIN_SERVER=$(az acr show --name "$ACR_NAME" --query loginServer -o tsv)

# ── 5. Create Container Apps environment ─────────────────────────────────────
echo "[5/7] Creating Container Apps environment: $CONTAINER_APP_ENV"
az containerapp env create \
  --name "$CONTAINER_APP_ENV" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --output none

# ── 6. Deploy container app ───────────────────────────────────────────────────
echo "[6/7] Deploying container app: $APP_NAME"
az containerapp create \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --environment "$CONTAINER_APP_ENV" \
  --image "$ACR_LOGIN_SERVER/$APP_NAME:$IMAGE_TAG" \
  --registry-server "$ACR_LOGIN_SERVER" \
  --registry-username "$ACR_USERNAME" \
  --registry-password "$ACR_PASSWORD" \
  --target-port 8501 \
  --ingress external \
  --min-replicas 0 \
  --max-replicas 2 \
  --cpu 1.0 \
  --memory 2.0Gi \
  --env-vars "OPENAI_API_KEY=${OPENAI_API_KEY}" \
  --output none

# ── 7. Print the live URL ─────────────────────────────────────────────────────
echo "[7/7] Deployment complete!"
APP_URL=$(az containerapp show \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query "properties.configuration.ingress.fqdn" -o tsv)

echo ""
echo "✅ AutoMLOps Genie is live at: https://$APP_URL"
echo ""
echo "To tear down all resources:"
echo "  az group delete --name $RESOURCE_GROUP --yes --no-wait"

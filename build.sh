#!/usr/bin/env bash
# This script will run on Render's servers to set up your project.

# Exit immediately if a command exits with a non-zero status.
set -o errexit

echo "Starting build process..."

# 1. Install Python dependencies
echo "Installing Python packages..."
pip install -r requirements.txt

# 2. Create the directory for model weights
mkdir -p backend/ml/weights
echo "Weights directory created."

# 3. Download model weights (REPLACE THESE URLs WITH YOUR OWN)
echo "Downloading model weights..."

curl -L "YOUR_DIRECT_DOWNLOAD_LINK_FOR_hybrid_cnn_model.pth" -o "backend/ml/weights/hybrid_cnn_model.pth"
curl -L "YOUR_DIRECT_DOWNLOAD_LINK_FOR_vision_transformer_model.pth" -o "backend/ml/weights/vision_transformer_model.pth"
curl -L "YOUR_DIRECT_DOWNLOAD_LINK_FOR_high_accuracy_hybrid_model.pth" -o "backend/ml/weights/high_accuracy_hybrid_model.pth"

echo "Model weights downloaded."
echo "Build finished successfully."
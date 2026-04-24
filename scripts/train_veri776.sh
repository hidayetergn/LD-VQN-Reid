#!/bin/bash
# ==============================================================================
# Script to train LD-VQN on the VeRi-776 Dataset
# Reproduces the training phase for Table 3.
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# Configuration
CONFIG_FILE="configs/veri776_ldvqn.yml"
GPU_ID=0  # Default GPU. Can be overridden via command line: GPU_ID=1 bash train.sh

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate a timestamp for the log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/train_veri776_${TIMESTAMP}.log"

echo "[*] Starting LD-VQN Training on VeRi-776..."
echo "[*] Using GPU: ${GPU_ID}"
echo "[*] Logging output to: ${LOG_FILE}"
echo "=============================================================================="

# Run the training script with unbuffered output for real-time logging
CUDA_VISIBLE_DEVICES=${GPU_ID} python -u train.py \
    --config ${CONFIG_FILE} \
    2>&1 | tee ${LOG_FILE}

echo "=============================================================================="
echo "[*] Training complete. Best weights are saved in the output directory."
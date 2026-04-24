#!/bin/bash
# ==============================================================================
# Script to evaluate LD-VQN on the VehicleID Dataset
# Reproduces the progressive test results (Small, Medium, Large) in Table 2.
# ==============================================================================

set -e

CONFIG_FILE="configs/vehicleid_ldvqn.yml"
WEIGHTS="weights/ldvqn_vehicleid_best.pth"
GPU_ID=0

mkdir -p logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/eval_vehicleid_${TIMESTAMP}.log"

# Check if pre-trained weights exist
if [ ! -f "$WEIGHTS" ]; then
    echo "[!] Error: Pre-trained weights not found at $WEIGHTS"
    echo "    Please download them as instructed in the README."
    exit 1
fi

echo "[*] Starting Evaluation on VehicleID Progressive Splits..."
echo "[*] Using Weights: ${WEIGHTS}"
echo "=============================================================================="

CUDA_VISIBLE_DEVICES=${GPU_ID} python -u test.py \
    --config ${CONFIG_FILE} \
    --resume ${WEIGHTS} \
    2>&1 | tee ${LOG_FILE}

echo "=============================================================================="
echo "[*] Evaluation complete."
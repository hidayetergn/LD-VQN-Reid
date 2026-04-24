#!/bin/bash
# ==============================================================================
# Script to generate D-TCR Cross-Modal Attention Heatmaps
# Reproduces the qualitative interpretability results in Figure 9.
# ==============================================================================

set -e

WEIGHTS="weights/ldvqn_veri_best.pth"
SAMPLE_IMG="data/sample_isomorphic_vehicle.jpg"

# The textual prompt designed to isolate a specific local primitive
PROMPT="distinctive rear taillight and obscured license plate"
OUTPUT_DIR="tools/outputs/heatmaps"

echo "[*] Generating Attention Heatmap for D-TCR Module..."
echo "[*] Image: ${SAMPLE_IMG}"
echo "[*] Text Prompt: '${PROMPT}'"
echo "=============================================================================="

python tools/visualize_attention.py \
    --image_path ${SAMPLE_IMG} \
    --text "${PROMPT}" \
    --weight ${WEIGHTS} \
    --output_dir ${OUTPUT_DIR}

echo "=============================================================================="
echo "[*] Heatmap saved to ${OUTPUT_DIR}. Open the image to verify localization."
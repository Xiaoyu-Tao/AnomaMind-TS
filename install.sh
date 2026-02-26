#!/bin/bash
# One-stop installation for AnomaMind / anomaly-detector Agent
set -e

echo "=== Step 1: agentlightning ==="
pip install --upgrade agentlightning

echo "=== Step 2: torch (cu128) ==="
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

echo "=== Step 3: flash-attn ==="
pip install flash-attn --no-build-isolation

echo "=== Step 4: vllm ==="
pip install vllm==0.10.2

echo "=== Step 5: verl ==="
pip install verl==0.5.0

echo "=== Step 6: requirements.txt ==="
pip install -r requirements.txt

echo "=== Done ==="

#!/usr/bin/env bash
# Inside the container. Override per-experiment.
set -euo pipefail
echo "Hello from $(hostname). GPU:"
nvidia-smi --query-gpu=name --format=csv,noheader
# Replace with: python train.py --config configs/config.yaml

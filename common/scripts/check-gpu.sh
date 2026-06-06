#!/usr/bin/env bash
# Quick GPU sanity checks. Run on the EC2/HyperPod node.
nvidia-smi
echo "---"
nvidia-smi -q | grep -E "Persistence|ECC Mode|PCIe.*Width|Driver Version" | head -10
echo "---"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,utilization.memory --format=csv

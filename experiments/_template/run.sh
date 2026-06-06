#!/usr/bin/env bash
# Default lifecycle: load env → build & push → run remotely.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
[ -f push.env ] || { echo "Copy push.env.sample to push.env first."; exit 1; }
source push.env

# 1. Build & push Docker image to ECR
bash "${ROOT}/common/infra/ec2-push.sh"

# 2. Run remotely (EC2 path; for HyperPod use slurm-submit.sh)
IMAGE="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG:-latest}"
bash "${ROOT}/common/infra/ssh-run.sh" "
  aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
  docker pull ${IMAGE}
  docker run --rm --gpus all -v \$HOME/zenmai-results:/results ${IMAGE}
"

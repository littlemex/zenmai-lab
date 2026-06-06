#!/usr/bin/env bash
# Build and push a Docker image to ECR. Driven entirely by env vars.
#
# Required env (typically set in experiment's push.env):
#   ECR_REPOSITORY    Repo name (e.g. "gr00t-train"). Auto-created if missing.
#   DOCKERFILE_PATH   Path to Dockerfile (relative to BUILD_CONTEXT).
#   BUILD_CONTEXT     Docker build context dir (defaults to '.').
#
# Optional env:
#   IMAGE_TAG         Defaults to "latest".
#   BUILD_ARG_<NAME>  Any var with this prefix is passed as --build-arg <NAME>=<value>
#                     (e.g. BUILD_ARG_ACCEPT_EULA=Y → --build-arg ACCEPT_EULA=Y)
#   PLATFORM          Defaults to "linux/amd64".

set -euo pipefail

# load shared env
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "${SCRIPT_DIR}/env" ] && source "${SCRIPT_DIR}/env"

: "${AWS_REGION:?AWS_REGION not set (see common/infra/env)}"
: "${AWS_ACCOUNT_ID:?AWS_ACCOUNT_ID not set (see common/infra/env)}"
: "${ECR_REPOSITORY:?ECR_REPOSITORY not set (see push.env)}"
: "${DOCKERFILE_PATH:?DOCKERFILE_PATH not set (see push.env)}"

BUILD_CONTEXT="${BUILD_CONTEXT:-.}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
PLATFORM="${PLATFORM:-linux/amd64}"
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_URI="${ECR_REGISTRY}/${ECR_REPOSITORY}:${IMAGE_TAG}"

# ensure ECR repo exists
aws ecr describe-repositories \
  --repository-names "${ECR_REPOSITORY}" \
  --region "${AWS_REGION}" >/dev/null 2>&1 \
  || aws ecr create-repository \
       --repository-name "${ECR_REPOSITORY}" \
       --region "${AWS_REGION}"

# login
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${ECR_REGISTRY}"

# collect build args from BUILD_ARG_* env vars
build_args=()
while IFS='=' read -r name value; do
  if [[ "$name" == BUILD_ARG_* ]]; then
    arg_name="${name#BUILD_ARG_}"
    build_args+=(--build-arg "${arg_name}=${value}")
  fi
done < <(env)

echo "Building ${IMAGE_URI} from ${BUILD_CONTEXT}/${DOCKERFILE_PATH}"
docker buildx build \
  --platform "${PLATFORM}" \
  -t "${IMAGE_URI}" \
  -f "${BUILD_CONTEXT}/${DOCKERFILE_PATH}" \
  "${build_args[@]}" \
  --push \
  "${BUILD_CONTEXT}"

echo "Pushed: ${IMAGE_URI}"

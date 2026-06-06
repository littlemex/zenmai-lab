#!/usr/bin/env bash
# Push large local artifacts (results, checkpoints) to S3.
# Wraps `aws s3 sync` with the bucket from common/infra/env.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../infra/env"

: "${S3_BUCKET:?S3_BUCKET not set in common/infra/env}"

SRC="${1:?usage: sync-to-s3.sh <local-dir> <s3-prefix>}"
PREFIX="${2:?usage: sync-to-s3.sh <local-dir> <s3-prefix>}"

aws s3 sync "${SRC}" "s3://${S3_BUCKET}/${PREFIX}/" \
  --exclude "*.tmp" \
  --exclude ".DS_Store"

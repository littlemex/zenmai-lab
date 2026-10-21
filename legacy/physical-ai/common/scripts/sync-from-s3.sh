#!/usr/bin/env bash
# Pull artifacts from S3 to a local dir.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../infra/env"
: "${S3_BUCKET:?S3_BUCKET not set in common/infra/env}"

PREFIX="${1:?usage: sync-from-s3.sh <s3-prefix> <local-dir>}"
DEST="${2:?usage: sync-from-s3.sh <s3-prefix> <local-dir>}"

aws s3 sync "s3://${S3_BUCKET}/${PREFIX}/" "${DEST}"

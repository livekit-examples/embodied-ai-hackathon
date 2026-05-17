#!/usr/bin/env bash
# Rsync data_collection/ (code only) to a training server so you can
# `uv run` a train script there.
#
# Usage:
#   ./scripts/push_to_server.sh <user@host:/remote/path> [extra rsync args...]
#
# Excludes `data/` — push the dataset to HF first via
# scripts/upload_to_hf.sh and let the training job pull it from there.
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <user@host:/remote/path> [extra rsync args...]" >&2
  exit 1
fi

DEST="$1"
shift

cd "$(dirname "$0")/.."

rsync -avz --progress \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '.venv' \
  --exclude '.DS_Store' \
  --exclude '*.pyc' \
  --exclude '.pytest_cache' \
  --exclude '.ruff_cache' \
  --exclude '.mypy_cache' \
  --exclude 'outputs' \
  --exclude 'checkpoints' \
  --exclude 'data' \
  --exclude '.env.local' \
  "$@" \
  ./ "$DEST"

# Also ship the monorepo-root portal.yaml into the destination so anything
# that loads it next to itself finds it on the remote.
if [ -f ../../portal.yaml ]; then
  rsync -avz --progress ../../portal.yaml "$DEST"
fi

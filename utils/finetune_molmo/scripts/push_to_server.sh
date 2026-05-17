#!/usr/bin/env bash
# Rsync finetune_molmo/ (code only) to a training server so you can
# `uv run python train.py` there.
#
# Usage:
#   ./scripts/push_to_server.sh <user@host:/remote/path> [extra rsync args...]
#
# Excludes outputs/, checkpoints/, and .venv/. Datasets are NOT pushed:
# either pull from HF on the remote (DATASET_REPO_ID=you/your-dataset) or
# upload via utils/data_collection/scripts/upload_to_hf.sh first.
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
  --exclude 'wandb' \
  --exclude '.env.local' \
  "$@" \
  ./ "$DEST"

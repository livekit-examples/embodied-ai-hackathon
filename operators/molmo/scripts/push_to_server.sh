#!/usr/bin/env bash
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
  "$@" \
  ./ "$DEST"

# Also ship the monorepo-root portal.yaml into the destination so
# molmo_operator.py finds it next to itself on the remote.
if [ -f ../../portal.yaml ]; then
  rsync -avz --progress ../../portal.yaml "$DEST"
fi

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

# Reconcile the remote venv against the freshly synced pyproject.toml /
# uv.lock. Without this, version bumps (e.g. livekit-portal) land on
# disk but the running .venv keeps the old package.
HOST="${DEST%%:*}"
REMOTE_PATH="${DEST#*:}"
if [ "$HOST" = "$DEST" ] || [ -z "$REMOTE_PATH" ]; then
  echo "push_to_server: DEST '$DEST' is not user@host:/path, skipping uv sync" >&2
else
  echo "push_to_server: running 'uv sync' on $HOST:$REMOTE_PATH"
  ssh "$HOST" "cd '$REMOTE_PATH' && uv sync"
fi

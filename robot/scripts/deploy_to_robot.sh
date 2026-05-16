#!/usr/bin/env bash
# Rsync this robot-only project to the robot host, then print run commands.
set -euo pipefail

if ! command -v rsync >/dev/null 2>&1; then
    echo "rsync not found on this machine" >&2
    exit 1
fi

HERE="$(cd "$(dirname "$0")/.." && pwd)"

_env_get() {
    local key="$1" file="$2"
    [[ -f "$file" ]] || return 0
    local line
    line="$(grep -E "^${key}=" "$file" | tail -1 || true)"
    [[ -n "$line" ]] || return 0
    local value="${line#${key}=}"
    value="${value%$'\r'}"
    [[ "$value" == \"*\" && "$value" == *\" ]] && value="${value:1:-1}"
    [[ "$value" == \'*\' && "$value" == *\' ]] && value="${value:1:-1}"
    printf '%s\n' "$value"
}

if [[ -f "$HERE/.env" ]]; then
    : "${PORTAL_ROBOT_REMOTE:=$(_env_get PORTAL_ROBOT_REMOTE "$HERE/.env")}"
    : "${PORTAL_ROBOT_REMOTE_ROOT:=$(_env_get PORTAL_ROBOT_REMOTE_ROOT "$HERE/.env")}"
fi

REMOTE="${REMOTE:-${PORTAL_ROBOT_REMOTE:-}}"
REMOTE_ROOT="${REMOTE_ROOT:-${PORTAL_ROBOT_REMOTE_ROOT:-~/workspace}}"

if [[ -z "$REMOTE" ]]; then
    echo "REMOTE is empty. Set PORTAL_ROBOT_REMOTE in .env or pass REMOTE=user@host." >&2
    exit 1
fi

IGNORE_FILE="$HERE/scripts/deploy.rsyncignore"

echo "[deploy] remote: $REMOTE:$REMOTE_ROOT"
echo "[deploy] ensuring remote directory exists ..."
ssh "$REMOTE" "mkdir -p '$REMOTE_ROOT/portal-robot-runner'"

echo "[deploy] syncing portal-robot-runner/ ..."
rsync -azP --delete --exclude-from="$IGNORE_FILE" "$HERE/" "$REMOTE:$REMOTE_ROOT/portal-robot-runner/"

cat <<EOF

[deploy] done. On the robot:

  cd $REMOTE_ROOT/portal-robot-runner
  uv sync
  uv run robot.py

EOF

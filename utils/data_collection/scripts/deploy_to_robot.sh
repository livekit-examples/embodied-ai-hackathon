#!/usr/bin/env bash
# Rsync portal-hitl to the robot PC, then print follow-up steps.
#
# Usage:
#   ./scripts/deploy_to_robot.sh
#   REMOTE=user@robot.local ./scripts/deploy_to_robot.sh
#   REMOTE=user@robot.local REMOTE_ROOT=~/code ./scripts/deploy_to_robot.sh
#
# Defaults come from .env (PORTAL_HITL_ROBOT_REMOTE and
# PORTAL_HITL_ROBOT_REMOTE_ROOT). The .env file itself is synced to the
# robot so it picks up LIVEKIT_* / LESLIDER_* without re-entry; per-
# machine overrides go in .env.local, which is NOT synced.
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
    : "${PORTAL_HITL_ROBOT_REMOTE:=$(_env_get PORTAL_HITL_ROBOT_REMOTE "$HERE/.env")}"
    : "${PORTAL_HITL_ROBOT_REMOTE_ROOT:=$(_env_get PORTAL_HITL_ROBOT_REMOTE_ROOT "$HERE/.env")}"
fi

REMOTE="${REMOTE:-${PORTAL_HITL_ROBOT_REMOTE:-}}"
REMOTE_ROOT="${REMOTE_ROOT:-${PORTAL_HITL_ROBOT_REMOTE_ROOT:-~/workspace}}"

if [[ -z "$REMOTE" ]]; then
    echo "REMOTE is empty. Set PORTAL_HITL_ROBOT_REMOTE in .env or pass REMOTE=user@host." >&2
    exit 1
fi

IGNORE_FILE="$HERE/scripts/deploy.rsyncignore"

echo "[deploy] remote: $REMOTE:$REMOTE_ROOT"
echo "[deploy] ensuring remote directory exists ..."
ssh "$REMOTE" "mkdir -p '$REMOTE_ROOT/portal-hitl'"

echo "[deploy] syncing portal-hitl/ ..."
rsync -azP --delete --exclude-from="$IGNORE_FILE" "$HERE/" "$REMOTE:$REMOTE_ROOT/portal-hitl/"

cat <<EOF

[deploy] done. On the robot:

  cd $REMOTE_ROOT/portal-hitl
  uv sync

  # First-time calibration walks through ranges of motion when robot.py
  # connects with no saved file. Saved files live under
  # ~/.cache/huggingface/lerobot/calibration/ keyed by LESLIDER_ID.

  uv run robot.py

EOF

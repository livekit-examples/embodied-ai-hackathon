#!/usr/bin/env bash
# Run robot.py under uv and automatically restart it if it crashes.
#
# Exits cleanly on Ctrl+C (SIGINT) or SIGTERM. Any other exit (including
# non-zero crashes) triggers a restart after a short backoff.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RESTART_DELAY="${RESTART_DELAY:-2}"

CHILD_PID=""
SHUTDOWN=0

shutdown() {
    SHUTDOWN=1
    if [[ -n "$CHILD_PID" ]] && kill -0 "$CHILD_PID" 2>/dev/null; then
        echo "[run_robot] Forwarding shutdown to robot.py (pid $CHILD_PID)..."
        kill -TERM "$CHILD_PID" 2>/dev/null || true
        wait "$CHILD_PID" 2>/dev/null || true
    fi
    exit 0
}

trap shutdown INT TERM

attempt=0
while true; do
    attempt=$((attempt + 1))
    echo "[run_robot] Starting robot.py (attempt #$attempt) at $(date '+%Y-%m-%d %H:%M:%S')"

    uv run robot.py &
    CHILD_PID=$!

    wait "$CHILD_PID"
    exit_code=$?
    CHILD_PID=""

    if [[ "$SHUTDOWN" -eq 1 ]]; then
        exit 0
    fi

    if [[ "$exit_code" -eq 0 ]]; then
        echo "[run_robot] robot.py exited cleanly (code 0). Restarting in ${RESTART_DELAY}s..."
    else
        echo "[run_robot] robot.py crashed with exit code $exit_code. Restarting in ${RESTART_DELAY}s..."
    fi

    sleep "$RESTART_DELAY"
done

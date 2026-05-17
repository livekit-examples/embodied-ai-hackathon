#!/usr/bin/env bash
# Push a locally-recorded LeRobot dataset to the Hugging Face Hub with the v3.0 tag.
#
# Usage:
#   scripts/upload_to_hf.sh <dataset_path> <hf_repo_id> [--private] [--branch <name>]
#
# Example:
#   scripts/upload_to_hf.sh data/local/portal-hitl pham-tuan-binh/portal-hitl
#
# Requires: HF_TOKEN env var, or a prior `huggingface-cli login`.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <dataset_path> <hf_repo_id> [--private] [--branch <name>]" >&2
    exit 1
fi

DATASET_PATH="$1"
HF_REPO_ID="$2"
shift 2

PRIVATE="False"
BRANCH=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --private) PRIVATE="True"; shift ;;
        --branch) BRANCH="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ ! -f "${DATASET_PATH}/meta/info.json" ]]; then
    echo "no meta/info.json under ${DATASET_PATH}; not a LeRobot dataset" >&2
    exit 1
fi

ABS_PATH="$(cd "${DATASET_PATH}" && pwd)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

unset HF_HUB_OFFLINE

uv run python - <<PY
import json
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = Path("${ABS_PATH}")
info = json.loads((root / "meta" / "info.json").read_text())
codebase_version = info.get("codebase_version", "v3.0")
robot_type = info.get("robot_type", "unknown")

print(f"resuming dataset: root={root} version={codebase_version} robot={robot_type}")
ds = LeRobotDataset.resume(repo_id="${HF_REPO_ID}", root=root)

tags = ["LeRobot", codebase_version, robot_type]
branch = "${BRANCH}" or None
private = ${PRIVATE}

print(f"pushing to {ds.repo_id} (tags={tags}, branch={branch}, private={private})")
ds.push_to_hub(
    branch=branch,
    tags=tags,
    tag_version=True,
    push_videos=True,
    private=private,
)
print("done")
PY

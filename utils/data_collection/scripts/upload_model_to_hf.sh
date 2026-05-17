#!/usr/bin/env bash
# Push a locally-trained LeRobot policy checkpoint to the Hugging Face Hub.
#
# Accepts any of three checkpoint path shapes (matches what `policies/<algo>/train.py`
# writes under OUTPUT_ROOT/RUN_NAME):
#   1. Run output dir, e.g. /outputs/act-leslider_steps20000_bs64_lr5e-5_chunk50
#      -> picks the highest valid checkpoints/<step>/pretrained_model.
#   2. Step dir, e.g. .../checkpoints/025000
#      -> uploads its pretrained_model/.
#   3. A pretrained_model/ dir directly.
#
# Usage:
#   scripts/upload_model_to_hf.sh <checkpoint_path> <hf_repo_id> \
#       [--private] [--branch <name>] [--commit-message <msg>] [--tag <tag>]...
#
# Example:
#   scripts/upload_model_to_hf.sh \
#       /outputs/act-leslider_steps20000_bs64_lr5e-5_chunk50 \
#       pham-tuan-binh/act-portal-hitl
#
# Requires: HF_TOKEN env var, or a prior `huggingface-cli login`.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <checkpoint_path> <hf_repo_id> [--private] [--branch <name>] [--commit-message <msg>] [--tag <tag>]..." >&2
    exit 1
fi

CHECKPOINT_PATH="$1"
HF_REPO_ID="$2"
shift 2

PRIVATE="False"
BRANCH=""
COMMIT_MSG=""
EXTRA_TAGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --private) PRIVATE="True"; shift ;;
        --branch) BRANCH="$2"; shift 2 ;;
        --commit-message) COMMIT_MSG="$2"; shift 2 ;;
        --tag) EXTRA_TAGS+=("$2"); shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ ! -d "${CHECKPOINT_PATH}" ]]; then
    echo "checkpoint path not found: ${CHECKPOINT_PATH}" >&2
    exit 1
fi

ABS_PATH="$(cd "${CHECKPOINT_PATH}" && pwd)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

unset HF_HUB_OFFLINE

EXTRA_TAGS_JSON="$(printf '%s\n' "${EXTRA_TAGS[@]+"${EXTRA_TAGS[@]}"}" | uv run python -c 'import json,sys; print(json.dumps([l for l in sys.stdin.read().splitlines() if l]))')"

uv run python - <<PY
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

src = Path("${ABS_PATH}")
extra_tags = json.loads('''${EXTRA_TAGS_JSON}''')


def find_pretrained_model_dir(root: Path) -> Path:
    """Resolve <run>/<step>/pretrained_model from any of the three accepted shapes."""
    if (root / "config.json").is_file() and (root / "train_config.json").is_file():
        return root  # already a pretrained_model dir

    if (root / "pretrained_model").is_dir():
        return root / "pretrained_model"  # step dir

    ckpt_root = root / "checkpoints"
    if not ckpt_root.is_dir():
        raise SystemExit(
            f"{root} is not a pretrained_model/, step, or run output dir "
            f"(no config.json, no pretrained_model/, no checkpoints/)"
        )
    candidates = sorted(
        (p for p in ckpt_root.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
        reverse=True,
    )
    for c in candidates:
        pm = c / "pretrained_model"
        cfg = pm / "train_config.json"
        if pm.is_dir() and cfg.is_file():
            return pm
    raise SystemExit(f"no valid checkpoint under {ckpt_root}")


pm_dir = find_pretrained_model_dir(src)
print(f"uploading from: {pm_dir}", flush=True)

train_cfg = json.loads((pm_dir / "train_config.json").read_text())
policy_type = (train_cfg.get("policy") or {}).get("type") or "unknown"
dataset_repo_id = (train_cfg.get("dataset") or {}).get("repo_id") or ""
print(f"policy_type={policy_type} dataset={dataset_repo_id}", flush=True)

tags = ["LeRobot", "robotics", f"lerobot:{policy_type}", policy_type]
tags.extend(extra_tags)
tags = sorted({t for t in tags if t})

branch = "${BRANCH}" or None
commit_msg = "${COMMIT_MSG}" or f"Upload {policy_type} checkpoint"
private = ${PRIVATE}

api = HfApi()
api.create_repo(
    repo_id="${HF_REPO_ID}",
    repo_type="model",
    private=private,
    exist_ok=True,
)

print(f"pushing to {api.endpoint}/${HF_REPO_ID} "
      f"(branch={branch}, private={private}, tags={tags})", flush=True)

if branch:
    try:
        api.create_branch(repo_id="${HF_REPO_ID}", branch=branch, exist_ok=True)
    except Exception as e:
        print(f"create_branch warning: {e}", file=sys.stderr)

api.upload_folder(
    folder_path=str(pm_dir),
    repo_id="${HF_REPO_ID}",
    repo_type="model",
    revision=branch,
    commit_message=commit_msg,
)

# Stamp tags into the model card so the repo is discoverable as a LeRobot policy.
try:
    from huggingface_hub import ModelCard, ModelCardData
    try:
        card = ModelCard.load("${HF_REPO_ID}")
        card.data.tags = sorted(set((card.data.tags or []) + tags))
    except Exception:
        card = ModelCard.from_template(
            ModelCardData(tags=tags, library_name="lerobot"),
            model_id="${HF_REPO_ID}",
        )
    if not getattr(card.data, "library_name", None):
        card.data.library_name = "lerobot"
    card.push_to_hub(repo_id="${HF_REPO_ID}", repo_type="model", revision=branch)
except Exception as e:
    print(f"model card update warning: {e}", file=sys.stderr)

print("done")
PY

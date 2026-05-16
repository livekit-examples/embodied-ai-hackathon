#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
sky launch -c molmo "$@" skypilot/inference.yaml

#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR%/scripts}"
python "${PROJECT_ROOT}/eval/eval_harness.py" --tests "${PROJECT_ROOT}/eval/tests.json" "$@"
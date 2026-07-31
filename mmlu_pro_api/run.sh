#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.."; pwd)"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"

exec "$PYTHON" "$ROOT/mmlu_pro_api/evaluate.py" "$@"

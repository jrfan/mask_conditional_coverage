#!/usr/bin/env bash
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec "${PYTHON_BIN:-python}" -B "$ROOT/main.py" --group 07 "$@"

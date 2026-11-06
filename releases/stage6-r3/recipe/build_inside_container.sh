#!/usr/bin/env bash
# Sanitized entry point; dependencies.json retains the successful target flags.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
exec /usr/bin/python3 -B "$ROOT/scripts/launcher.py" inside-build "$@"

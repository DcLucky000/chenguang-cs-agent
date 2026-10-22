#!/usr/bin/env bash
# Merge gate: a non-zero eval exit fails the script (set -e).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python3 -m chenguang_agent eval
python3 -m unittest discover -s tests -v

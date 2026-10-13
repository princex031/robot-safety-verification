#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT="outputs/audit-${STAMP}.txt"
mkdir -p outputs
exec > >(tee "$REPORT") 2>&1

echo "=== ROBOT SAFETY REPOSITORY AUDIT ==="
echo "UTC: $STAMP"
echo "ROOT: $ROOT"
echo
echo "=== GIT STATUS ==="
git status --short
echo
echo "=== TRACKED FILE INVENTORY ==="
git ls-files
echo
echo "=== PYTHON VERSION ==="
python3 --version
echo
echo "=== SYNTAX CHECK ==="
python3 -m py_compile simulator/experiment_suite.py tests/test_experiment_suite.py
echo
echo "=== UNIT TESTS ==="
python3 -m unittest discover -s tests -p 'test_*.py' -v
echo
echo "=== SMALL SMOKE EXPERIMENT ==="
python3 simulator/experiment_suite.py --episodes 1000
echo
echo "=== HASHES ==="
find simulator tests configs scripts -type f -print0 |
  sort -z | xargs -0 sha256sum
echo
echo "=== OUTPUT FILES ==="
find outputs -maxdepth 1 -type f -printf '%p (%s bytes)\n' | sort
echo
echo "Audit finished. Report: $REPORT"
echo "NOTE: smoke test uses 1000 episodes; run full suite separately."

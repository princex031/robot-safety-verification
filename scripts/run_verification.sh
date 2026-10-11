#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_verification.sh path/to/dataset.csv [threshold]"
  exit 2
fi
python -m unittest discover -s tests -v
python src/main.py --dataset "$1" --threshold "${2:-0.80}"

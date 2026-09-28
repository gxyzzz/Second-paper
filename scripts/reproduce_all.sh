#!/usr/bin/env bash
set -euo pipefail
GPU=0
if [ "$#" -gt 0 ]; then
  GPU="$1"
fi
for DATASET in baby sports elec; do
  python scripts/reproduce.py --dataset "$DATASET" --stage all --gpu "$GPU"
done

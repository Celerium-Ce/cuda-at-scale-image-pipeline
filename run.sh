#!/usr/bin/env bash
# Builds the project (if needed) and runs the batch Gaussian-blur +
# Sobel edge-detection pipeline over data/input, writing results to
# data/output and a timestamped proof-of-execution log to logs/.
#
# Usage: ./run.sh [extra batch_edge_detect args, e.g. --limit 10]
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d "data/input" ] || [ -z "$(ls -A data/input 2>/dev/null)" ]; then
  echo "data/input is empty - generating synthetic dataset..."
  python3 data/generate_dataset.py
fi

make all

mkdir -p logs
log_file="logs/run_$(date +%Y%m%d_%H%M%S).log"

./bin/batch_edge_detect --input data/input --output data/output \
    --log "$log_file" "$@"

echo "Done. Log written to $log_file"

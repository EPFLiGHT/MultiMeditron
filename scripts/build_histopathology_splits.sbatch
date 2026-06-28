#!/bin/bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/lightscratch/users/cljordan/multimeditron/scripts/logs/build_histopathology_splits_%j.out

set -euo pipefail

REPO=/lightscratch/users/cljordan/multimeditron
cd "$REPO"

source .venv/bin/activate

mkdir -p scripts/logs

echo "=== build_histopathology_splits ==="
echo "Started: $(date)"
echo "Benchmark: ${1:-all}"

python scripts/build_histopathology_splits.py \
  --benchmark "${1:-all}" \
  --histo-output-dir benchmark_splits/multimediset/histopathology \
  --ct-output-dir    benchmark_splits/multimediset/ct

echo "Done: $(date)"

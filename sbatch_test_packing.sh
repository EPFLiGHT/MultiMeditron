#!/bin/bash
#SBATCH --job-name test-packing
#SBATCH --output /users/surech/meditron/reports/R-%x.%j.out
#SBATCH --error /users/surech/meditron/reports/R-%x.%j.err
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 72
#SBATCH --time 00:10:00
#SBATCH --partition debug
#SBATCH -A a127

# Run sequence-packing unit tests inside the multimeditron container.
# Submit:
#   sbatch sbatch_test_packing.sh

REPO_DIR=/users/surech/meditron/MultiMeditron
export PYTHONPATH=$REPO_DIR/src:$PYTHONPATH

echo "START TIME: $(date)"
echo "REPO: $REPO_DIR"

srun \
  --ntasks=1 \
  --environment /users/surech/.edf/multimeditron.toml \
  --export=ALL \
  bash -c "
    cd $REPO_DIR
    python3 tests/test_packing.py
  "

echo "END TIME: $(date)"

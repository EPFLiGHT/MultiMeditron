#!/bin/bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_%j.out

source .venv/bin/activate

cd /users/cjordan/meditron/MultiMeditron/src/multimeditron/experts

python train_new_pipeline.py \
  --config_file configurations/all_medical_and_general_baseline_random_config_1.yaml
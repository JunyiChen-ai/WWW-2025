#!/bin/bash
#SBATCH --job-name=exmrd_main
#SBATCH --output=train_results/main_%j.out
#SBATCH --error=train_results/main_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Usage:
#   sbatch src/run_main_sbatch.sh [Dataset] [ConfigName]
# Examples:
#   sbatch src/run_main_sbatch.sh FakeSV ExMRD_FakeSV
#   sbatch src/run_main_sbatch.sh FakeSV ExMRD_Retrieval_FakeSV

DATASET=${1:-FakeSV}
CONF_NAME=${2:-ExMRD_FakeSV}

echo "Starting ExMRD main run: dataset=${DATASET}, config=${CONF_NAME}"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | Time: $(date)"

# Project root
cd /data/jehc223/ExMRD_ours

# Prepare output dirs
mkdir -p train_results

# Conda
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate ExMRD

# GPU info
nvidia-smi || true

# Run
echo "python src/main.py dataset=${DATASET} +override hydra.run.dir=train_results/${DATASET}_$(date +%Y%m%d_%H%M%S)"
python src/main.py dataset=${DATASET} hydra.run.dir="train_results/${DATASET}_$(date +%Y%m%d_%H%M%S)" --config-name "${CONF_NAME}"

echo "Main run finished at $(date)"


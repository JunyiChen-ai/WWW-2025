#!/bin/bash
#SBATCH --job-name=exmrd_main
#SBATCH --output=train_results/main_%j.out
#SBATCH --error=train_results/main_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Usage:
#   sbatch src/run_main_sbatch.sh [ConfigName]
# Examples:
#   sbatch src/run_main_sbatch.sh ExMRD_FakeSV
#   sbatch src/run_main_sbatch.sh ExMRD_Retrieval_FakeSV

CONF_NAME=${1:-ExMRD_FakeSV}

echo "Starting ExMRD main run: config=${CONF_NAME}"
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
RUN_DIR="train_results/${CONF_NAME}_$(date +%Y%m%d_%H%M%S)"
echo "python src/main.py hydra.run.dir=${RUN_DIR} --config-name ${CONF_NAME}"
python src/main.py hydra.run.dir="${RUN_DIR}" --config-name "${CONF_NAME}"

echo "Main run finished at $(date)"

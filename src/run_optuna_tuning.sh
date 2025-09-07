#!/bin/bash
#SBATCH --job-name=optuna_tuning
#SBATCH --output=optuna_results/optuna_%j.out
#SBATCH --error=optuna_results/optuna_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Parse dataset argument (default to FakeTT if not provided)
DATASET=${1:-FakeTT}
TRIALS=${2:-2000}

echo "Starting Optuna hyperparameter tuning for dataset: $DATASET"
echo "Number of trials: $TRIALS"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Time: $(date)"

# Navigate to project directory
cd /data/jehc223/ExMRD_ours

# Create optuna_results directory if it doesn't exist
mkdir -p optuna_results

# Activate conda environment
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate ExMRD

# Display GPU info
nvidia-smi

# Run hyperparameter tuning
echo "Running: python src/optuna_hyperparameter_tuning.py --dataset $DATASET --trials $TRIALS"
python src/optuna_hyperparameter_tuning.py --dataset $DATASET --trials $TRIALS

echo "Optuna tuning for $DATASET completed at $(date)"
echo "Results saved to: optuna_results/best_hyperparameters_${DATASET,,}.json"
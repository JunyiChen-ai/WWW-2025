#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --output=preprocess/slurm_outputs/%j.out
#SBATCH --error=preprocess/slurm_outputs/%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Parse arguments
SCRIPT_NAME=${1?"Error: Please provide Python script name as first argument"}
SCRIPT_ARGS=${2:-""}

echo "======================================"
echo "SLURM Job Information"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Script: $SCRIPT_NAME"
echo "Arguments: $SCRIPT_ARGS"
echo "======================================"

# Navigate to project directory
cd /data/jehc223/ExMRD_ours

# Create output directory if it doesn't exist
mkdir -p preprocess/slurm_outputs

# Activate conda environment
echo "Activating conda environment..."
source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate ExMRD

# Display environment info
echo ""
echo "Environment Information:"
echo "Python: $(which python)"
echo "Current directory: $(pwd)"
echo "GPU info:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

echo ""
echo "======================================"
echo "Starting execution..."
echo "======================================"

# Determine script path
if [[ $SCRIPT_NAME == *.py ]]; then
    # If it's already a .py file, check if it's in preprocess/
    if [[ -f "preprocess/$SCRIPT_NAME" ]]; then
        FULL_SCRIPT_PATH="preprocess/$SCRIPT_NAME"
    elif [[ -f "$SCRIPT_NAME" ]]; then
        FULL_SCRIPT_PATH="$SCRIPT_NAME"
    else
        echo "ERROR: Script not found: $SCRIPT_NAME"
        echo "Searched in:"
        echo "  - preprocess/$SCRIPT_NAME" 
        echo "  - $SCRIPT_NAME"
        exit 1
    fi
else
    # If no .py extension, assume it's in preprocess/
    FULL_SCRIPT_PATH="preprocess/${SCRIPT_NAME}.py"
    if [[ ! -f "$FULL_SCRIPT_PATH" ]]; then
        echo "ERROR: Script not found: $FULL_SCRIPT_PATH"
        exit 1
    fi
fi

echo "Running: python $FULL_SCRIPT_PATH $SCRIPT_ARGS"
echo ""

# Run the Python script
if [[ -z "$SCRIPT_ARGS" ]]; then
    python "$FULL_SCRIPT_PATH"
else
    eval "python $FULL_SCRIPT_PATH $SCRIPT_ARGS"
fi

EXIT_CODE=$?

echo ""
echo "======================================"
echo "Execution completed"
echo "======================================"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "Job ID: $SLURM_JOB_ID"

if [[ $EXIT_CODE -eq 0 ]]; then
    echo "SUCCESS: Script completed successfully"
else
    echo "ERROR: Script failed with exit code $EXIT_CODE"
fi

echo "======================================"

exit $EXIT_CODE
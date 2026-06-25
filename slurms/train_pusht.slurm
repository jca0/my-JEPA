#!/bin/bash
#SBATCH --job-name=myjepa_pusht
#SBATCH --partition=mit_normal_gpu
#SBATCH --output=logs/train_pusht_%j.out
#SBATCH --error=logs/train_pusht_%j.err
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:l40s:1

# Create logs directory if it doesn't exist
mkdir -p logs

# Print some info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Activate virtual environment
source /home/jingcao/workspace/my-JEPA/.venv/bin/activate

# Set data directory to scratch
export DATA_DIR=/scratch/jingcao/my-JEPA

# Create datasets directory on this node's scratch
mkdir -p /scratch/jingcao/my-JEPA/datasets

# Download dataset if not already present
if [ ! -f /scratch/jingcao/my-JEPA/datasets/pusht_expert_train.h5 ]; then
    echo "Downloading dataset from Hugging Face..."
    cd /scratch/jingcao/my-JEPA/datasets
    hf download quentinll/lewm-pusht --repo-type=dataset --local-dir .

    # Decompress if compressed
    if [ -f pusht_expert_train.h5.zst ]; then
        echo "Decompressing dataset..."
        zstd -d pusht_expert_train.h5.zst
        echo "Decompression complete"
    fi
else
    echo "Dataset already exists on this node"
fi

# Change to the working directory
cd /home/jingcao/workspace/my-JEPA

# Run training
python train.py data=pusht model=mobilenet subdir=pusht_mobilenet_001

echo "Job finished at: $(date)"

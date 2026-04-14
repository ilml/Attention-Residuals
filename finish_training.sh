#!/bin/bash
#SBATCH --job-name=attnres-finish
#SBATCH --account=coreai_dlalgo_genai
#SBATCH --partition=batch_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/finish_%j.out
#SBATCH --error=logs/finish_%j.err

# Finish all incomplete experiments on a single node (4 GPUs).
# Loads the latest checkpoint and trains the remaining steps.
# Single node = no multi-node NCCL crashes.

set -e
mkdir -p logs
cd /home/tolong/work/Attention-Residuals

export WANDB_MODE=offline
export OMP_NUM_THREADS=4

DATA_DIR="/lustre/fsw/portfolios/coreai/users/tolong/data/nemotron_tokenized_150B"
SAVE_DIR="checkpoints_paper"

echo "========================================"
echo "FINISH ALL INCOMPLETE EXPERIMENTS"
echo "Job: $SLURM_JOB_ID | Node: $SLURM_NODELIST | $(date)"
echo "========================================"

CONFIGS=("194M" "241M" "296M" "436M" "528M")
VARIANTS=("baseline" "full_attnres" "block_attnres")

for config in "${CONFIGS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
        echo ""
        echo ">>> Config=$config Variant=$variant $(date)"

        torchrun --nproc_per_node=4 train.py \
            --config "$config" \
            --variant "$variant" \
            --large \
            --data_dir "$DATA_DIR" \
            --save_dir "$SAVE_DIR" \
            --wandb_project attn-residuals-paper \
            --val_interval 500 \
            --log_interval 50 \
            --save_interval 5000 \
        || echo "WARNING: $config/$variant failed"

        echo "    Done: $config / $variant"
    done
done

echo ""
echo "=========================================="
echo " Running analysis..."
echo "=========================================="
python3 analyze.py --results_dir "$SAVE_DIR" --output scaling_law_paper.png
echo "All done: $(date)"

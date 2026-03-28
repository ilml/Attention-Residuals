#!/bin/bash
#SBATCH --job-name=attnres-miss
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=batch_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/missing_%j.out
#SBATCH --error=logs/missing_%j.err

set -e
mkdir -p logs

cd /home/tolong/work/Attention-Residuals

export WANDB_MODE=offline
export OMP_NUM_THREADS=4

DATA_DIR="/lustre/fsw/portfolios/coreai/users/tolong/data/nemotron_tokenized"
SAVE_DIR="checkpoints_large"
NPROC=4

echo "========================================"
echo "Job: $SLURM_JOB_ID | Node: $SLURM_NODELIST | GPUs: $NPROC"
echo "Running MISSING experiments on single node"
echo "Start: $(date)"
echo "========================================"

# These are the experiments that failed on multi-node
EXPERIMENTS=(
    "172M block_attnres"
    "231M baseline"
    "231M block_attnres"
    "313M block_attnres"
    "401M baseline"
    "401M full_attnres"
    "401M block_attnres"
)

for exp in "${EXPERIMENTS[@]}"; do
    config=$(echo "$exp" | cut -d' ' -f1)
    variant=$(echo "$exp" | cut -d' ' -f2)

    echo ""
    echo ">>> Config=$config Variant=$variant $(date)"

    torchrun --nproc_per_node=$NPROC train.py \
        --config "$config" \
        --variant "$variant" \
        --large \
        --data_dir "$DATA_DIR" \
        --save_dir "$SAVE_DIR" \
        --wandb_project attn-residuals-large \
        --val_interval 200 \
        --log_interval 20 \
        --save_interval 0 \
    || echo "WARNING: $config/$variant failed"

    echo "    Done: $config / $variant"
done

echo ""
echo "=========================================="
echo " Running analysis..."
echo "=========================================="
python3 analyze.py --results_dir "$SAVE_DIR" --output scaling_law_large.png
echo "End: $(date)"

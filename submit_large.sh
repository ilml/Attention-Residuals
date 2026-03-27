#!/bin/bash
#SBATCH --job-name=attnres-large
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=batch_long
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/large_%j.out
#SBATCH --error=logs/large_%j.err
#SBATCH --exclusive

set -e
mkdir -p logs

cd /home/tolong/work/Attention-Residuals

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=4
export WANDB_MODE=offline

NNODES=$SLURM_NNODES
NPROC=4
TOTAL_GPUS=$((NNODES * NPROC))
DATA_DIR="/lustre/fsw/portfolios/coreai/users/tolong/data/nemotron_tokenized"
SAVE_DIR="checkpoints_large"

mkdir -p "$SAVE_DIR"

echo "========================================"
echo "Job: $SLURM_JOB_ID | Nodes: $NNODES ($SLURM_NODELIST) | GPUs: $TOTAL_GPUS"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Start: $(date)"
echo "========================================"

CONFIGS=("124M" "172M" "231M" "313M" "401M")
VARIANTS=("baseline" "full_attnres" "block_attnres")

EXP=0
TOTAL=$((${#CONFIGS[@]} * ${#VARIANTS[@]}))

for config in "${CONFIGS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
        EXP=$((EXP + 1))
        echo ""
        echo ">>> [$EXP/$TOTAL] Config=$config Variant=$variant $(date)"

        # Use a unique port per experiment to avoid stale rendezvous state
        RDZV_PORT=$((29500 + EXP))

        srun torchrun \
            --nproc_per_node=$NPROC \
            --nnodes=$NNODES \
            --rdzv_backend=c10d \
            --rdzv_endpoint="$MASTER_ADDR:$RDZV_PORT" \
            train.py \
            --config "$config" \
            --variant "$variant" \
            --large \
            --data_dir "$DATA_DIR" \
            --save_dir "$SAVE_DIR" \
            --wandb_project attn-residuals-large \
            --val_interval 200 \
            --log_interval 20 \
            --save_interval 0 \
        || echo "WARNING: $config/$variant failed, continuing..."

        echo "    Done: $config / $variant"
        sleep 5  # Brief pause between experiments
    done
done

echo ""
echo "=========================================="
echo " All experiments done. Running analysis..."
echo "=========================================="

python3 analyze.py --results_dir "$SAVE_DIR" --output scaling_law_large.png

echo "End: $(date)"

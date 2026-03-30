#!/bin/bash
#SBATCH --job-name=attnres-paper
#SBATCH --account=coreai_dlalgo_genai
#SBATCH --partition=batch_long
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/paper_%j.out
#SBATCH --error=logs/paper_%j.err
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
DATA_DIR="/lustre/fsw/portfolios/coreai/users/tolong/data/nemotron_tokenized_150B"
SAVE_DIR="checkpoints_paper"

mkdir -p "$SAVE_DIR"

echo "========================================"
echo "FULL PAPER REPRODUCTION"
echo "Job: $SLURM_JOB_ID | Nodes: $NNODES | GPUs: $TOTAL_GPUS"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Data: $DATA_DIR"
echo "Start: $(date)"
echo "========================================"

CONFIGS=("194M" "241M" "296M" "436M" "528M")
VARIANTS=("baseline" "full_attnres" "block_attnres")

EXP=0
TOTAL=$((${#CONFIGS[@]} * ${#VARIANTS[@]}))

for config in "${CONFIGS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
        EXP=$((EXP + 1))
        RDZV_PORT=$((29500 + EXP))

        echo ""
        echo ">>> [$EXP/$TOTAL] Config=$config Variant=$variant $(date)"

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
            --wandb_project attn-residuals-paper \
            --val_interval 500 \
            --log_interval 50 \
            --save_interval 5000 \
        || echo "WARNING: $config/$variant failed, continuing..."

        echo "    Done: $config / $variant"
        sleep 10
    done
done

echo ""
echo "=========================================="
echo " All experiments done. Running analysis..."
echo "=========================================="

python3 analyze.py --results_dir "$SAVE_DIR" --output scaling_law_paper.png

echo "End: $(date)"

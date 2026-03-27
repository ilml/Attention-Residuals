#!/bin/bash
# Run all 15 scaling law experiments (5 sizes x 3 variants)
# Usage: bash run_scaling.sh [--large]
#   --large: use large-scale configs with full Nemotron dataset

set -e

LARGE_FLAG=""
SAVE_DIR="checkpoints"
if [[ "$1" == "--large" ]]; then
    LARGE_FLAG="--large"
    SAVE_DIR="checkpoints_large"
fi

export WANDB_MODE=offline
mkdir -p "$SAVE_DIR"

CONFIGS=("124M" "172M" "231M" "313M" "401M")
VARIANTS=("baseline" "full_attnres" "block_attnres")

# Detect GPU count
if [ -n "$SLURM_NTASKS_PER_NODE" ] && [ -n "$SLURM_NNODES" ]; then
    NPROC=$SLURM_NTASKS_PER_NODE
    NNODES=$SLURM_NNODES
else
    NPROC=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 4)
    NNODES=1
fi
TOTAL_GPUS=$((NPROC * NNODES))

echo "=========================================="
echo " Attention Residuals - Scaling Law Sweep"
echo "=========================================="
echo "GPUs: $TOTAL_GPUS ($NNODES nodes x $NPROC GPUs)"
echo "Configs: ${CONFIGS[*]}"
echo "Variants: ${VARIANTS[*]}"
echo "Save dir: $SAVE_DIR"
echo "Large: ${LARGE_FLAG:-no}"
echo "=========================================="

EXP=0
TOTAL=$((${#CONFIGS[@]} * ${#VARIANTS[@]}))

for config in "${CONFIGS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
        EXP=$((EXP + 1))
        echo ""
        echo ">>> [$EXP/$TOTAL] Config=$config  Variant=$variant  $(date)"

        torchrun \
            --nproc_per_node=$NPROC \
            --nnodes=$NNODES \
            --rdzv_backend=c10d \
            --rdzv_endpoint="${MASTER_ADDR:-localhost}:${MASTER_PORT:-29500}" \
            train.py \
            --config "$config" \
            --variant "$variant" \
            --save_dir "$SAVE_DIR" \
            --wandb_project attn-residuals \
            --val_interval 200 \
            --log_interval 20 \
            --save_interval 1000 \
            $LARGE_FLAG

        echo "    Done: $config / $variant"
    done
done

echo ""
echo "=========================================="
echo " All experiments complete! Running analysis..."
echo "=========================================="

python3 analyze.py --results_dir "$SAVE_DIR" --output scaling_law.png
echo "Done."

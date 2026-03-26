#!/bin/bash
# Run all 15 scaling law experiments (5 sizes x 3 variants)
# Usage: bash run_scaling.sh [WANDB_PROJECT]

set -e

WANDB_PROJECT="${1:-attn-residuals}"
SAVE_DIR="checkpoints"
NPROC=4

export WANDB_MODE=offline

mkdir -p "$SAVE_DIR"

CONFIGS=("124M" "172M" "231M" "313M" "401M")
VARIANTS=("baseline" "full_attnres" "block_attnres")

echo "=========================================="
echo " Attention Residuals - Scaling Law Sweep"
echo "=========================================="
echo "Configs: ${CONFIGS[*]}"
echo "Variants: ${VARIANTS[*]}"
echo "Total experiments: $((${#CONFIGS[@]} * ${#VARIANTS[@]}))"
echo "GPUs: $NPROC"
echo "wandb project: $WANDB_PROJECT (offline mode)"
echo "=========================================="

EXP=0
TOTAL=$((${#CONFIGS[@]} * ${#VARIANTS[@]}))

for config in "${CONFIGS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
        EXP=$((EXP + 1))
        echo ""
        echo ">>> [$EXP/$TOTAL] Config=$config  Variant=$variant"
        echo "    $(date)"

        torchrun --nproc_per_node=$NPROC train.py \
            --config "$config" \
            --variant "$variant" \
            --wandb_project "$WANDB_PROJECT" \
            --save_dir "$SAVE_DIR" \
            --val_interval 100 \
            --log_interval 10

        echo "    Done: $config / $variant"
    done
done

echo ""
echo "=========================================="
echo " All experiments complete! Running analysis..."
echo "=========================================="

python3 analyze.py --results_dir "$SAVE_DIR" --output scaling_law.png

echo "Done. Results in $SAVE_DIR/, plot in scaling_law.png"
echo "To sync wandb runs: wandb sync wandb/offline-run-*"

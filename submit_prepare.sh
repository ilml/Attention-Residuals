#!/bin/bash
#SBATCH --job-name=prepare-data
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=04:00:00
#SBATCH --output=logs/prepare_%j.out
#SBATCH --error=logs/prepare_%j.err

set -e
mkdir -p logs

cd /home/tolong/work/Attention-Residuals

echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST | $(date)"

python3 prepare_data.py \
    --data_dir /lustre/fsw/portfolios/coreai/users/tolong/data/nemotron \
    --out_dir /lustre/fsw/portfolios/coreai/users/tolong/data/nemotron_tokenized \
    --max_tokens 12000000000 \
    --seq_len 8192 \
    --workers 48 \
    --min_age 600

echo "Done: $(date)"

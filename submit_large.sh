#!/bin/bash
#SBATCH --job-name=attnres-large
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=batch_long
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/large_%j.out
#SBATCH --error=logs/large_%j.err
#SBATCH --exclusive

set -e
mkdir -p logs

cd /home/tolong/work/Attention-Residuals

# Multi-node NCCL config
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=eth0
export OMP_NUM_THREADS=4

echo "========================================"
echo "Job: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES x $SLURM_NTASKS_PER_NODE GPUs = $((SLURM_NNODES * SLURM_NTASKS_PER_NODE)) total"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Start: $(date)"
echo "========================================"

# Verify GPU access on first node
srun --nodes=1 --ntasks=1 bash -c 'nvidia-smi | head -5; python3 -c "import torch; print(f\"PyTorch {torch.__version__}, GPUs: {torch.cuda.device_count()}\")"'
echo ""

# Run all 15 experiments
srun bash run_scaling.sh --large

echo ""
echo "End: $(date)"

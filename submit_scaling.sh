#!/bin/bash
#SBATCH --job-name=attn-res-scaling
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --partition=batch_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/scaling_%j.out
#SBATCH --error=logs/scaling_%j.err

set -e

mkdir -p logs

cd /home/tolong/work/Attention-Residuals

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo ""

nvidia-smi | head -5
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
echo ""

bash run_scaling.sh

echo ""
echo "End: $(date)"

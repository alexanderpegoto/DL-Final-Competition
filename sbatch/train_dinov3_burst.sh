#!/bin/bash
#SBATCH --partition=a100_short     # check this exists on Burst (use `sinfo`)
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=2-00:00:00
#SBATCH --mem=256GB
#SBATCH --gres=gpu:2
#SBATCH --job-name=train_gpu2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ap9283@nyu.edu   # <-- your email
#SBATCH --output="logs/%x/%j.out"
#SBATCH --error="logs/%x/%j.err"

TRAIN_EXP_ID=${SLURM_JOB_ID}

echo "=================================================="
echo "Training Experiment ID: ${TRAIN_EXP_ID}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "SLURM_NTASKS=$SLURM_NTASKS"
echo "Start Time: $(date)"
echo "=================================================="

module load cuda/12.6          # adjust if Burst has a different version
source ~/.bashrc               # <- use your own .bashrc
conda activate ssl
cd /scratch/ap9283/deep_learning/DL-Final-Competition
mkdir -p logs/${SLURM_JOB_NAME}

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export HYDRA_FULL_ERROR=1
export PYTHONPATH=/scratch/ap9283/deep_learning/DL-Final-Competition:$PYTHONPATH
export TIMM_FUSED_ATTN=1
export TORCHRUN_PROC_NUM=2

START_TIME=$(date +%s)

torchrun --standalone --nproc_per_node=${TORCHRUN_PROC_NUM} scripts/train.py \
  distributed.enabled=true \
  experiment_name=${TRAIN_EXP_ID}

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
DAYS=$((ELAPSED/86400))
HOURS=$(((ELAPSED%86400)/3600))
MINUTES=$(((ELAPSED%3600)/60))
SECONDS=$((ELAPSED%60))

echo "=================================================="
echo "Training Duration: ${DAYS}d ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "End Time: $(date)"
echo "=================================================="

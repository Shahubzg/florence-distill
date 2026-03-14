#!/bin/bash
#SBATCH --job-name=distill_train
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/leonardo_work/IscrC_DEMOLLM/florence_distill/logs/%x_%j.out
#SBATCH --error=/leonardo_work/IscrC_DEMOLLM/florence_distill/logs/%x_%j.err

module load profile/deeplrn
module load python/3.11.7

source /leonardo_work/IscrC_DEMOLLM/florence_distill/env/venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /leonardo_work/IscrC_DEMOLLM/florence_distill/repo

python src/distill_train.py \
  --teacher_model_id  /leonardo_work/IscrC_DEMOLLM/florence_distill/models/florence-2-base \
  --pairs_csv         /leonardo_work/IscrC_DEMOLLM/florence_distill/outputs/results_baseline/pairs_preview.csv \
  --embeddings_npz    /leonardo_work/IscrC_DEMOLLM/florence_distill/outputs/results_baseline/embeddings.npz \
  --output_dir        /leonardo_work/IscrC_DEMOLLM/florence_distill/ckpts/distill_student \
  --batch_size        8 \
  --lr                1e-4 \
  --epochs            3 \
  --max_response_len  64 \
  --lambda_im         1.0 \
  --lambda_vl         1.0 \
  --lambda_vl_bbox    0.5 \
  --temperature       1.0 \
  --log_every         20

#!/bin/bash
#SBATCH --job-name=eval_student_vs_teacher
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/leonardo_work/IscrC_DEMOLLM/florence_distill/logs/%x_%j.out
#SBATCH --error=/leonardo_work/IscrC_DEMOLLM/florence_distill/logs/%x_%j.err

module load profile/deeplrn
module load python/3.11.7

source /leonardo_work/IscrC_DEMOLLM/florence_distill/env/venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /leonardo_work/IscrC_DEMOLLM/florence_distill/repo

CHECKPOINT=${CHECKPOINT:-/leonardo_work/IscrC_DEMOLLM/florence_distill/ckpts/distill_student/student_epoch3.pt}
OUTPUT_DIR=${OUTPUT_DIR:-/leonardo_work/IscrC_DEMOLLM/florence_distill/outputs/eval_student_vs_teacher}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-64}
LIMIT_SAMPLES=${LIMIT_SAMPLES:-0}
INCLUDE_BBOX_PROMPT_0=${INCLUDE_BBOX_PROMPT_0:-0}

EXTRA_ARGS=()
if [ "${INCLUDE_BBOX_PROMPT_0}" = "1" ]; then
  EXTRA_ARGS+=(--include_bbox_prompt_0)
fi

python src/eval_student_vs_teacher.py \
  --teacher_model_id  /leonardo_work/IscrC_DEMOLLM/florence_distill/models/florence-2-base \
  --pairs_csv         /leonardo_work/IscrC_DEMOLLM/florence_distill/outputs/results_baseline/pairs_preview.csv \
  --embeddings_npz    /leonardo_work/IscrC_DEMOLLM/florence_distill/outputs/results_baseline/embeddings.npz \
  --checkpoint        "${CHECKPOINT}" \
  --output_dir        "${OUTPUT_DIR}" \
  --batch_size        "${BATCH_SIZE}" \
  --max_response_len  "${MAX_RESPONSE_LEN}" \
  --limit_samples     "${LIMIT_SAMPLES}" \
  "${EXTRA_ARGS[@]}"

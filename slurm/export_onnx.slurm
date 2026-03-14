#!/bin/bash
#SBATCH --job-name=export_onnx
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/leonardo_work/IscrC_DEMOLLM/florence_distill/logs/%x_%j.out
#SBATCH --error=/leonardo_work/IscrC_DEMOLLM/florence_distill/logs/%x_%j.err

module load profile/deeplrn
module load python/3.11.7

source /leonardo_work/IscrC_DEMOLLM/florence_distill/env/venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /leonardo_work/IscrC_DEMOLLM/florence_distill/repo

# Optional: pass --checkpoint to load a trained student
python scripts/export_student_tflite.py \
  --teacher_model_id /leonardo_work/IscrC_DEMOLLM/florence_distill/models/florence-2-base \
  --output_dir       /leonardo_work/IscrC_DEMOLLM/florence_distill/ckpts/export \
  --onnx_name        tinyclip_student.onnx \
  --tflite_name      tinyclip_student.tflite \
  --height           224 \
  --width            224 \
  --seq_len          64 \
  --opset            14 \
  --device           cpu

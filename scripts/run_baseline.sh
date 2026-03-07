#!/bin/bash

set -e

IMAGE_ROOT="/leonardo_work/IscrC_DEMOLLM/florence_distill/data/coco/val2017"
CAPTIONS_JSON="/leonardo_work/IscrC_DEMOLLM/florence_distill/data/coco/annotations/captions_val2017.json"
OUTPUT_DIR="/leonardo_work/IscrC_DEMOLLM/florence_distill/outputs/results_baseline"

python src/teacher_baseline.py \
  --image_root ${IMAGE_ROOT} \
  --captions_json ${CAPTIONS_JSON} \
  --output_dir ${OUTPUT_DIR} \
  --num_samples 500 \
  --batch_size 4

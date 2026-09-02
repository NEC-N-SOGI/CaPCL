#!/usr/bin/env bash
# Test on the synthetic-shapes sample (3 tasks, 72 images).
# Requires src/capcl/directory_settings.json (see README) and one GPU.
set -eu

cd "$(dirname "$0")/../.."
MACHINE="${MACHINE:-local}"

uv run python experiments/sample/generate_synthetic_data.py --machine "${MACHINE}"

uv run python src/capcl/tasks/learner.py \
    --machine "${MACHINE}" \
    --task_config tasks/sample_synthetic.yaml \
    --optim_config optim/adamw_lora.json \
    --batch_size 4 \
    --num_train_epochs 2 \
    --logging_steps 1

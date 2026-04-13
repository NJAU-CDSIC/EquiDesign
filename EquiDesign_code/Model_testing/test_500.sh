#!/usr/bin/env bash
set -euo pipefail

python evaluate_ts500.py \
  --path_for_outputs "./" \
  --previous_checkpoint "../Model/model_weights/best_model.pt"


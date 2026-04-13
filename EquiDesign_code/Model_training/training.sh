#!/usr/bin/env bash
set -euo pipefail

JSONL_FILE="../../Datasets/CATH4.4/chain_set.jsonl"
SPLIT_FILE="../../Datasets/CATH4.4/chain_set_splits.json"

python train_equidesign.py \
  --jsonl_file "${JSONL_FILE}" \
  --split_file "${SPLIT_FILE}" \
  --path_for_outputs "./exp_equidesign" \
  --num_epochs 600 \
  --batch_size 10000 \
  --max_protein_length 10000 \
  --hidden_dim 128 \
  --num_encoder_layers 3 \
  --num_decoder_layers 3 \
  --num_neighbors 48 \
  --dropout 0.32 \
  --mixed_precision True \
  --equiformer_out_vector 0


#!/usr/bin/env bash
set -euo pipefail

# 1) Sync code from the original project
pwsh ./sync_from_project.ps1 || true

# 2) Adjust dataset paths as needed
JSONL_FILE="../Datasets/CATH4.4/chain_set.jsonl"
SPLIT_FILE="../Datasets/CATH4.4/chain_set_splits.json"

python ../EquiDesign_code/Model_training/train_equidesign.py \
  --jsonl_file "${JSONL_FILE}" \
  --split_file "${SPLIT_FILE}" \
  --path_for_outputs "./exp_equiformer" \
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

#!/usr/bin/env bash
set -euo pipefail

# 1) Sync code from the original project
pwsh ./sync_from_project.ps1 || true

# 2) Adjust dataset paths as needed
JSONL_FILE="../../data/chain_set.jsonl"
SPLIT_FILE="../../data/chain_set_splits.json"

python ../EquiDesign_code/training_equiformer.py \
  --jsonl_file "${JSONL_FILE}" \
  --split_file "${SPLIT_FILE}" \
  --path_for_outputs "./exp_equiformer" \
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

#!/usr/bin/env bash
set -euo pipefail

# Example: train EquiDesign (EquiFormer encoder) on CATH 4.4
# 1) Install deps:
#    pip install -r requirements.txt
# 2) Prepare dataset files (outside repo), then set paths below.

JSONL_FILE="../../data/chain_set.jsonl"
SPLIT_FILE="../../data/chain_set_splits.json"

python ../EquiDesign_code/training_equiformer.py \
  --jsonl_file "${JSONL_FILE}" \
  --split_file "${SPLIT_FILE}" \
  --path_for_outputs "./exp_equiformer" \
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


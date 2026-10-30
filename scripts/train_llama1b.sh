#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${COMMERGE_DATA:-data}
kind=${1:-lora}
declare -A samples=([snli]=10000 [mnli]=10000 [sick]=4439 [qnli]=8000 [rte]=2490 [scitail]=8000)
declare -A lora_epochs=([snli]=1 [mnli]=1 [sick]=2 [qnli]=1 [rte]=3 [scitail]=1)
declare -A full_epochs=([snli]=2 [mnli]=2 [sick]=3 [qnli]=3 [rte]=3 [scitail]=3)
for task in snli mnli sick qnli rte scitail; do
  if [ "$kind" == lora ]; then
    python scripts/train_llama_lora.py meta-llama/Llama-3.2-1B $task --out $DATA/checkpoints/llama32_1b_lora/$task \
      --samples ${samples[$task]} --epochs ${lora_epochs[$task]} --lr 1e-4 --amp --data-root $DATA
  else
    python scripts/train_llama_full.py $task --lrs 5e-6 1e-5 2e-5 --max-epochs ${full_epochs[$task]} --out $DATA/checkpoints/llama32_1b_full/$task
  fi
done

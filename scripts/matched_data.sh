#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
settings=("$@"); [ ${#settings[@]} -eq 0 ] && settings=(vit_b32_lora vit_l14_lora llama32_1b_lora)
for s in "${settings[@]}"; do
  python -m commerge.matched merge "$s"
  python -m commerge.matched finetune "$s"
done

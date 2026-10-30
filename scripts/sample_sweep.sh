#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
settings=("$@"); [ ${#settings[@]} -eq 0 ] && settings=(vit_b32_lora vit_l14_lora)
for s in "${settings[@]}"; do
  for n in 5 10 50 100 200 300 400 500; do for seed in 420 1 2; do
    python -m commerge.run experiments/$s/com.yaml data.calibration.num_samples=$n data.calibration.seed=$seed --out outputs/sample_sweep/$s/n$n
  done; done
done

#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
settings=("$@"); [ ${#settings[@]} -eq 0 ] && settings=(vit_b32_lora vit_l14_lora)
for s in "${settings[@]}"; do
  for mcs in false true; do for norm in false true; do for fc in none gram_offdiag; do
    for seed in 420 1 2; do
      python -m commerge.run experiments/$s/com.yaml method.chain=$mcs method.normalize_features=$norm method.task_weights=$fc \
        data.calibration.seed=$seed --out outputs/ablation/$s/mcs-${mcs}_norm-${norm}_fc-${fc}
    done
  done; done; done
done

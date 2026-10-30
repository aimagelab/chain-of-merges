#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
settings=("$@"); [ ${#settings[@]} -eq 0 ] && settings=(vit_b32_lora vit_l14_lora)
for s in "${settings[@]}"; do
  if [[ $s == llama* ]]; then
    declare -A variants=(
      [balanced_val]="data.calibration.split=val data.calibration.top_up=train"
      [random_val]="data.calibration.split=val data.calibration.top_up=train data.calibration.sampling=random"
      [balanced_train]="data.calibration.split=train"
      [random_train]="data.calibration.split=train data.calibration.sampling=random")
  else
    declare -A variants=(
      [balanced_val]="data.calibration.split=val"
      [random_val]="data.calibration.split=val data.calibration.sampling=random"
      [balanced_train]="data.calibration.split=train"
      [random_train]="data.calibration.split=train data.calibration.sampling=random")
  fi
  variants[noise0.01]="data.calibration.noise=0.01"
  variants[noise0.1]="data.calibration.noise=0.1"
  variants[noise1.0]="data.calibration.noise=1.0"
  variants[few_classes]="data.calibration.max_class_fraction=0.03"
  for v in "${!variants[@]}"; do for seed in 420 1 2; do
    python -m commerge.run experiments/$s/com.yaml ${variants[$v]} data.calibration.seed=$seed --out outputs/calibration_sensitivity/$s/$v
  done; done
  unset variants
done

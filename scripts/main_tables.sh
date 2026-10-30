#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
settings=("$@"); [ ${#settings[@]} -eq 0 ] && settings=(vit_b32_lora vit_l14_lora vit_b32_full vit_l14_full)
for s in "${settings[@]}"; do
  for cfg in configs/experiments/$s/*.yaml; do
    method=$(basename "$cfg" .yaml)
    case $method in
      com) for seed in 420 1 2; do python -m commerge.run experiments/$s/com.yaml data.calibration.seed=$seed --out outputs/main; done ;;
      dare_ties) if [[ $s == llama* ]]; then for ds in 421 422 423; do python -m commerge.run experiments/$s/dare_ties.yaml method.dare_seed=$ds --out outputs/main; done
                 else python -m commerge.run experiments/$s/dare_ties.yaml --out outputs/main; fi ;;
      *) python -m commerge.run experiments/$s/$method.yaml --out outputs/main ;;
    esac
  done
  python -m commerge.run experiments/$s/com.yaml method.name=zeroshot --out outputs/zeroshot
done

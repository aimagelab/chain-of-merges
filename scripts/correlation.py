import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from commerge.config import load_config
from commerge.run import evaluate_config, merge_delta, reference_accuracies, set_seed
from commerge.setup import build_world


def main():
    ap = argparse.ArgumentParser(description="Paper Fig. 6 / Fig. 10: correlation-based task importance vs performance-based importance.")
    ap.add_argument("setting", choices=["vit_b32_lora", "vit_l14_lora", "llama3_8b_lora"])
    ap.add_argument("--out", default="outputs/correlation")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    device, path = torch.device(a.device), f"experiments/{a.setting}/com.yaml"
    cfg = load_config(path)
    cfg["_config_path"] = path
    set_seed(cfg.get("seed", 420))
    world = build_world(cfg, device)
    recorder = []
    merge_delta(cfg, world, device, {"importance_recorder": recorder})
    importance = np.asarray(recorder).mean(0)
    zeroshot = evaluate_config({**cfg, "method": {"name": "zeroshot"}}, world, device)
    individual = reference_accuracies(cfg, world)
    gap = np.array([individual[t] - zeroshot[f"{t}/acc"] for t in world.tasks])
    result = {"tasks": world.tasks, "correlation_importance": importance.tolist(), "performance_gap": gap.tolist(),
              "matrices": len(recorder), "pearson": float(pearsonr(importance, gap)[0]), "spearman": float(spearmanr(importance, gap)[0])}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump(result, open(Path(a.out) / f"{a.setting}.json", "w"), indent=1)
    print(json.dumps({k: result[k] for k in ("tasks", "pearson", "spearman")}))


if __name__ == "__main__":
    main()

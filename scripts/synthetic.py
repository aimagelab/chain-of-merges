import argparse
import json
from pathlib import Path

import torch
import yaml

from commerge.config import CONFIG_DIR, load_config
from commerge.data.synthetic import synthetic_labelled
from commerge.models.clip import load_preprocess
from commerge.run import evaluate_config, evaluate_model, merge, set_seed
from commerge.setup import build_world, data_root


@torch.no_grad()
def synthetic_accuracy(world, model, images, device) -> float:
    model.to(device).eval()
    accs = []
    for i, (x, y) in enumerate(images):
        logits = torch.cat([world.logits_fn(i, model, (x[k:k + 50], y[k:k + 50])).float().cpu() for k in range(0, len(x), 50)])
        accs.append((logits.argmax(-1) == y).float().mean().item() * 100)
    return sum(accs) / len(accs)


def main():
    ap = argparse.ArgumentParser(description="Paper Tab. 7 / Fig. 5: calibration and hyper-parameter selection on Stable Diffusion images.")
    ap.add_argument("setting", choices=["vit_b32_lora", "vit_l14_lora"])
    ap.add_argument("--out", default="outputs/synthetic")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    device, protocol = torch.device(a.device), yaml.safe_load(open(CONFIG_DIR / "synthetic.yaml"))
    out = Path(a.out) / a.setting
    out.mkdir(parents=True, exist_ok=True)
    synthetic = ["data.calibration.source=synthetic"]
    regmean = [f"method.{k}={v}" for k, v in protocol["regmean"].items()]
    for method, overrides in (("com", synthetic), ("regmean", synthetic + regmean + ["data.calibration.num_samples=500"])):
        path = f"experiments/{a.setting}/{method}.yaml"
        cfg = load_config(path, overrides)
        cfg["_config_path"] = path
        set_seed(cfg.get("seed", 420))
        metrics = evaluate_config(cfg, build_world(cfg, device), device)
        json.dump(metrics, open(out / f"{method}.json", "w"), indent=1)
        print(f"{a.setting} {method} (synthetic calibration): {metrics['avg/norm_acc']:.2f}", flush=True)
    world = None
    for method, grids in protocol["methods"].items():
        path = f"experiments/{a.setting}/{method}.yaml"
        cfg = load_config(path)
        cfg["_config_path"] = path
        if world is None:
            set_seed(420)
            world = build_world(cfg, device)
            preprocess = load_preprocess(cfg["model"])
            images = [synthetic_labelled(data_root(cfg), t, preprocess) for t in cfg["model"]["tasks"]]
        trials = []
        for point in grids[a.setting]:
            set_seed(420)
            model = merge(load_config(path, [f"method.{k}={v}" for k, v in point.items()]), world, device)
            trials.append({"params": point, "synthetic_acc": synthetic_accuracy(world, model, images, device)})
            del model
            torch.cuda.empty_cache()
        best = max(trials, key=lambda t: t["synthetic_acc"])
        set_seed(420)
        test = evaluate_model(cfg, world, merge(load_config(path, [f"method.{k}={v}" for k, v in best["params"].items()]), world, device), "test")
        json.dump({"selected": best["params"], "trials": trials, "test": test}, open(out / f"{method}.json", "w"), indent=1)
        print(f"{a.setting} {method} (selected on synthetic images: {best['params']}): {test['avg/norm_acc']:.2f}", flush=True)


if __name__ == "__main__":
    main()

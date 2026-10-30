import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

from commerge.config import CONFIG_DIR, load_config
from commerge.run import evaluate_model, merge, set_seed
from commerge.search import coordinate_search
from commerge.setup import build_world

SETTINGS = {"vit_b32_lora": "vision", "vit_l14_lora": "vision", "llama32_1b_lora": "llama"}
NS, SEEDS = [10, 50, 100, 500], [420, 1, 2]


def protocol(setting: str) -> dict:
    return yaml.safe_load(open(CONFIG_DIR / "matched" / f"{SETTINGS[setting]}.yaml"))


def make_world(setting: str, n: int, seed: int, device):
    path = f"models/{setting}"
    cfg = load_config(path, [f"data.calibration.num_samples={n}", f"data.calibration.seed={seed}", "data.calibration.sampling=random"])
    cfg["_config_path"] = f"experiments/{setting}/matched.yaml"
    cfg["method"] = {"name": "zeroshot"}
    set_seed(seed)
    return cfg, build_world(cfg, device)


def default_params(method: str) -> dict:
    return dict(yaml.safe_load(open(CONFIG_DIR / "methods" / f"{method}.yaml"))["method"])


def merged_model(cfg, world, params, device):
    set_seed(420)
    return merge({**deepcopy(cfg), "method": dict(params)}, world, device)


def labels_of(batch):
    return batch[1] if isinstance(batch, (tuple, list)) else batch["labels"]


@torch.no_grad()
def accuracy_on_examples(world, model, device) -> float:
    model.eval().to(device)
    accs = []
    for i, batches in enumerate(world.calib):
        correct = total = 0
        for b in batches:
            correct += (world.logits_fn(i, model, b).argmax(-1).cpu() == labels_of(b)).sum().item()
            total += len(labels_of(b))
        accs.append(correct / total)
    return float(np.mean(accs))


def tune(cfg, world, method: str, grid: dict, device):
    base = default_params(method)
    space = {k: list(v) for k, v in grid.items()}

    def score(params):
        net = merged_model(cfg, world, {**base, **params}, device)
        value = accuracy_on_examples(world, net, device)
        del net
        torch.cuda.empty_cache()
        return value

    start = {k: base[k] for k in space} if all(k in base for k in space) else {k: v[0] for k, v in space.items()}
    best, history = coordinate_search(space, score, start)
    for _ in range(4):
        values = sorted(space["scale"])
        if best["scale"] == values[0]:
            new = round(values[0] * 0.5, 4)
        elif best["scale"] == values[-1]:
            new = round(values[-1] * 1.5, 4)
        else:
            break
        space["scale"].append(new)
        params = {**best, "scale": new}
        value = score(params)
        history.append({"params": params, "score": value})
        if value > max(h["score"] for h in history[:-1]):
            best = params
        else:
            break
    best = max(history, key=lambda h: (h["score"], h["params"].get("scale", 0)))["params"]
    return {**base, **best}, history


def result_path(out: Path, kind: str, setting: str, n: int, seed: int, method: str) -> Path:
    return out / setting / kind / f"n{n}_s{seed}_{method}.json"


def run_merges(setting, n, seed, methods, out: Path, device):
    proto = protocol(setting)
    cfg, world = make_world(setting, n, seed, device)
    for method in methods:
        path = result_path(out, "merge", setting, n, seed, method)
        if path.exists():
            continue
        start, history = time.time(), None
        if method == "zeroshot":
            params = {"name": "zeroshot"}
        elif method in proto["untuned"]:
            params = default_params(method)
        else:
            params, history = tune(cfg, world, method, proto["grids"][method], device)
        net = merged_model(cfg, world, params, device)
        record = {"setting": setting, "n": n, "seed": seed, "method": method, "params": params, "history": history,
                  "accuracy_on_examples": accuracy_on_examples(world, net, device), "test": evaluate_model(cfg, world, net, "test"),
                  "seconds": time.time() - start}
        path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(record, open(path, "w"), indent=1)
        print(f"merge {setting} n={n} seed={seed} {method}: {record['test']['avg/norm_acc']:.2f}", flush=True)
        del net
        torch.cuda.empty_cache()


def run_finetune(setting, n, seed, method, out: Path, device):
    from commerge.finetune import finetune
    path = result_path(out, "finetune", setting, n, seed, method)
    if path.exists():
        return
    if method == "zeroshot":
        params = {"name": "ta", "merging_type": "sum", "scale": 0.0}
    else:
        params = json.load(open(result_path(out, "merge", setting, n, seed, method)))["params"]
    cfg, world = make_world(setting, n, seed, device)
    record = finetune(cfg, world, merged_model(cfg, world, params, device), setting, seed, protocol(setting)["finetune"], device)
    record.update(setting=setting, n=n, seed=seed, method=method, merge_params=params)
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(record, open(path, "w"), indent=1)
    print(f"finetune {setting} n={n} seed={seed} {method}: lr {record['selected_lr']} -> {record['test']['avg/norm_acc']:.2f}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Merging vs fine-tuning with the same n examples per task (paper Tab. 8 / Fig. 4).")
    ap.add_argument("stage", choices=["merge", "finetune"])
    ap.add_argument("setting", choices=list(SETTINGS))
    ap.add_argument("--n", type=int, nargs="+", default=NS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--methods", nargs="+", default=None, help="default: zeroshot and every method of configs/matched")
    ap.add_argument("--out", default="outputs/matched")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    device, out = torch.device(a.device), Path(a.out)
    methods = a.methods or ["zeroshot"] + protocol(a.setting)["methods"]
    for n in a.n:
        for seed in a.seeds:
            if a.stage == "merge":
                run_merges(a.setting, n, seed, methods, out, device)
            else:
                for method in methods:
                    run_finetune(a.setting, n, seed, method, out, device)


if __name__ == "__main__":
    main()

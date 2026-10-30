import argparse
import json
import logging
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

import commerge.methods  # noqa: F401
from commerge.config import load_config
from commerge.eval.metrics import load_reference, normalized_report
from commerge.methods.base import MergeContext
from commerge.models.task_vectors import apply_task_vector
from commerge.registry import METHODS
from commerge.setup import World, build_world

log = logging.getLogger("commerge")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def measure_individual(world: World, split: str) -> Dict[str, float]:
    return {t: 100 * world.evaluate(world.task_model(i), i, split) for i, t in enumerate(world.tasks)}


def reference_accuracies(cfg: Dict[str, Any], world: World) -> Dict[str, float]:
    name = Path(cfg["_config_path"]).parent.name
    cache = os.path.join("outputs", "reference", f"{name}_{cfg['eval']['split']}.json")
    return load_reference(cfg["eval"]["reference_acc"], cache, lambda: measure_individual(world, cfg["eval"]["split"]))


def merge_delta(cfg: Dict[str, Any], world: World, device: torch.device, extras: Dict[str, Any] = None):
    ctx = MergeContext(
        pretrained=world.pretrained, task_vectors=world.task_vectors, cfg=dict(cfg["method"]), device=device,
        arch=world.arch, calib=world.calib, task_model=world.task_model,
        extras={"logits_fn": world.logits_fn, "cache_dir": cfg.get("cache_dir", os.path.join("outputs", "cache")),
                "task_names": world.tasks, "calibration": dict(cfg["data"]["calibration"]),
                "checkpoints": cfg["model"]["checkpoints"],
                "model_id": f"{cfg['model']['base'].replace('/', '_')}_{cfg['model']['finetune']}_n{cfg['data']['calibration']['num_samples']}",
                **(extras or {})},
    )
    return METHODS.get(cfg["method"]["name"])(ctx)


def merge(cfg: Dict[str, Any], world: World, device: torch.device, extras: Dict[str, Any] = None) -> torch.nn.Module:
    if cfg["method"]["name"] == "zeroshot":
        return deepcopy(world.pretrained)
    delta = merge_delta(cfg, world, device, extras)
    return apply_task_vector(deepcopy(world.pretrained), delta)


def evaluate_model(cfg: Dict[str, Any], world: World, model, split: str = None) -> Dict[str, float]:
    split = split or cfg["eval"]["split"]
    acc = {t: world.evaluate(model, i, split) for i, t in enumerate(world.tasks)}
    return normalized_report(acc, reference_accuracies({**cfg, "eval": {**cfg["eval"], "split": split}}, world), world.tasks)


def evaluate_config(cfg: Dict[str, Any], world: World, device: torch.device) -> Dict[str, float]:
    split = cfg["eval"]["split"]
    reference = reference_accuracies(cfg, world)
    if cfg["method"]["name"] == "individual":
        acc = {t: a / 100 for t, a in measure_individual(world, split).items()}
    else:
        start = time.time()
        model = merge(cfg, world, device)
        log.info("merged in %.1fs", time.time() - start)
        acc = {t: world.evaluate(model, i, split) for i, t in enumerate(world.tasks)}
    return normalized_report(acc, reference, world.tasks)


def run(cfg: Dict[str, Any], device: torch.device) -> Dict[str, float]:
    set_seed(cfg.get("seed", 420))
    return evaluate_config(cfg, build_world(cfg, device), device)


def output_dir(root: str, config: str) -> Path:
    out = Path(root) / Path(config).with_suffix("").as_posix().replace("/", "_") / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    return out


def setup_logging(out_dir: Path) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "log.txt")])


def main(argv=None) -> Dict[str, float]:
    parser = argparse.ArgumentParser(description="Merge the models of one experiment config and evaluate the result.")
    parser.add_argument("config")
    parser.add_argument("overrides", nargs="*", help="key.sub=value")
    parser.add_argument("--out", default="outputs/runs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    cfg["_config_path"] = args.config
    out_dir = output_dir(args.out, args.config)
    setup_logging(out_dir)
    yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}, open(out_dir / "config.yaml", "w"))
    metrics = run(cfg, torch.device(args.device))
    json.dump(metrics, open(out_dir / "metrics.json", "w"), indent=1)
    log.info("avg/norm_acc %.2f  avg/acc %.2f  -> %s", metrics["avg/norm_acc"], metrics["avg/acc"], out_dir)
    return metrics


if __name__ == "__main__":
    main()

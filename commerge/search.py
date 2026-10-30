import argparse
import itertools
import json
import logging
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import torch
import yaml

import commerge.methods  # noqa: F401
from commerge.config import apply_override, load_config
from commerge.run import evaluate_config, set_seed
from commerge.setup import build_world

log = logging.getLogger("commerge")

Space = Dict[str, List[Any]]
Params = Dict[str, Any]
Score = Callable[[Params], float]


def parse_space(items: List[str]) -> Space:
    space = {}
    for item in items:
        key, values = item.split("=", 1)
        space[key] = [yaml.safe_load(v) for v in values.split(",")]
    return space


def grid_search(space: Space, score: Score) -> Tuple[Params, List[dict]]:
    history = []
    for values in itertools.product(*space.values()):
        params = dict(zip(space, values))
        history.append({"params": params, "score": score(params)})
    return max(history, key=lambda h: h["score"])["params"], history


def coordinate_search(space: Space, score: Score, start: Params) -> Tuple[Params, List[dict]]:
    current, history, cache = dict(start), [], {}

    def evaluate(params: Params) -> float:
        key = json.dumps(params, sort_keys=True)
        if key not in cache:
            cache[key] = score(dict(params))
            history.append({"params": dict(params), "score": cache[key]})
        return cache[key]

    for name, candidates in space.items():
        best_value, best_score = None, float("-inf")
        for value in candidates:
            s = evaluate({**current, name: value})
            if s > best_score:
                best_value, best_score = value, s
            else:
                break
        current[name] = best_value
    best = max(history, key=lambda h: h["score"])
    return dict(best["params"]), history


def _with(cfg: Dict[str, Any], params: Params) -> Dict[str, Any]:
    for key, value in params.items():
        cfg = apply_override(cfg, f"{key}={json.dumps(value)}")
    return cfg


def search(cfg: Dict[str, Any], space: Space, mode: str, device: torch.device) -> Dict[str, Any]:
    set_seed(cfg.get("seed", 420))
    world = build_world(cfg, device)

    def evaluate(params: Params, split: str) -> Dict[str, float]:
        trial = _with(deepcopy(cfg), params)
        trial["eval"]["split"] = split
        set_seed(trial.get("seed", 420))
        return evaluate_config(trial, world, device)

    def val_score(params: Params) -> float:
        value = evaluate(params, "val")["avg/norm_acc"]
        log.info("%s -> val %.2f", params, value)
        return value

    if mode == "grid":
        best, history = grid_search(space, val_score)
    elif mode == "coordinate":
        start = {}
        for key in space:
            node = cfg
            try:
                for part in key.split("."):
                    node = node[part]
                start[key] = node
            except (KeyError, TypeError):
                start[key] = space[key][0]
        best, history = coordinate_search(space, val_score, start)
    else:
        raise ValueError(f"unknown search mode '{mode}'")
    test = evaluate(best, "test")
    return {"mode": mode, "best_params": best, "val": history, "test": test}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Grid or coordinate search of a config's hyper-parameters on the validation split; the winner is evaluated once on test.")
    parser.add_argument("config")
    parser.add_argument("params", nargs="*", help="key=v1,v2,... axes (default: the config's `search:` block); "
                                                  "plain key=value are fixed overrides")
    parser.add_argument("--mode", choices=["grid", "coordinate"], default=None)
    parser.add_argument("--out", default="outputs/search")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    fixed = [p for p in args.params if "," not in p.split("=", 1)[1]]
    cfg = load_config(args.config, fixed)
    cfg["_config_path"] = args.config
    axes = [p for p in args.params if p not in fixed]
    if axes:
        space, mode = parse_space(axes), args.mode or "grid"
    else:
        declared = cfg.get("search")
        if not declared:
            parser.error("no search axes on the command line and no `search:` block in the config")
        space, mode = {k: list(v) for k, v in declared["params"].items()}, args.mode or declared.get("mode", "grid")

    out_dir = Path(args.out) / Path(args.config).with_suffix("").as_posix().replace("/", "_") / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out_dir / "log.txt")])
    result = search(cfg, space, mode, torch.device(args.device))
    result["best"] = yaml.safe_load(yaml.safe_dump(_with({}, result["best_params"])))
    json.dump(result, open(out_dir / "search.json", "w"), indent=1)
    log.info("best %s -> val-selected test avg norm acc %.2f", result["best_params"], result["test"]["avg/norm_acc"])


if __name__ == "__main__":
    main()

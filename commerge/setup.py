import math
import os
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import torch
from torch.utils.data import DataLoader

from commerge.data.calibration import calibration_dataset
from commerge.models.arch import ArchAdapter
from commerge.models.task_vectors import TaskVector, apply_task_vector
from commerge.prepare import ensure_asset, ensure_vision


@dataclass
class World:
    pretrained: torch.nn.Module
    arch: ArchAdapter
    task_vectors: List[TaskVector]
    tasks: List[str]
    task_data: List[Any]
    calib: List[Any]
    evaluate: Callable[[torch.nn.Module, int, str], float]
    logits_fn: Callable[[int, torch.nn.Module, Any], torch.Tensor]
    heads: List[torch.Tensor]

    def task_model(self, i: int) -> torch.nn.Module:
        return apply_task_vector(deepcopy(self.pretrained), self.task_vectors[i])


def data_root(cfg) -> str:
    return os.environ.get("COMMERGE_DATA", cfg["data"]["root"])


def build_world(cfg: Dict[str, Any], device: torch.device) -> World:
    family = cfg["model"]["family"]
    if family == "clip":
        return _build_clip_world(cfg, device)
    if family == "llama":
        return _build_llama_world(cfg, device)
    raise ValueError(f"unknown model family '{family}'")


def vision_calibration(tasks, names, cal, root, preprocess, workers=8):
    from commerge.data.synthetic import synthetic_batches
    from commerge.data.vision import NUM_CLASSES
    calib = []
    for name, task in zip(names, tasks):
        if cal.get("source") == "synthetic":
            calib.append(synthetic_batches(root, name, preprocess, cal["batch_size"]))
        elif cal["num_samples"] == "all":
            calib.append(DataLoader(task.split(cal["split"]), batch_size=cal["batch_size"], shuffle=False, num_workers=workers))
        else:
            subset = calibration_dataset(task, cal, random.Random(cal["seed"]), range(NUM_CLASSES[name]))
            calib.append([(x, y) for x, y in DataLoader(subset, batch_size=cal["batch_size"], shuffle=False, num_workers=workers)])
    if cal.get("noise"):
        generator = torch.Generator().manual_seed(cal["seed"])
        calib = [[(x + cal["noise"] * torch.randn(x.shape, generator=generator), y) for x, y in batches] for batches in calib]
    return calib


def nli_calibration(tasks, cal, batch_size):
    calib = []
    for i, task in enumerate(tasks):
        if cal["num_samples"] == "all":
            calib.append(DataLoader(task.split(cal["split"]), batch_size=batch_size, shuffle=False, collate_fn=task.collate_fn))
            continue
        classes = sorted(set(int(lab) for lab in task.labels(cal["split"]))) if cal.get("max_class_fraction") else None
        subset = calibration_dataset(task, cal, random.Random(cal["seed"] * 1000 + i), classes)
        calib.append(list(DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=task.collate_fn)))
    return calib


def _build_clip_world(cfg, device) -> World:
    from commerge.data.vision import load_vision_task
    from commerge.eval.vision import evaluate_vision
    from commerge.models import clip

    mcfg, dcfg, ecfg, cal = cfg["model"], cfg["data"], cfg["eval"], cfg["data"]["calibration"]
    root, workers = data_root(cfg), dcfg.get("num_workers", 8)
    ensure_vision(root, mcfg["tasks"])
    if cal.get("source") == "synthetic":
        ensure_asset(root, "synthetic")
    pretrained = clip.load_pretrained(mcfg)
    task_vectors = clip.load_task_vectors(mcfg, pretrained)
    preprocess = clip.load_preprocess(mcfg)
    tasks = [load_vision_task(t, root, preprocess) for t in mcfg["tasks"]]
    heads = [clip.load_head(root, mcfg["head_dir"], t) for t in mcfg["tasks"]]

    calib = vision_calibration(tasks, mcfg["tasks"], cal, root, preprocess, workers)

    def evaluate(model, i, split):
        return evaluate_vision(model, heads[i], tasks[i].split(split), device, ecfg.get("batch_size", 128), workers)

    def logits_fn(i, model, batch):
        emb = model(batch[0].to(device))
        return 100.0 * (emb / emb.norm(dim=-1, keepdim=True)) @ heads[i].to(device).T

    return World(pretrained, clip.ClipArch(pretrained), task_vectors, list(mcfg["tasks"]), tasks, calib, evaluate, logits_fn, heads)


def embedding_rms(model, calib) -> float:
    emb = model.model.embed_tokens.weight.detach().float().cpu()
    total, count = 0.0, 0
    for batches in calib:
        for b in batches:
            mask = b["attention_mask"].bool()
            total += float((emb[b["input_ids"]][mask] ** 2).sum())
            count += int(mask.sum()) * emb.shape[1]
    return math.sqrt(total / count)


def _build_llama_world(cfg, device) -> World:
    from commerge.data.nli import load_nli_task
    from commerge.eval.nli import evaluate_nli, nli_logits
    from commerge.models import llama

    mcfg, dcfg, cal, root = cfg["model"], cfg["data"], cfg["data"]["calibration"], data_root(cfg)
    if any(c.startswith("checkpoints/llama32_1b_lora/") for c in mcfg["checkpoints"]):
        ensure_asset(root, "llama1b_adapters")
    mcfg = {**mcfg, "checkpoints": [os.path.join(root, c) if os.path.isdir(os.path.join(root, c)) else c for c in mcfg["checkpoints"]]}
    pretrained = llama.load_pretrained(mcfg)
    task_vectors = llama.load_task_vectors(mcfg, pretrained)
    heads = llama.load_task_heads(mcfg)
    tokenizer = llama.load_tokenizer(mcfg)
    tasks = [load_nli_task(t, tokenizer, root=root) for t in mcfg["tasks"]]

    calib = nli_calibration(tasks, cal, dcfg["batch_size"])
    arch = llama.LlamaArch()
    if cal.get("noise"):
        arch = llama.LlamaArch(noise_std=cal["noise"] * embedding_rms(pretrained, calib), noise_seed=1000 * cal["seed"] + 7)

    def evaluate(model, i, split):
        return evaluate_nli(model, heads[i], tasks[i], batch_size=dcfg["batch_size"], device=device, split=split)

    def logits_fn(i, model, batch):
        return nli_logits(model, heads[i], tasks[i], batch, device)

    return World(pretrained, arch, task_vectors, list(mcfg["tasks"]), tasks, calib, evaluate, logits_fn, heads)

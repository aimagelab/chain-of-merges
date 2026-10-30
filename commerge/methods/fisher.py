import os
from typing import Callable, Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F

from commerge.methods.base import MergeContext
from commerge.methods.ops import average_others, task_weights
from commerge.models.task_vectors import TaskVector, matrix_keys
from commerge.registry import METHODS


def _to(x, device):
    return x.to(device) if torch.is_tensor(x) else {k: v.to(device) for k, v in x.items()}


def diag_fisher(model: torch.nn.Module, batches: Iterable, logits_fn: Callable, keys: List[str],
                micro_batch: int = 16) -> Tuple[Dict[str, torch.Tensor], int]:
    params = dict(model.named_parameters())
    modules = {k: model.get_submodule(k.rsplit(".", 1)[0]) for k in keys}
    saved = {k: p.requires_grad for k, p in params.items()}
    for p in params.values():
        p.requires_grad_(False)
    for k in keys:
        params[k].requires_grad_(True)
    device = params[keys[0]].device
    fisher, n = {k: torch.zeros_like(params[k], dtype=torch.float32) for k in keys}, 0
    try:
        for inputs, labels in batches:
            seen, handles = {}, []
            for k, module in modules.items():
                handles.append(module.register_forward_hook(
                    lambda mod, inp, out, k=k: seen.__setitem__(k, (inp[0].detach(), out))))
            try:
                y = labels.to(device)
                logits = logits_fn(model, (_to(inputs, device), y))
            finally:
                for h in handles:
                    h.remove()
            log_p = F.log_softmax(logits.float(), dim=-1).gather(1, y.unsqueeze(1)).squeeze(1)
            objective = (log_p.detach().exp().sqrt() * log_p).sum()
            grads = torch.autograd.grad(objective, [seen[k][1] for k in keys])
            for k, g in zip(keys, grads):
                x = seen[k][0]
                for start in range(0, len(y), micro_batch):
                    gb, xb = g[start:start + micro_batch].float(), x[start:start + micro_batch].float()
                    per_sample = (torch.einsum("bto,bti->boi", gb, xb) if gb.dim() == 3
                                  else torch.einsum("bo,bi->boi", gb, xb))
                    fisher[k] += per_sample.pow(2).sum(0)
            n += len(y)
    finally:
        for k, p in params.items():
            p.requires_grad_(saved[k])
    return fisher, n


def _num_samples(calib) -> int:
    return len(calib.dataset) if hasattr(calib, "dataset") else sum(len(labels) for _, labels in calib)


def task_fishers(ctx: MergeContext, keys: List[str]) -> List[Dict[str, torch.Tensor]]:
    logits_fn, cache_dir = ctx.extras["logits_fn"], ctx.extras.get("cache_dir")
    normalize = ctx.cfg.get("fisher_normalize", "per_task")
    if normalize not in ("per_task", "none"):
        raise ValueError(f"fisher_normalize must be 'per_task' or 'none', got '{normalize}'")
    names = ctx.extras.get("task_names", list(range(ctx.num_tasks)))
    out = []
    for t in range(ctx.num_tasks):
        n_samples = _num_samples(ctx.calib[t])
        path, raw, n = None, None, None
        if cache_dir:
            path = os.path.join(cache_dir, f"fisher_raw_{ctx.extras.get('model_id', 'model')}_{names[t]}_{n_samples}.pt")
            if os.path.exists(path):
                cached = torch.load(path, map_location="cpu", mmap=True)
                if set(keys) <= set(cached["fisher"]):
                    raw, n = {k: cached["fisher"][k] for k in keys}, cached["n"]
        if raw is None:
            model = ctx.task_model(t).to(ctx.device).eval()
            raw, n = diag_fisher(model, ctx.calib[t], lambda m, b, t=t: logits_fn(t, m, b), keys)
            raw = {k: f.cpu() for k, f in raw.items()}
            del model
            if path:
                os.makedirs(cache_dir, exist_ok=True)
                torch.save({"fisher": raw, "n": n}, path)
                raw = torch.load(path, map_location="cpu", mmap=True)["fisher"]
        out.append({k: (f / n if normalize == "per_task" else f) for k, f in raw.items()})
    return out


def linear_keys(ctx: MergeContext) -> List[str]:
    return [k for k in matrix_keys(ctx.task_vectors)
            if isinstance(ctx.pretrained.get_submodule(k.rsplit(".", 1)[0]), torch.nn.Linear)]


@METHODS.register("fisher")
def merge(ctx: MergeContext) -> TaskVector:
    keys, weights = linear_keys(ctx), task_weights(ctx)
    fishers = task_fishers(ctx, keys)
    out = {}
    for k in keys:
        num = sum(w * f[k] * tv[k].float() for w, f, tv in zip(weights, fishers, ctx.task_vectors))
        den = sum(f[k] for f in fishers).clamp(min=1e-30)
        out[k] = num / den
    out.update(average_others(ctx.task_vectors, keys))
    return out

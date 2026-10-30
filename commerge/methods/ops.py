import re
from typing import List, Optional

import torch

from commerge.models.task_vectors import TaskVector


def common_keys(task_vectors: List[TaskVector]) -> List[str]:
    keys = set(task_vectors[0])
    for tv in task_vectors[1:]:
        keys &= set(tv)
    return sorted(keys)


def task_weights(ctx, key: str = "scale", default: float = 1.0) -> List[float]:
    if ctx.scaling is not None:
        assert len(ctx.scaling) == ctx.num_tasks, "ctx.scaling needs one coefficient per task"
        return [float(w) for w in ctx.scaling]
    return [float(ctx.cfg.get(key, default))] * ctx.num_tasks


def average_others(task_vectors: List[TaskVector], skip: List[str]) -> TaskVector:
    skip = set(skip)
    return {k: torch.stack([tv[k].float() for tv in task_vectors]).mean(0)
            for k in common_keys(task_vectors) if k not in skip}


def layer_index(name: str) -> Optional[int]:
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def stack_flat(task_vectors: List[TaskVector], keys: List[str], device="cpu") -> torch.Tensor:
    dim = sum(task_vectors[0][k].numel() for k in keys)
    stack = torch.empty(len(task_vectors), dim)
    for row, tv in zip(stack, task_vectors):
        offset = 0
        for k in keys:
            n = tv[k].numel()
            row[offset:offset + n] = tv[k].reshape(-1)
            offset += n
    return stack.to(device)


def topk_prune_(stack: torch.Tensor, frac: float) -> torch.Tensor:
    if frac >= 1:
        return stack
    d = stack.shape[1]
    keep = max(1, int(d * frac))
    for row in stack:
        thr = row.abs().kthvalue(d - keep + 1).values
        row.mul_(row.abs() >= thr)
    return stack


def elect_sign(stack: torch.Tensor) -> torch.Tensor:
    sign = torch.sign(stack.sum(0))
    sign[sign == 0] = torch.sign(sign.sum())
    return sign


def disjoint_mean(stack: torch.Tensor, sign: torch.Tensor, weights: Optional[List[float]] = None,
                  merging_type: str = "mean", chunk: int = 1 << 24, device=None) -> torch.Tensor:
    if merging_type not in ("mean", "sum"):
        raise ValueError(f"merging_type must be 'mean' or 'sum', got '{merging_type}'")
    dev = device or stack.device
    w = None if weights is None else torch.tensor(weights, dtype=stack.dtype, device=dev).unsqueeze(1)
    out = torch.empty(stack.shape[1], dtype=stack.dtype, device=stack.device)
    for start in range(0, stack.shape[1], chunk):
        part, part_sign = stack[:, start:start + chunk].to(dev), sign[start:start + chunk].to(dev)
        agree = torch.where(part_sign.unsqueeze(0) > 0, part > 0, part < 0)
        selected = part * agree
        if w is not None:
            selected = selected * w
        total = selected.sum(0)
        total = total if merging_type == "sum" else total / (agree & (part != 0)).sum(0).clamp(min=1)
        out[start:start + chunk] = total.to(out.device)
    return out


def dare_(stack: torch.Tensor, p: float, generator: torch.Generator) -> torch.Tensor:
    for row in stack:
        keep = torch.rand(row.shape, generator=generator) >= p
        row.mul_(keep.to(row.device)).div_(1.0 - p)
    return stack


def ties_merge(stack: torch.Tensor, weights: List[float], topk: float = 1.0, merging_type: str = "mean",
               dare_p: float = 0.0, seed: int = 0, device=None) -> torch.Tensor:
    if dare_p > 0:
        dare_(stack, dare_p, torch.Generator().manual_seed(seed))
    topk_prune_(stack, topk)
    return disjoint_mean(stack, elect_sign(stack), weights, merging_type, device=device)


def ties_merge_streaming(task_vectors: List[TaskVector], keys: List[str], weights: List[float], topk: float = 1.0,
                         merging_type: str = "mean", device="cpu") -> TaskVector:
    dim = sum(task_vectors[0][k].numel() for k in keys)
    keep = max(1, int(dim * topk))
    thresholds = []
    for tv in task_vectors:
        if topk >= 1:
            thresholds.append(0.0)
            continue
        row = torch.cat([tv[k].reshape(-1).float() for k in keys]).abs_()
        thresholds.append(row.kthvalue(dim - keep + 1).values.item())
        del row

    def pruned(key: str) -> torch.Tensor:
        stack = torch.stack([tv[key].reshape(-1).float() for tv in task_vectors]).to(device)
        thr = torch.tensor(thresholds, device=stack.device).unsqueeze(1)
        return stack * (stack.abs() >= thr) if topk < 1 else stack

    sign_mass = 0.0
    for key in keys:
        sign_mass += torch.sign(pruned(key).sum(0)).sum().item()
    fallback = 1.0 if sign_mass > 0 else (-1.0 if sign_mass < 0 else 0.0)
    out = {}
    for key in keys:
        stack = pruned(key)
        sign = torch.sign(stack.sum(0))
        sign[sign == 0] = fallback
        out[key] = disjoint_mean(stack, sign, weights, merging_type).cpu().view_as(task_vectors[0][key])
    return out

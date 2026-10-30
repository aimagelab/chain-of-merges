from typing import Dict, Iterable, List

import torch
import torch.nn as nn

TaskVector = Dict[str, torch.Tensor]


def apply_task_vector(model: nn.Module, delta: TaskVector, scale: float = 1.0) -> nn.Module:
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in delta.items():
            params[name].add_(value.to(device=params[name].device, dtype=params[name].dtype), alpha=scale)
    return model


def mean_task_vector(task_vectors: List[TaskVector], keys: Iterable[str] = None) -> TaskVector:
    keys = list(task_vectors[0]) if keys is None else list(keys)
    return {k: torch.stack([tv[k].float() for tv in task_vectors]).mean(0) for k in keys}


def matrix_keys(task_vectors: List[TaskVector]) -> List[str]:
    common = set(task_vectors[0])
    for tv in task_vectors[1:]:
        common &= set(tv)
    return sorted(k for k in common if task_vectors[0][k].ndim == 2)


def flatten(tv: TaskVector, keys: List[str]) -> torch.Tensor:
    return torch.cat([tv[k].reshape(-1) for k in keys])


def unflatten(vec: torch.Tensor, like: TaskVector, keys: List[str]) -> TaskVector:
    out, start = {}, 0
    for k in keys:
        n = like[k].numel()
        out[k] = vec[start:start + n].view_as(like[k]).clone()
        start += n
    return out

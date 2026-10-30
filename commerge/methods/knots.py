from typing import Dict, List, Tuple

import torch

from commerge.methods.base import MergeContext
from commerge.methods.ops import average_others, stack_flat, task_weights, ties_merge
from commerge.models.task_vectors import TaskVector, matrix_keys, unflatten
from commerge.registry import METHODS

MIN_SINGULAR_VALUE = 1e-5


def knots_decompose(task_vectors: List[TaskVector], keys: List[str], device="cpu"
                    ) -> Tuple[Dict[str, torch.Tensor], List[TaskVector]]:
    bases, coeffs = {}, [{} for _ in task_vectors]
    for key in keys:
        in_dim = task_vectors[0][key].shape[1]
        wide = torch.cat([tv[key].to(device=device, dtype=torch.float64) for tv in task_vectors], dim=1)
        u, s, vh = torch.linalg.svd(wide, full_matrices=False)
        supported = s > MIN_SINGULAR_VALUE
        u, s, vh = u[:, supported], s[supported], vh[supported]
        bases[key] = u.float().cpu()
        for t, block in enumerate(torch.split(s.unsqueeze(1) * vh, in_dim, dim=1)):
            coeffs[t][key] = block.float().cpu()
    return bases, coeffs


def _knots(ctx: MergeContext) -> TaskVector:
    cfg, keys = ctx.cfg, matrix_keys(ctx.task_vectors)
    bases, coeffs = knots_decompose(ctx.task_vectors, keys, ctx.device)
    stack = stack_flat(coeffs, keys, ctx.device)
    merged = ties_merge(stack, task_weights(ctx), topk=float(cfg.get("topk", 1.0)),
                        merging_type=cfg.get("merging_type", "mean"),
                        dare_p=float(cfg.get("dare_p", 0.0)), seed=int(cfg.get("dare_seed", 0)))
    merged = unflatten(merged.cpu(), coeffs[0], keys)
    out = {k: bases[k] @ merged[k] for k in keys}
    out.update(average_others(ctx.task_vectors, keys))
    return out


METHODS.register("knots_ties")(_knots)
